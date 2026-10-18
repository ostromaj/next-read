"""Next Read model lab.

Tests every candidate model the same honest way (walk-forward: each week is predicted using only
games played before it, and settings are tuned only on earlier weeks), then picks this week's model.

Candidates
  naive        season target share only (the bar every model must clear)
  current      situation-aware shares + live in-game updating (what the site ran before the lab)
  same_day     current + folds in games that finished earlier the same day (e.g. 1:00 games before 4:25)
  pass_rate    current + how pass-heavy the team is in this game vs. its season rate
  pace         current + how many plays per drive the team is running in this game
  rate_pace    current + both of the above
  ml_logistic  machine learning: logistic regression trained on every past player-drive
  ml_boosted   machine learning: gradient-boosted trees on the same features (tested, not on the site yet)

Selection rule: a challenger replaces the model already on the site only if (1) it beats it in at least 90% of
bootstrap resamples of the test drives, so noise can't flip it, and (2) it cuts the Brier score by at least 0.1%,
so a real-but-trivial gain doesn't either. Among challengers that pass both, the lowest Brier score wins.

Usage: python3 -I lab.py <data_dir> <out_dir>
  data_dir needs pbp_2024/2025/2026.csv.gz and rosters_2026.csv from nflverse.
  Writes out_dir/model.json, leaderboard.json, backtest_summary.json, selection.json
"""
import itertools, json, math, os, sys, time
from datetime import date
import numpy as np
import pandas as pd

DATA, OUT = sys.argv[1], sys.argv[2]
SEASON = 2026
DECAY, K_CTX, A_SHARE, K_CATCH = 0.85, 40.0, 15.0, 15.0
MAX_PLAYERS = 12
SWITCH_CONFIDENCE = 0.90
MIN_GAIN_PCT = 0.10   # and must cut the Brier score by at least this much (relative %), so tiny gains never trigger a switch
COLS = ["season", "season_type", "week", "game_id", "game_date", "start_time", "posteam", "defteam", "fixed_drive",
        "play_id", "down", "ydstogo", "yardline_100", "score_differential", "qtr", "half_seconds_remaining",
        "pass_attempt", "receiver_player_id", "receiver_player_name", "complete_pass", "xpass", "play_type"]
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- data
def load(year):
    d = pd.read_csv(f"{DATA}/pbp_{year}.csv.gz", usecols=COLS, low_memory=False)
    d = d[d.season_type == "REG"].copy()
    d["kick"] = pd.to_datetime(d.start_time, format="%m/%d/%y, %H:%M:%S", errors="coerce")
    d["kick"] = d.groupby("game_id").kick.transform("min").fillna(pd.to_datetime(d.game_date))
    return d


def fbk(y): return np.select([y > 80, y > 60, y > 40, y > 20], ["own0-20", "own20-40", "mid", "opp20-40"], "redzone")
def dbk(d): return np.select([d == 1, d == 2], ["1", "2"], "3+")
def sbk(s): return np.select([s <= -9, s < 0, s == 0, s <= 8], ["trail9", "trail", "tied", "lead"], "lead9")
def tbk(t): return np.where(t <= 120, "2min", "normal")


hist = pd.concat([load(y) for y in (2024, 2025, SEASON)])
scrim = hist[hist.play_type.isin(["pass", "run"]) & hist.down.notna()].copy()
scrim["tgt"] = scrim.receiver_player_id.notna().astype(int)
scrim = scrim.sort_values(["game_id", "fixed_drive", "play_id"])
scrim["rem_tgt"] = scrim.iloc[::-1].groupby(["game_id", "fixed_drive"]).tgt.cumsum().iloc[::-1]
scrim["key"] = fbk(scrim.yardline_100) + "|" + dbk(scrim.down) + "|" + sbk(scrim.score_differential) + "|" + tbk(scrim.half_seconds_remaining)
scrim["fd"] = fbk(scrim.yardline_100) + "|" + dbk(scrim.down)
p26 = hist[hist.season == SEASON].copy()
roster = pd.read_csv(f"{DATA}/rosters_2026.csv", low_memory=False)
MAX_WEEK = int(p26.week.max())
log("loaded", len(hist), "plays; 2026 through week", MAX_WEEK)

_vol_cache = {}
def build_volume(cutoff):
    if cutoff in _vol_cache: return _vol_cache[cutoff]
    s = scrim[scrim.kick < cutoff]
    league = float(s.rem_tgt.mean())
    fd = s.groupby("fd").rem_tgt.mean()
    v = s.groupby(["key", "fd"]).rem_tgt.agg(["mean", "count"]).reset_index()
    v["value"] = (v["mean"] * v["count"] + v.fd.map(fd) * 40) / (v["count"] + 40)
    out = ({k: float(x) for k, x in zip(v.key, v.value)}, league)
    _vol_cache[cutoff] = out
    return out


INACTIVE = {"RES", "CUT", "RET", "DEV", "EXE"}
def build_state(cutoff, roster_week):
    """Everything the models know before `cutoff`: player shares, splits, catch rates, team tendencies."""
    cur = p26[p26.kick < cutoff]
    if cur.empty: return None
    vol, league = build_volume(cutoff)
    games = cur.groupby(["posteam", "game_id"]).kick.first().reset_index()
    games["gidx"] = games.groupby("posteam").kick.rank(ascending=False, method="first") - 1
    gidx = {(t, g): int(i) for t, g, i in zip(games.posteam, games.game_id, games.gidx)}
    lat = roster[roster.week <= max(1, roster_week)].sort_values("week").groupby("gsis_id").tail(1).set_index("gsis_id")
    passes = cur[cur.receiver_player_id.notna()].copy()
    passes["w"] = [DECAY ** gidx[(t, g)] for t, g in zip(passes.posteam, passes.game_id)]
    ctxs = {"rz": passes.yardline_100 <= 20, "3d": passes.down >= 3, "trail": passes.score_differential <= -9,
            "2min": passes.half_seconds_remaining <= 120, "lead": passes.score_differential >= 9}
    for c, m in ctxs.items(): passes["c_" + c] = m
    passes["pos"] = passes.receiver_player_id.map(lat.position).fillna("WR")
    pos_cr = passes.groupby("pos").complete_pass.mean().to_dict()
    sc = cur[cur.play_type.isin(["pass", "run"])]
    dr = sc.groupby(["posteam", "game_id", "fixed_drive"]).size().reset_index(name="snaps")
    dr["w"] = [DECAY ** gidx[(t, g)] for t, g in zip(dr.posteam, dr.game_id)]
    tdw, ttw = dr.groupby("posteam").w.sum(), passes.groupby("posteam").w.sum()
    league_tpd = ttw.sum() / tdw.sum()
    pr = sc.groupby("posteam").pass_attempt.mean().to_dict()
    ppd = dr.groupby("posteam").snaps.mean().to_dict()
    xp = sc[sc.xpass.notna()].groupby("posteam").apply(lambda d: (d.pass_attempt.mean() - d.xpass.mean()) * 100, include_groups=False).to_dict()
    teams = {}
    for team, tp in passes.groupby("posteam"):
        tw = tp.w.sum(); ctot = {c: tp.loc[tp["c_" + c], "w"].sum() for c in ctxs}
        players = []
        for pid, pp in tp.groupby("receiver_player_id"):
            info = lat.loc[pid] if pid in lat.index else None
            if info is not None and (info.team != team or info.status in INACTIVE): continue
            share = pp.w.sum() / tw
            if share < 0.015: continue
            ctx = {c: float(min(2.2, max(0.4, ((pp.loc[pp["c_" + c], "w"].sum() + K_CTX * share) / (ctot[c] + K_CTX)) / share))) for c in ctxs}
            pos = info.position if info is not None else "WR"
            cr = (pp.complete_pass.sum() + K_CATCH * pos_cr.get(pos, 0.65)) / (len(pp) + K_CATCH)
            players.append({"id": pid, "name": info.full_name if info is not None else pp.receiver_player_name.iloc[0], "pos": pos,
                            "espn": str(int(info.espn_id)) if info is not None and pd.notna(info.espn_id) else None,
                            "share": float(share), "ctx": ctx, "catch": float(cr), "targets": int(len(pp)), "rec": int(pp.complete_pass.sum()),
                            "games_since": int(min(gidx[(team, g)] for g in pp.game_id.unique()))})
        players.sort(key=lambda p: -p["share"]); players = players[:MAX_PLAYERS]
        tot = sum(p["share"] for p in players) or 1; n = len(players) or 1
        for p in players: p["share"] = ((p["share"] / tot) * tw + A_SHARE / n) / (tw + A_SHARE)
        teams[team] = {"mult": float((ttw.get(team, 0) + 20 * league_tpd) / (tdw.get(team, 0) + 20) / league_tpd),
                       "pr": float(pr.get(team, 0.58)), "ppd": float(ppd.get(team, 5.5)), "proe": round(float(xp.get(team, 0.0)), 1),
                       "players": players}
    return {"teams": teams, "vol": vol, "league": league, "through_week": int(cur.week.max())}


# ---------------------------------------------------------------- drives to predict
def drives_for_week(w):
    d = p26[(p26.week == w) & p26.play_type.isin(["pass", "run"]) & p26.down.notna()].sort_values(["kick", "game_id", "fixed_drive", "play_id"])
    out = []
    for (gid, dr), x in d.groupby(["game_id", "fixed_drive"], sort=False):
        f = x.iloc[0]
        tg = [(pid, int(c == 1)) for pid, c in zip(x.receiver_player_id, x.complete_pass) if isinstance(pid, str)]
        out.append({"key": f"{gid}#{int(dr)}", "game": gid, "kick": f.kick, "team": f.posteam, "y": float(f.yardline_100),
                    "down": int(f.down), "score": float(f.score_differential), "two": bool(f.half_seconds_remaining <= 120),
                    "targets": tg, "snaps": len(x), "passes": int(x.pass_attempt.sum())})
    return out


WEEKS = list(range(2, MAX_WEEK + 1))
DRIVES = {w: drives_for_week(w) for w in WEEKS}
WEEK_START = {w: p26[p26.week == w].kick.min() for w in WEEKS}
STATE_WEEK = {w: build_state(WEEK_START[w], w - 1) for w in WEEKS}
SLOT_STATE = {}
for w in WEEKS:
    for k in sorted({d["kick"] for d in DRIVES[w]}):
        SLOT_STATE[(w, k)] = build_state(k, w - 1)
log("states built:", len(STATE_WEEK), "weekly,", len(SLOT_STATE), "same-day slots")


# ---------------------------------------------------------------- the predictor (mirrored exactly in the app)
CTX_KEYS = ["rz", "3d", "trail", "2min", "lead"]
def predict(st, team, sit, live, prm, flags):
    t = st["teams"].get(team)
    if not t or not t["players"]: return None
    y, down, score, two = sit
    fb_ = "own0-20" if y > 80 else "own20-40" if y > 60 else "mid" if y > 40 else "opp20-40" if y > 20 else "redzone"
    sb_ = "trail9" if score <= -9 else "trail" if score < 0 else "tied" if score == 0 else "lead" if score <= 8 else "lead9"
    key = f"{fb_}|{'1' if down == 1 else '2' if down == 2 else '3+'}|{sb_}|{'2min' if two else 'normal'}"
    rem = st["vol"].get(key, st["league"]) * t["mult"] * prm["cal"]
    if flags.get("pr") and live["plays"] > 0:
        rem *= ((live["passes"] + prm["k_pr"] * t["pr"]) / (live["plays"] + prm["k_pr"])) / t["pr"]
    if flags.get("pace") and live["drives"] > 0:
        rem *= ((live["snaps"] + prm["k_pace"] * t["ppd"]) / (live["drives"] + prm["k_pace"])) / t["ppd"]
    act = {"rz": y <= 20, "3d": down >= 3, "trail": score <= -9, "2min": two, "lead": score >= 9}
    K, KC, tt, by = prm["k_live"], prm["kc_live"], live["tgt"], live["by"]
    raw = []
    for p in t["players"]:
        s = p["share"]
        if tt > 0: s = (K * s + by.get(p["id"], (0, 0))[0]) / (K + tt)
        for c in CTX_KEYS:
            if act[c]: s *= p["ctx"][c]
        raw.append(s)
    tot = sum(raw) or 1
    rows = []
    for p, s in zip(t["players"], raw):
        lam = s / tot * rem
        ut, ur = by.get(p["id"], (0, 0))
        cr = (KC * p["catch"] + ur) / (KC + ut) if ut else p["catch"]
        pc = 1 - math.exp(-lam * cr)
        feats = [math.log(lam + 1e-4), math.log(min(0.98, max(0.02, cr)) / (1 - min(0.98, max(0.02, cr)))), math.log1p(ut), ut / (tt + 1), math.log(rem + 1e-4),
                 float(act["rz"]), float(act["trail"]), float(act["lead"]), float(two), float(p["pos"] == "TE"), float(p["pos"] == "RB")]
        rows.append((p["id"], pc, 1 - math.exp(-lam), feats))
    return rows

FEATURES = ["log_expected_targets", "logit_catch_rate", "log1p_targets_today", "share_of_team_targets_today", "log_drive_volume",
            "red_zone", "trailing_9plus", "leading_9plus", "two_minute", "is_TE", "is_RB"]


def run_week(w, prm, flags, same_day=False, naive=False):
    """Predict every drive of week w at its first snap. Returns rows (drive_key, player, p_catch, caught, features)."""
    rows, so = [], {}
    for d in DRIVES[w]:
        st = SLOT_STATE[(w, d["kick"])] if same_day else STATE_WEEK[w]
        if st is None: continue
        g = so.setdefault((d["game"], d["team"]), {"tgt": 0, "by": {}, "plays": 0, "passes": 0, "snaps": 0, "drives": 0})
        caught = {pid for pid, c in d["targets"] if c}
        if naive:
            t = st["teams"].get(d["team"])
            pr_ = [(p["id"], 1 - math.exp(-p["share"] * st["league"] * p["catch"]), None, None) for p in t["players"]] if t else None
        else:
            pr_ = predict(st, d["team"], (d["y"], d["down"], d["score"], d["two"]), g, prm, flags)
        if pr_:
            for pid, pc, pt, feats in pr_:
                rows.append((d["key"], pid, pc, int(pid in caught), feats))
        for pid, c in d["targets"]:
            g["tgt"] += 1; a, b = g["by"].get(pid, (0, 0)); g["by"][pid] = (a + 1, b + c)
        g["plays"] += d["snaps"]; g["passes"] += d["passes"]; g["snaps"] += d["snaps"]; g["drives"] += 1
    return rows


def brier(rows): return float(np.mean([(r[2] - r[3]) ** 2 for r in rows])) if rows else float("nan")


# ---------------------------------------------------------------- candidates and their tuning grids
BASE = {"cal": [0.7, 0.8, 0.9], "k_live": [15, 25, 40], "kc_live": [10, 20]}
CANDS = {
    "naive":     {"label": "Season share only", "grid": {}, "flags": {}, "naive": True, "deployable": False},
    "current":   {"label": "Base: situations + live catches", "grid": BASE, "flags": {}, "deployable": True},
    "same_day":  {"label": "Base + same-day games", "grid": BASE, "flags": {"fold": True}, "same_day": True, "deployable": False},
    "pass_rate": {"label": "Base + live pass rate", "grid": {**BASE, "k_pr": [20, 60]}, "flags": {"pr": True}, "deployable": True},
    "pace":      {"label": "Base + live pace", "grid": {**BASE, "k_pace": [2, 6]}, "flags": {"pace": True}, "deployable": True},
    "rate_pace": {"label": "Base + pass rate + pace", "grid": {**BASE, "k_pr": [20, 60], "k_pace": [2, 6]}, "flags": {"pr": True, "pace": True}, "deployable": True},
}
def grid_points(g):
    keys = list(g); return [dict(zip(keys, v)) for v in itertools.product(*[g[k] for k in keys])] or [{}]
DEFAULT = {"cal": 0.8, "k_live": 25, "kc_live": 10, "k_pr": 40, "k_pace": 4}

results = {}   # cand -> {"by_week": {w: rows}, "params": {w: prm}}
grid_scores = {}
for name, c in CANDS.items():
    pts = grid_points(c["grid"])
    per = {}  # (point idx, week) -> rows
    for i, pt in enumerate(pts):
        prm = {**DEFAULT, **pt}
        for w in WEEKS:
            per[(i, w)] = run_week(w, prm, c["flags"], same_day=c.get("same_day", False), naive=c.get("naive", False))
    chosen, by_week = {}, {}
    for w in WEEKS:
        tune = [x for x in WEEKS if x < w]
        if tune:
            sq = [sum((r[2] - r[3]) ** 2 for x in tune for r in per[(i, x)]) / max(1, sum(len(per[(i, x)]) for x in tune)) for i in range(len(pts))]
            best = int(np.argmin(sq))
        else:
            best = next((i for i, pt in enumerate(pts) if all(DEFAULT.get(k) == v for k, v in pt.items())), 0)
        chosen[w] = {**DEFAULT, **pts[best]}; by_week[w] = per[(best, w)]
    results[name] = {"by_week": by_week, "params": chosen}
    # the setting to ship = tuned on every week so far
    sq_all = [sum((r[2] - r[3]) ** 2 for x in WEEKS for r in per[(i, x)]) / max(1, sum(len(per[(i, x)]) for x in WEEKS)) for i in range(len(pts))]
    results[name]["final_params"] = {**DEFAULT, **pts[int(np.argmin(sq_all))]}
    log(f"{name:<10} {len(pts):>3} settings tested")

# ---------------------------------------------------------------- machine-learning challengers (trained walk-forward)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAVE_SK = True
except Exception:
    HAVE_SK = False

def ml_fit(kind, rows):
    X = np.array([r[4] for r in rows]); y = np.array([r[3] for r in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    if kind == "logistic":
        m = LogisticRegression(C=1.0, max_iter=2000).fit((X - mu) / sd, y)
    else:
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, min_samples_leaf=80).fit((X - mu) / sd, y)
    return m, mu, sd

if HAVE_SK:
    feat_src = results["current"]["by_week"]
    for kind, name, label, dep in [("logistic", "ml_logistic", "ML: logistic regression", True),
                                   ("boosted", "ml_boosted", "ML: boosted trees", False)]:
        by_week = {}
        for w in WEEKS:
            train = [r for x in WEEKS if x < w for r in feat_src[x]]
            if len(train) < 1500: continue
            m, mu, sd = ml_fit(kind, train)
            te = feat_src[w]
            p = m.predict_proba((np.array([r[4] for r in te]) - mu) / sd)[:, 1]
            by_week[w] = [(r[0], r[1], float(pi), r[3], r[4]) for r, pi in zip(te, p)]
        results[name] = {"by_week": by_week, "params": {}, "final_params": dict(results["current"]["final_params"])}
        CANDS[name] = {"label": label, "deployable": dep, "ml": kind}
        log(f"{name:<10} trained walk-forward on {len(feat_src)} weeks")

# ---------------------------------------------------------------- leaderboard (out-of-sample weeks only)
TEST_WEEKS = [w for w in WEEKS if w >= 4 and all(w in r["by_week"] for r in results.values())]
def test_rows(name): return [r for w in TEST_WEEKS for r in results[name]["by_week"][w]]

def drive_sums(rows):
    s = {}
    for r in rows:
        a = s.setdefault(r[0], [0.0, 0]); a[0] += (r[2] - r[3]) ** 2; a[1] += 1
    return s

def prob_better(a, b, n=2000, seed=7):
    """Share of bootstrap resamples (by drive) in which model a has a lower Brier score than model b."""
    sa, sb = drive_sums(test_rows(a)), drive_sums(test_rows(b))
    keys = [k for k in sa if k in sb]
    A = np.array([sa[k] for k in keys]); B = np.array([sb[k] for k in keys])
    rng = np.random.default_rng(seed); wins = 0
    for _ in range(n):
        i = rng.integers(0, len(keys), len(keys))
        wins += (A[i, 0].sum() / A[i, 1].sum()) < (B[i, 0].sum() / B[i, 1].sum())
    return wins / n

def top_hit(rows):
    by = {}
    for r in rows:
        if r[0] not in by or r[2] > by[r[0]][2]: by[r[0]] = r
    return sum(r[3] for r in by.values()) / max(1, len(by)), len(by)

naive_b = brier(test_rows("naive"))
board = []
for name, c in CANDS.items():
    rows = test_rows(name)
    hit, nd = top_hit(rows)
    board.append({"name": name, "label": c["label"], "deployable": c["deployable"], "brier": round(brier(rows), 5),
                  "vs_naive_pct": round(100 * (1 - brier(rows) / naive_b), 2), "top_pick_caught": round(hit, 3), "drives": nd,
                  "by_week": {str(w): round(brier(results[name]["by_week"][w]), 5) for w in TEST_WEEKS}})
board.sort(key=lambda r: r["brier"])

sel_path = f"{OUT}/selection.json"
incumbent = json.load(open(sel_path))["model"] if os.path.exists(sel_path) else "current"
inc_b = next(r["brier"] for r in board if r["name"] == incumbent)
switched, conf, winner = False, None, incumbent
for r in board:   # sorted best first
    if not r["deployable"] or r["name"] == incumbent: continue
    gain = 100 * (1 - r["brier"] / inc_b)
    p = prob_better(r["name"], incumbent)
    r["vs_incumbent_gain_pct"], r["p_beats_incumbent"] = round(gain, 3), round(p, 3)
    if gain >= MIN_GAIN_PCT and p >= SWITCH_CONFIDENCE:
        winner, switched, conf = r["name"], True, p
        break
best_dep = next(r for r in board if r["deployable"])
for r in board:
    if r["name"] not in ("naive", winner):
        r["p_beats_winner"] = round(prob_better(r["name"], winner), 3)
log("leaderboard (test weeks", TEST_WEEKS, ")")
for r in board:
    print(f"   {r['name']:<12} brier {r['brier']:.5f}  vs naive {r['vs_naive_pct']:+.2f}%  top pick {r['top_pick_caught']:.1%}  "
          f"{'' if r['deployable'] else '(not on site)'}  {('P(beats winner)=' + str(r.get('p_beats_winner'))) if 'p_beats_winner' in r else ''}")
print(f"   incumbent={incumbent} best_deployable={best_dep['name']} confidence={conf} -> winner={winner}")

# ---------------------------------------------------------------- ship the winner
now = p26.kick.max() + pd.Timedelta(days=1)
prod = build_state(now, MAX_WEEK)
wc = CANDS[winner]
base_name = "current" if wc.get("ml") else winner
prm = results[base_name]["final_params"]
spec = {"name": winner, "label": wc["label"], "flags": {k: True for k in ("pr", "pace") if CANDS[base_name]["flags"].get(k)},
        "chosen": str(date.today()), "switched": switched}
if wc.get("ml") == "logistic":
    allrows = [r for w in WEEKS for r in results["current"]["by_week"][w]]
    m, mu, sd = ml_fit("logistic", allrows)
    spec["logit"] = {"features": FEATURES, "mean": mu.round(6).tolist(), "scale": sd.round(6).tolist(),
                     "coef": m.coef_[0].round(6).tolist(), "intercept": round(float(m.intercept_[0]), 6)}
model = {"built": str(date.today()), "season": SEASON, "through_week": prod["through_week"], "league_rem_mean": round(prod["league"], 3),
         "decay": DECAY, "cal": prm["cal"], "k_live": prm["k_live"], "kc_live": prm["kc_live"], "k_pr": prm["k_pr"], "k_pace": prm["k_pace"],
         "model": spec, "volume": {k: round(v, 3) for k, v in prod["vol"].items()},
         "teams": {t: {"mult": round(v["mult"], 3), "proe": v["proe"], "pr": round(v["pr"], 4), "ppd": round(v["ppd"], 3),
                       "players": [{**p, "share": round(p["share"], 4), "catch": round(p["catch"], 3),
                                    "ctx": {k: round(x, 3) for k, x in p["ctx"].items()}} for p in v["players"]]}
                   for t, v in prod["teams"].items()}}
json.dump(model, open(f"{OUT}/model.json", "w"), separators=(",", ":"))

wr = test_rows(winner)
cal = pd.DataFrame([(r[2], r[3]) for r in wr], columns=["p", "y"])
cal["bin"] = pd.cut(cal.p, [0, .1, .2, .3, .4, 1])
ct = cal.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
hit, nd = top_hit(wr)
summary = {"model": winner, "label": wc["label"], "test_weeks": TEST_WEEKS, "drives": nd, "player_drives": len(wr),
           "brier": round(brier(wr), 5), "brier_naive": round(naive_b, 5), "improvement_pct": round(100 * (1 - brier(wr) / naive_b), 2),
           "top_pick_caught": round(hit, 3),
           "calibration": [{"bin": str(i), "pred": round(float(r.pred), 3), "actual": round(float(r.actual), 3), "n": int(r.n)} for i, r in ct.iterrows()]}
json.dump(summary, open(f"{OUT}/backtest_summary.json", "w"), indent=1)
json.dump({"built": str(date.today()), "through_week": prod["through_week"], "test_weeks": TEST_WEEKS, "incumbent": incumbent,
           "winner": winner, "switched": switched, "switch_confidence": conf, "rule": f"challenger must win >= {SWITCH_CONFIDENCE:.0%} of bootstrap resamples and cut Brier by >= {MIN_GAIN_PCT}%",
           "board": board}, open(f"{OUT}/leaderboard.json", "w"), indent=1)
json.dump({"model": winner, "since": str(date.today())}, open(sel_path, "w"))
log("shipped", winner, "| players", sum(len(t["players"]) for t in model["teams"].values()))
