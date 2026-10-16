"""Build the Drive Target Predictor model from free nflverse data.

Inputs (CSV, from nflverse-data GitHub releases):
  play_by_play_2024/2025/2026.csv.gz, roster_weekly_2026.csv, injuries_2026.csv
Outputs:
  model.json   - everything the app needs (team/player shares, drive volume table)
  backtest.json - how well the model did on the most recent week it did not see

Usage: python3 -I build_model.py <pbp_dir> <extra_dir> <out_dir> [--holdout-week N]
"""
import json, math, os, sys
from datetime import date
import numpy as np
import pandas as pd

PBP_DIR, EXTRA_DIR, OUT_DIR = sys.argv[1], sys.argv[2], sys.argv[3]
HOLDOUT = None
if "--holdout-week" in sys.argv:
    HOLDOUT = int(sys.argv[sys.argv.index("--holdout-week") + 1])

SEASON = 2026
DECAY = float(os.environ.get("DECAY", 0.85))  # weight per game back in time (most recent game = 1.0)
K_CTX = float(os.environ.get("K_CTX", 8.0))  # shrinkage: weighted targets needed before a situational split is trusted
K_CATCH = 15.0      # shrinkage for catch rate toward position average
A_SHARE = float(os.environ.get("A_SHARE", 0))  # weighted targets of pull toward an even split
CAL = float(os.environ.get("CAL", 1.0))  # overall scale on expected targets (calibration)
K_LIVE = float(os.environ.get("K_LIVE", 1e9))  # live: how many in-game targets the pregame share is worth
KC_LIVE = float(os.environ.get("KC_LIVE", 1e9))  # live: how many in-game targets the catch rate is worth
COLS = ["season", "season_type", "week", "game_id", "game_date", "posteam", "defteam", "fixed_drive",
        "play_id", "down", "ydstogo", "yardline_100", "score_differential", "qtr",
        "half_seconds_remaining", "pass_attempt", "sack", "qb_spike", "receiver_player_id",
        "receiver_player_name", "complete_pass", "xpass", "play_type"]


def load(year):
    d = pd.read_csv(f"{PBP_DIR}/pbp_{year}.csv.gz", usecols=COLS, low_memory=False)
    return d[d.season_type == "REG"]


def score_bucket(s):
    return np.select([s <= -9, s < 0, s == 0, s <= 8], ["trail9", "trail", "tied", "lead"], "lead9")


def field_bucket(y):
    return np.select([y > 80, y > 60, y > 40, y > 20], ["own0-20", "own20-40", "mid", "opp20-40"], "redzone")


def down_bucket(d):
    return np.select([d == 1, d == 2], ["1", "2"], "3+")


def time_bucket(t):
    return np.where(t <= 120, "2min", "normal")


# ---------- 1. Drive volume table: expected remaining targets from any spot in a drive ----------
hist = pd.concat([load(y) for y in (2024, 2025, SEASON)])
if HOLDOUT:
    hist = hist[~((hist.season == SEASON) & (hist.week >= HOLDOUT))]
scrim = hist[hist.play_type.isin(["pass", "run"]) & hist.down.notna()].copy()
scrim["tgt"] = scrim.receiver_player_id.notna().astype(int)
scrim = scrim.sort_values(["game_id", "fixed_drive", "play_id"])
# targets from this play to the end of the drive (inclusive)
scrim["rem_tgt"] = scrim.iloc[::-1].groupby(["game_id", "fixed_drive"]).tgt.cumsum().iloc[::-1]
scrim["fb"] = field_bucket(scrim.yardline_100)
scrim["sb"] = score_bucket(scrim.score_differential)
scrim["db"] = down_bucket(scrim.down)
scrim["tb"] = time_bucket(scrim.half_seconds_remaining)
vol = (scrim.groupby(["fb", "db", "sb", "tb"]).rem_tgt.agg(["mean", "count"]).reset_index())
league_rem_mean = scrim.rem_tgt.mean()
# shrink thin cells toward the field+down average
fd = scrim.groupby(["fb", "db"]).rem_tgt.mean().rename("fd_mean").reset_index()
vol = vol.merge(fd, on=["fb", "db"])
vol["value"] = (vol["mean"] * vol["count"] + vol["fd_mean"] * 40) / (vol["count"] + 40)
volume_table = {f"{r.fb}|{r.db}|{r.sb}|{r.tb}": round(float(r.value), 3) for r in vol.itertuples()}

# ---------- 2. This season's usage, recency weighted ----------
cur = load(SEASON)
if HOLDOUT:
    cur = cur[cur.week < HOLDOUT]
roster = pd.read_csv(f"{EXTRA_DIR}/rosters_2026.csv", low_memory=False)
roster = roster[roster.week <= (HOLDOUT - 1 if HOLDOUT else roster.week.max())]
latest = roster.sort_values("week").groupby("gsis_id").tail(1).set_index("gsis_id")

passes = cur[cur.receiver_player_id.notna()].copy()
# game index per team, most recent = 0
team_games = cur.groupby("posteam").game_id.unique()
gidx = {}
for team, games in team_games.items():
    order = sorted(games, key=lambda g: cur.loc[cur.game_id == g, "game_date"].iloc[0], reverse=True)
    for i, g in enumerate(order):
        gidx[(team, g)] = i
passes["w"] = [DECAY ** gidx[(t, g)] for t, g in zip(passes.posteam, passes.game_id)]
passes["ctx_rz"] = passes.yardline_100 <= 20
passes["ctx_3d"] = passes.down >= 3
passes["ctx_trail"] = passes.score_differential <= -9
passes["ctx_2min"] = passes.half_seconds_remaining <= 120
passes["ctx_lead"] = passes.score_differential >= 9

# league catch rate by position
passes["pos"] = passes.receiver_player_id.map(latest.position).fillna("WR")
pos_cr = passes.groupby("pos").complete_pass.mean().to_dict()

# team pace: targets per drive vs league (this season, weighted, shrunk)
drives = cur[cur.play_type.isin(["pass", "run"])].groupby(["posteam", "game_id", "fixed_drive"]).size().reset_index()
drives["w"] = [DECAY ** gidx[(t, g)] for t, g in zip(drives.posteam, drives.game_id)]
team_drive_w = drives.groupby("posteam").w.sum()
team_tgt_w = passes.groupby("posteam").w.sum()
league_tpd = team_tgt_w.sum() / team_drive_w.sum()
team_mult = ((team_tgt_w + 20 * league_tpd) / (team_drive_w + 20) / league_tpd).to_dict()

# pass rate over expected
dropbacks = cur[cur.xpass.notna() & cur.play_type.isin(["pass", "run"])]
proe = (dropbacks.groupby("posteam").apply(lambda d: (d.pass_attempt.mean() - d.xpass.mean()) * 100,
                                            include_groups=False)).round(1).to_dict()

INACTIVE = {"RES", "CUT", "RET", "DEV", "EXE"}
teams_out = {}
for team, tp in passes.groupby("posteam"):
    tw = tp.w.sum()
    ctx_tot = {c: tp.loc[tp[c], "w"].sum() for c in ["ctx_rz", "ctx_3d", "ctx_trail", "ctx_2min", "ctx_lead"]}
    players = []
    for pid, pp in tp.groupby("receiver_player_id"):
        info = latest.loc[pid] if pid in latest.index else None
        if info is not None and info.team != team:
            continue  # traded or released: his old targets don't belong to this team anymore
        if info is not None and info.status in INACTIVE:
            continue
        share = pp.w.sum() / tw
        if share < 0.015:
            continue
        ratios = {}
        for c, tot in ctx_tot.items():
            ctx_share = (pp.loc[pp[c], "w"].sum() + K_CTX * share) / (tot + K_CTX)
            ratios[c[4:]] = round(float(min(2.2, max(0.4, ctx_share / share))), 3)
        pos = info.position if info is not None else "WR"
        cr = (pp.complete_pass.sum() + K_CATCH * pos_cr.get(pos, 0.65)) / (len(pp) + K_CATCH)
        last_game = min(gidx[(team, g)] for g in pp.game_id.unique())
        name = info.full_name if info is not None else pp.receiver_player_name.iloc[0]
        players.append({
            "id": pid, "name": name, "pos": pos,
            "espn": str(int(info.espn_id)) if info is not None and pd.notna(info.espn_id) else None,
            "share": round(float(share), 4), "ctx": ratios, "catch": round(float(cr), 3),
            "targets": int(len(pp)), "rec": int(pp.complete_pass.sum()),
            "games_since": int(last_game),
        })
    players.sort(key=lambda p: -p["share"])
    total = sum(p["share"] for p in players)
    n = len(players) or 1
    for p in players:
        p["share"] = round(((p["share"] / total) * tw + A_SHARE / n) / (tw + A_SHARE), 4)
    teams_out[team] = {"mult": round(float(team_mult.get(team, 1.0)), 3),
                       "proe": proe.get(team, 0.0), "players": players[:12]}


def predict(team, yardline_100, down, score_diff, half_secs, out_ids=(), today=None, team_tgt_today=0):
    """Expected targets and P(at least one catch) on the rest of this drive, per player."""
    key = f"{field_bucket(np.array([yardline_100]))[0]}|{down_bucket(np.array([down]))[0]}|" \
          f"{score_bucket(np.array([score_diff]))[0]}|{time_bucket(np.array([half_secs]))[0]}"
    rem = volume_table.get(key, league_rem_mean) * teams_out[team]["mult"]
    active = {"rz": yardline_100 <= 20, "3d": down >= 3, "trail": score_diff <= -9,
              "2min": half_secs <= 120, "lead": score_diff >= 9}
    ps = [p for p in teams_out[team]["players"] if p["id"] not in out_ids]
    raw = []
    today = today or {}
    for p in ps:
        s = p["share"]
        if team_tgt_today > 0:  # live update: blend pregame share with this game's target share
            s = (K_LIVE * s + today.get(p["id"], (0, 0))[0]) / (K_LIVE + team_tgt_today)
        for c, on in active.items():
            if on:
                s *= p["ctx"][c]
        raw.append(s)
    tot = sum(raw) or 1
    out = []
    for p, s in zip(ps, raw):
        lam = s / tot * rem * CAL
        t_tgt, t_rec = today.get(p["id"], (0, 0))
        cr = (KC_LIVE * p["catch"] + t_rec) / (KC_LIVE + t_tgt)
        out.append({"id": p["id"], "name": p["name"], "xt": lam, "p_target": 1 - math.exp(-lam),
                    "p_catch": 1 - math.exp(-lam * cr)})
    return sorted(out, key=lambda r: -r["p_catch"]), rem


# ---------- 3. Backtest on a week the model never saw ----------
if HOLDOUT:
    test = load(SEASON)
    test = test[(test.week == HOLDOUT) & test.play_type.isin(["pass", "run"]) & test.down.notna()]
    test = test.sort_values(["game_id", "fixed_drive", "play_id"])
    league_rate = 1 - math.exp(-league_rem_mean)
    rows, base_rows, top_hits, n_drives = [], [], 0, 0
    calib = []
    so_far = {}  # (game, team) -> {player: [targets, receptions]} before the current drive
    static_rows = []
    for (gid, dr), d in test.groupby(["game_id", "fixed_drive"]):
        team = d.posteam.iloc[0]
        if team not in teams_out:
            continue
        first = d.iloc[0]
        g = so_far.setdefault((gid, team), {})
        today = {k: tuple(v) for k, v in g.items()}
        preds, rem = predict(team, first.yardline_100, first.down, first.score_differential,
                             first.half_seconds_remaining, today=today, team_tgt_today=sum(v[0] for v in g.values()))
        static, _ = predict(team, first.yardline_100, first.down, first.score_differential,
                            first.half_seconds_remaining)
        caught = set(d.loc[d.complete_pass == 1, "receiver_player_id"])
        for r in static:
            static_rows.append((r["p_catch"] - (1 if r["id"] in caught else 0)) ** 2)
        for _, pl in d[d.receiver_player_id.notna()].iterrows():
            v = g.setdefault(pl.receiver_player_id, [0, 0]); v[0] += 1; v[1] += int(pl.complete_pass == 1)
        n_drives += 1
        # naive baseline: season share only, league-average drive volume, no situation
        for p in teams_out[team]["players"]:
            lam = p["share"] * league_rem_mean
            y = 1 if p["id"] in caught else 0
            base_rows.append((1 - math.exp(-lam * 0.65) - y) ** 2)
        for r in preds:
            y = 1 if r["id"] in caught else 0
            rows.append((r["p_catch"] - y) ** 2)
            calib.append((r["p_catch"], y))
        if preds and preds[0]["id"] in caught:
            top_hits += 1
    cal = pd.DataFrame(calib, columns=["p", "y"])
    cal["bin"] = pd.cut(cal.p, [0, .1, .2, .3, .4, .6, 1.0])
    cal_tbl = cal.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size"))
    has_catch = (test.groupby(["game_id", "fixed_drive"]).complete_pass.sum() > 0).mean()
    bt = {
        "holdout_week": HOLDOUT, "drives": n_drives,
        "brier_model": round(float(np.mean(rows)), 4), "brier_no_live": round(float(np.mean(static_rows)), 5), "brier_baseline": round(float(np.mean(base_rows)), 4),
        "top_pick_caught_rate": round(top_hits / n_drives, 3),
        "drives_with_any_catch": round(float(has_catch), 3),
        "calibration": [{"bin": str(i), "pred": round(float(r.pred), 3), "actual": round(float(r.actual), 3),
                         "n": int(r.n)} for i, r in cal_tbl.iterrows()],
    }
    bt["raw"] = {"static_sq": static_rows, "model_sq": rows, "base_sq": base_rows, "calib": calib, "top_hits": top_hits}
    json.dump(bt, open(f"{OUT_DIR}/backtest_w{HOLDOUT}.json", "w"))
    print({k: v for k, v in bt.items() if k not in ("raw", "calibration")})
else:
    model = {
        "built": str(date.today()), "season": SEASON, "through_week": int(cur.week.max()),
        "league_rem_mean": round(float(league_rem_mean), 3), "decay": DECAY, "cal": CAL, "k_live": K_LIVE if K_LIVE < 1e8 else None, "kc_live": KC_LIVE if KC_LIVE < 1e8 else None,
        "volume": volume_table, "teams": teams_out,
    }
    json.dump(model, open(f"{OUT_DIR}/model.json", "w"), separators=(",", ":"))
    print("teams", len(teams_out), "through week", model["through_week"],
          "players", sum(len(t["players"]) for t in teams_out.values()))
    for team in ["KC", "DAL"]:
        if team in teams_out:
            preds, rem = predict(team, 75, 1, 0, 900)
            print(team, f"exp targets rest of drive {rem:.2f}")
            for r in preds[:5]:
                print(f"  {r['name']:<22} xT {r['xt']:.2f}  P(target) {r['p_target']:.0%}  P(catch) {r['p_catch']:.0%}")
