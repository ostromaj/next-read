// Next Read live server: serves the app plus two live endpoints built from ESPN's public NFL feeds.
//   GET /api/games        every game this week: score, clock, possession, down, distance, ball spot
//   GET /api/game/:id     players ruled out + each receiver's catches/targets so far in that game
// No dependencies. Needs Node 18+ (built-in fetch). Run: node server.js   (PORT env var optional)
// NOTE: ESPN's feeds are unofficial and for personal, noncommercial use. Swap the two fetch URLs
// for a licensed feed (API-Sports, BallDontLie, Sportradar) before charging subscribers.
const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = process.env.PORT || 3000;
const BASE = process.env.ESPN_BASE || "https://site.api.espn.com/apis/site/v2/sports/football/nfl";
const PAGE = fs.readFileSync(path.join(__dirname, "public", "index.html"));
const ESPN_TO_NFLVERSE = { WSH: "WAS", LAR: "LA" };
const abbr = a => ESPN_TO_NFLVERSE[a] || a;
const OUT_WORDS = /^(out|doubtful|injured reserve|ir|suspended|inactive)/i;

function clockSeconds(text) {
  const m = /^(\d+):(\d{2})/.exec(text || "");
  return m ? +m[1] * 60 + +m[2] : null;
}

// "KC 25" with KC on offense -> 75 yards to the end zone; "BUF 25" -> 25; "50" -> 50
function yardline100(possessionText, posteam) {
  if (!possessionText) return null;
  const m = /([A-Z]{2,3})?\s*(\d{1,2})$/.exec(possessionText.trim());
  if (!m) return null;
  const n = +m[2];
  if (!m[1] || n === 50) return 50;
  return abbr(m[1]) === posteam ? 100 - n : n;
}

function simplifyScoreboard(sb) {
  const games = [];
  for (const ev of sb.events || []) {
    const comp = (ev.competitions || [])[0];
    if (!comp) continue;
    const home = comp.competitors.find(c => c.homeAway === "home");
    const away = comp.competitors.find(c => c.homeAway === "away");
    if (!home || !away) continue;
    const state = comp.status?.type?.state || ev.status?.type?.state || "pre";
    const sit = comp.situation || {};
    const offense = [home, away].find(c => c.team && sit.possession && c.team.id === sit.possession);
    const posteam = state === "in" && offense ? abbr(offense.team.abbreviation) : null;
    const period = comp.status?.period || 0;
    const clk = clockSeconds(comp.status?.displayClock);
    const hs = +home.score || 0, as = +away.score || 0;
    const H = abbr(home.team.abbreviation), A = abbr(away.team.abbreviation);
    games.push({
      id: String(ev.id), state, date: ev.date,
      home: H, away: A, homeScore: hs, awayScore: as,
      period, clock: comp.status?.displayClock || "",
      posteam,
      down: posteam && sit.down > 0 ? sit.down : null,
      distance: posteam && sit.distance > 0 ? sit.distance : null,
      yardline_100: posteam ? yardline100(sit.possessionText, posteam) : null,
      redZone: !!sit.isRedZone,
      scoreDiff: posteam ? (posteam === H ? hs - as : as - hs) : 0,
      halfSeconds: clk == null ? null : (period === 2 || period === 4 ? clk : clk + 900),
      text: sit.downDistanceText || comp.status?.type?.shortDetail || "",
      lastPlay: sit.lastPlay?.text || "",
    });
  }
  const order = { in: 0, pre: 1, post: 2 };
  games.sort((a, b) => (order[a.state] - order[b.state]) || String(a.date).localeCompare(String(b.date)));
  return games;
}

// "P.Mahomes pass short right to T.Kelce to BUF 14 for 6 yards" -> { name: "T.Kelce", caught: true }
function parseTarget(text) {
  if (!text || !/\bpass\b/i.test(text)) return null;
  if (/no play|nullified|two-point|two point|conversion attempt|spiked/i.test(text)) return null;
  const m = /\bpass\b([^]*?)\b(?:to|intended for)\s+([A-Z][^\s.]*\.\s?[^]*)$/.exec(text);
  if (!m) return null;
  const toks = m[2].split(/\s+/), parts = [toks[0].replace(/[),;:]+$/, "")];
  for (const tk of toks.slice(1)) {
    const t = tk.replace(/[),;:]+$/, "");
    if (parts.length < 3 && (/^(Jr\.?|Sr\.?|II|III|IV|V)$/.test(t) || /^[A-Z][a-z'’-]+\.?$/.test(t))) parts.push(t); else break;
  }
  const name = parts.join(" ").replace(/\.$/, "");
  return { name, caught: !/incomplete/i.test(m[1]) && !/intercepted/i.test(text) };
}

// ESPN drives -> each drive's start spot, score at the start, and every target in it (in order)
function simplifyDrives(s) {
  const dd = s.drives || {};
  const list = [...(dd.previous || [])];
  if (dd.current && !list.some(x => x.id === dd.current.id)) list.push(dd.current);
  let h = 0, a = 0;
  return list.map((dr, i) => {
    const team = abbr(dr.team?.abbreviation || "");
    const plays = dr.plays || [];
    const snaps = plays.filter(p => p.start && p.start.down > 0);
    // scrimmage plays the way nflverse counts them: passes (incl. sacks) and runs (incl. scrambles); no punts, kicks, kneels, spikes or wiped-out plays
    const scrim = snaps.filter(p => !/no play|punts|field goal|kneels|spiked|extra point|two-point|timeout|end of|kicks/i.test(p.text || ""));
    const passes = scrim.filter(p => /\bpass\b|sacked/i.test(p.text || "")).length;
    const p0 = snaps[0] || plays[0] || {};
    let y100 = p0.start?.yardsToEndzone;
    if (y100 == null) y100 = yardline100(dr.start?.text || p0.start?.possessionText, team);
    const period = dr.start?.period?.number ?? p0.period?.number ?? null;
    const clockText = dr.start?.clock?.displayValue ?? p0.clock?.displayValue ?? "";
    const clk = clockSeconds(clockText);
    const h0 = h, a0 = a;
    for (const p of plays) { if (p.homeScore != null) h = +p.homeScore; if (p.awayScore != null) a = +p.awayScore; }
    const targets = [];
    for (const p of plays) { const t = parseTarget(p.text); if (t) targets.push(t); }
    const result = dr.displayResult || dr.result || "";
    return {
      id: String(dr.id ?? i), team, period, clock: clockText,
      halfSeconds: clk == null || period == null ? null : (period === 2 || period === 4 ? clk : clk + 900),
      y100: y100 ?? null, startText: dr.start?.text || p0.start?.possessionText || "",
      home0: h0, away0: a0, result, done: !!result || i < list.length - 1,
      snaps: scrim.length, passes, targets,
    };
  });
}

function simplifySummary(s) {
  const out = new Set();
  for (const team of s.injuries || []) {
    for (const inj of team.injuries || []) {
      if (OUT_WORDS.test(inj.status || "") && inj.athlete?.id) out.add(String(inj.athlete.id));
    }
  }
  const receiving = {}, teamTargets = {};
  for (const team of s.boxscore?.players || []) {
    const tAbbr = abbr(team.team?.abbreviation || "");
    for (const cat of team.statistics || []) {
      if (cat.name !== "receiving") continue;
      const keys = cat.keys || [];
      const iRec = keys.indexOf("receptions"), iTgt = keys.indexOf("receivingTargets");
      for (const a of cat.athletes || []) {
        if (!a.athlete?.id || !a.stats) continue;
        const rec = iRec >= 0 ? +a.stats[iRec] || 0 : 0, tgt = iTgt >= 0 ? +a.stats[iTgt] || 0 : null;
        receiving[String(a.athlete.id)] = { rec, tgt, team: tAbbr };
        if (tgt != null) teamTargets[tAbbr] = (teamTargets[tAbbr] || 0) + tgt;
      }
    }
  }
  return { out: [...out], receiving, teamTargets, drives: simplifyDrives(s) };
}

// tiny cache so many viewers share one upstream request
const cache = new Map();
async function cached(key, ttlMs, fn) {
  const hit = cache.get(key);
  if (hit && Date.now() - hit.at < ttlMs) return hit.body;
  if (hit && hit.pending) return hit.pending;
  const pending = fn().then(body => { cache.set(key, { at: Date.now(), body }); return body; })
    .catch(e => { cache.delete(key); throw e; });
  cache.set(key, { at: hit ? hit.at : 0, body: hit && hit.body, pending });
  return pending;
}
async function espn(url) {
  const r = await fetch(url, { headers: { "user-agent": "next-read/0.2" } });
  if (!r.ok) throw new Error("ESPN " + r.status);
  return r.json();
}

function send(res, code, obj) {
  res.writeHead(code, { "content-type": "application/json", "cache-control": "no-store" });
  res.end(typeof obj === "string" ? obj : JSON.stringify(obj));
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, "http://x");
  try {
    if (url.pathname === "/api/games") {
      const body = await cached("games", 3000, async () =>
        JSON.stringify({ updated: new Date().toISOString(), games: simplifyScoreboard(await espn(BASE + "/scoreboard")) }));
      return send(res, 200, body);
    }
    const m = /^\/api\/game\/(\d+)$/.exec(url.pathname);
    if (m) {
      const body = await cached("game:" + m[1], 5000, async () =>
        JSON.stringify(simplifySummary(await espn(BASE + "/summary?event=" + m[1]))));
      return send(res, 200, body);
    }
  } catch (e) {
    return send(res, 502, { error: "Live feed unavailable right now. It will retry automatically.", detail: String(e.message || e) });
  }
  if (url.pathname === "/healthz") return send(res, 200, { ok: true });
  if (url.pathname === "/" || url.pathname === "/index.html") {
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    return res.end(PAGE);
  }
  res.writeHead(404); res.end("Not found");
});

if (require.main === module) server.listen(PORT, () => console.log(`Next Read on http://localhost:${PORT}`));
module.exports = { simplifyScoreboard, simplifySummary, simplifyDrives, parseTarget, yardline100 };
