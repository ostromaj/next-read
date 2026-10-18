# Next Read

Live NFL drive predictions: for the team with the ball, each receiver's chance of at least one target and one catch before the drive ends.

## What's in here

| Folder | What it does |
| --- | --- |
| `model/build_model.py` | Builds the model from free nflverse play-by-play and runs the backtest |
| `out/model.json` | The built model (2026 season through Week 6) |
| `out/backtest_summary.json` | Weeks 3–5 backtest results |
| `app/template.html` | The app (one page, works on phones and desktops) |
| `app/build_page.py` | Puts the model into the page |
| `server/` | Small Node server: hosts the page and adds live game situations at `/api/live` |

## Weekly model tryouts (runs every Tuesday)

`model/lab.py` tests every candidate model the same way: each week is predicted at the first snap of every drive,
using only games played before it, with settings tuned only on earlier weeks. Candidates:

| Name | What it adds |
| --- | --- |
| `current` | Situation-aware target shares + live in-game catches/targets (the base) |
| `pass_rate` | + how pass-heavy the team is in this game vs. its season rate |
| `pace` | + plays per drive in this game |
| `rate_pace` | + both |
| `same_day` | + games that finished earlier the same day (tested only) |
| `ml_logistic` | Machine learning: logistic regression on every past player-drive |
| `ml_boosted` | Machine learning: gradient-boosted trees (tested only) |

A challenger replaces the model on the site only if it wins at least 90% of bootstrap resamples of the test drives
**and** cuts the Brier score by at least 0.1%. `out/selection.json` remembers which model is on the site.

```bash
mkdir -p data
for y in 2024 2025 2026; do
  curl -L -o data/pbp_$y.csv.gz https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_$y.csv.gz
done
curl -L -o data/rosters_2026.csv https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_2026.csv
pip install pandas numpy scikit-learn
python3 -I model/lab.py data out      # writes out/model.json, leaderboard.json, backtest_summary.json, selection.json
python3 -I app/build_page.py          # puts the winner into the site
```

`model/build_model.py` is the original single-model script, kept for reference.

## Run it with live data on your computer

Needs Node 18 or newer.

```bash
cd server
node server.js
```

Open http://localhost:3000. During games, a "Live games" list appears. Tap a game and the board follows it every 15 seconds.

## Put it online for free (website + phone app)

1. Make a free GitHub account and upload this folder as a new repository.
2. On render.com, create a **Web Service** from that repository.
   - Root directory: `server`
   - Build command: leave blank
   - Start command: `node server.js`
   - Instance type: Free
3. Open the URL Render gives you.
4. On a phone, open that URL and choose **Add to Home Screen**. It then opens like an app.

The free Render tier sleeps after about 15 minutes with no visitors, so the first load can take up to a minute.

## Before charging anyone

- `server.js` reads ESPN's public scoreboard. That feed is unofficial and meant for personal use.
- Point `liveFeed()` in `server.js` (and its `simplify()` parser) at a licensed feed, such as API-Sports or BallDontLie, before charging subscribers.
- Add Stripe Payment Links for the $9.99/month plan.
- Keep the disclaimer and the 1-800-GAMBLER line in the footer.
