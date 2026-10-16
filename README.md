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
| `server/` | Small Node server: hosts the page, `/api/games` (live scoreboard) and `/api/game/:id` (players out, catches so far) |

## Update the model each week (Tuesday, after Monday night)

```bash
mkdir -p data
for y in 2024 2025 2026; do
  curl -L -o data/pbp_$y.csv.gz https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_$y.csv.gz
done
curl -L -o data/rosters_2026.csv https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_2026.csv
pip install pandas numpy
DECAY=0.85 K_CTX=40 A_SHARE=15 CAL=0.8 K_LIVE=25 KC_LIVE=10 python3 model/build_model.py data data out
python3 app/build_page.py
```

To re-check accuracy on a finished week (example: Week 6), add `--holdout-week 6`.

## Run it with live data on your computer

Needs Node 18 or newer.

```bash
cd server
node server.js
```

Open http://localhost:3000. Every game this week is listed. Tap one and the board follows whoever has the ball, refreshing every 4 seconds.

## Put it online for free (website + phone app)

1. Make a free GitHub account and upload this folder as a new repository.
2. On render.com, create a **Web Service** from that repository.
   - Build command: `echo ok`
   - Start command: `node server/server.js`
   - Instance type: Free
3. Open the URL Render gives you.
4. On a phone, open that URL and choose **Add to Home Screen**. It then opens like an app.

The free Render tier sleeps after about 15 minutes with no visitors, so the first load can take up to a minute.

## Before charging anyone

- `server.js` reads ESPN's public scoreboard. That feed is unofficial and meant for personal use.
- Point `liveFeed()` in `server.js` (and its `simplify()` parser) at a licensed feed, such as API-Sports or BallDontLie, before charging subscribers.
- Add Stripe Payment Links for the $9.99/month plan.
- Keep the disclaimer and the 1-800-GAMBLER line in the footer.
