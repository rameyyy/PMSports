# UFC Scraper v2 — Status & Changelog

## What Changed (v2)

### Bug Fixes
- **`push_fights` NULL overwrite** — ON DUPLICATE KEY UPDATE uses `IF(VALUES(col) IS NULL, col, VALUES(col))` for all columns; re-runs never wipe existing data
- **`push_totals` / `push_rounds` name matching** — replaced hard 0.88 threshold with argmax (always picks closer of two fighters)
- **Duplicate `_normalize_date`** — removed duplicate definition in `fighter.py`
- **HTTP timeout** — `cloudscraper.get()` in `events.py` and `fighter.py` had no timeout, causing indefinite hangs. Fixed to 30s with exponential backoff retries

### New Cron Structure
- `main.py` retired as entrypoint
- `scrape.py` — Tapology/UFCStats scrape + odds, runs Sundays
- `oddsapi.py` — Odds API only, runs daily (8 requests/run, ~240/month)
- `run_ufc.sh` — cron entrypoint, mirrors `run_ncaamb.sh` pattern

### Odds API (new)
- Pulls `h2h` + `totals` (O/U rounds) markets in one call
- **`bookmaker_moneyline`** table — fight_id, bookmaker, fighter1/2_id, odds, implied %, last_updated
- **`bookmaker_rounds`** table — fight_id, bookmaker, line, over/under price + implied %, last_updated
- `bookmaker_odds` is legacy and unused going forward

### Claude Name Matching (replaces Gemini)
- **Decision: Claude Code CLI via subprocess — Gemini dropped entirely.** Both Gemini models (Flash 24/20 RPD, Flash Lite ~500 RPD) hit daily quota from test runs and are not reliable enough at free tier.
- `scrapes/claude_match.py` — calls `claude -p --no-session-persistence` via subprocess with prompt passed via stdin (not shell arg). Runs from `~` to avoid project context.
- `EventNameIndex.preload()` — collects ALL UFCStats names for an event upfront, sends ONE `claude -p` batch call, caches results
- `find()` — exact match -> cache only, no per-name LLM calls ever
- `update_ufc_db.py` restructured to two-pass: fetch all fighter data -> preload -> push loop
- Claude CLI path: `C:\Users\crame\AppData\Roaming\npm\claude.cmd`

---

## Where We Left Off (2026-04-21)

### Confirmed Working
- **`scrape.py` full run** — 12 events processed, one Claude batch call per event, correct matches returned
  - Song Yadong -> Yadong Song (name-order swap)
  - Daria Zhelezniakova -> Darya Zheleznyakova (transliteration)
- **`oddsapi.py`** — 27/48 fights matched, 21 skipped (non-UFC / not yet in DB), 0 errors
- **`bookmaker_moneyline` + `bookmaker_rounds`** — created and populated with live data, timestamps correct (UTC)
- **`run_ufc.sh`** — daily/Sunday split in place

---

## Still Needs Testing

### Odds Pipeline
- [ ] Run `oddsapi.py` again after a fight card update to confirm upsert (ON DUPLICATE KEY UPDATE) works correctly
- [ ] Confirm `bookmaker_rounds` line value (2.5) stored as decimal correctly across fights

### Odds Pipeline
- [ ] Run `oddsapi.py` again after a fight card update to confirm upsert (ON DUPLICATE KEY UPDATE) works correctly
- [ ] Confirm `bookmaker_rounds` line value (2.5) stored as decimal correctly across fights

### ML Section (still commented out)
- [ ] Uncomment and test `make_predictions()` + downstream in `scrape.py`
- [ ] Verify `prediction_simplified` table populates for upcoming fights
- [ ] Test `bookmaker_odds` EV calculation once predictions are live (or retire it entirely)

---

## Next Up — ML Pipeline (not started)

This is the full remaining implementation roadmap before the cron is complete. Everything below is v2 work that hasn't been touched yet.

### Step A: Build Raw DF
- Need to wire `build_raw_df_all_fights_with_rounds.py` into `scrape.py` so it runs after the DB push
- Should query directly from DB (not from old parquet snapshots) and produce a fresh parquet each Sunday
- Key question: does the existing script handle the current DB schema cleanly, or does it need updates from the backfill/snapshot work done in RnD?
- Output: `fight_snapshots_all_with_rounds.parquet`

### Step B: Generate Features
- `models/fight_features.py` takes the raw parquet → outputs flat feature df
- Need to verify the feature set is compatible with the current trained model (same column names, same scaling)
- If the model was trained on a different feature set from the RnD branch, will need to reconcile or retrain
- Output: flat df ready for model input

### Step C: Load Model + Generate Predictions
- Load `scaler.pkl` + model `.pkl` files, scale features, run predictions for upcoming fights only
- Need to decide: keep existing 3-model ensemble or simplify to single best model (LR based on RnD results ~0.63 ROC-AUC)
- Output: per-fight win probabilities for each upcoming fight

### Step D: New Prediction Tables
- `prediction_simplified` exists but was populated manually from RnD — need to verify schema still matches
- Decide whether to keep `predictions` (full multi-model table) or just use `prediction_simplified` going forward
- Either way, upsert logic needs to be null-safe same as the scrape tables
- New consideration: store implied odds from `bookmaker_moneyline` alongside model prob so EV can be computed at query time on the frontend instead of in the cron

### Step E: Update Old Predictions (final cron step)
- After each fight card completes, `update_predictions_winners()` needs to run to mark `correct = 1/0` based on actual winner from `fights` table
- This is the last step of the weekly cron — runs after `update_last2_events_outcomes()` has pushed final results
- Also triggers `update_accuracies()` to recalculate per-model accuracy stats in `model_accuracies`
- Needs to be added to `scrape.py` after the scrape phase

---

## After Cron — UI + Backend (out of scope for now)

Once the cron is solid end-to-end:
- Backend API endpoints to serve predictions, odds, EV to the frontend
- UI pages for upcoming fight card with AlgoPicks + moneyline + O/U data
- Historical performance / model accuracy display
- Bet tracking UI (bets table + bet_analytics already in DB, just needs frontend)

---

## API Budget Notes
- **Odds API:** 8 requests/daily run × 30 = ~240/month of 500 free budget
- **Claude Code:** subprocess `claude -p`, one call per event per Sunday scrape — no separate API key needed
- **Gemini:** both models at daily quota limit as of 2026-04-21, not used going forward
