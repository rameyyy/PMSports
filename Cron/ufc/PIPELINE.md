# UFC Prediction Pipeline

Weekly scrape + daily odds refresh. Scrapes fight data, builds ML features, generates predictions, pushes odds lines — all stored to DB for display on site.

---

## Cron Schedule (`run_ufc.sh`)

| Script | Schedule | What it does |
|---|---|---|
| `oddsapi.py` | Daily | Fetch moneyline + O/U rounds from The Odds API → push to DB |
| `scrape.py` | Sundays only | Full Tapology/UFCStats scrape + odds refresh |

---

## Script Overview

### `oddsapi.py` — Daily
- Calls `bookmakers/sportsbook_api.py` → `GET /v4/sports/mma_mixed_martial_arts/odds?markets=h2h,totals`
- Costs **8 API requests** per run (~240/month of 500 budget)
- Pushes to `bookmaker_moneyline` and `bookmaker_rounds` tables

### `scrape.py` — Sundays
- Runs the full Tapology/UFCStats scrape pipeline (Steps 1–2 below)
- Calls `get_new_upcoming_events()`, `update_scrapes_for_upcoming_events()`, `update_last2_events_outcomes()`
- Then calls `update_bookmakers()` for a fresh odds pull

---

## Step 1: Scrape

**Files:** `scrapes/events.py`, `scrapes/fighter.py`, `update_ufc_db.py`

**What it does:**
- Scrapes upcoming UFC events from Tapology (`get_all_events`, `get_event_data`)
- For each fight on the card, scrapes both fighters from UFCStats (`get_fighter_data`)
- Re-scrapes the last 2 past events to capture final results + methods
- Refreshes data for any upcoming events already in DB

**Two-pass architecture per event:**
1. Fetch all fighter data for the event upfront, collect every UFCStats name
2. One Claude `claude -p` batch call to resolve all UFCStats ↔ Tapology name mismatches
3. Push loop uses cached matches — zero additional LLM calls per event

**HTTP:** 30s timeout on all cloudscraper requests, exponential backoff on 429/connection errors

---

## Step 2: Push to DB

**Files:** `scrapes/sqlpush.py`, `scrapes/namematch.py`

**Name matching:** `EventNameIndex` built from Tapology event card. `preload()` sends all UFCStats names to Claude in one batch call. `find()` checks exact match → cache only, no per-name LLM calls.

**DB Tables Written:**
| Table | Key Columns |
|---|---|
| `events` | event_id, title, date, location |
| `fighters` | fighter_id, name, nickname, img_link, height_in, reach_in, dob, slpm, str_acc, td_avg, win, loss |
| `fights` | fight_id, event_id, fighter1_id, fighter2_id, winner_id, method, weight_class |
| `fight_totals` | fight_id, fighter_id, kd, sig_str, td, ctrl_time_s, head/body/leg/distance/clinch/ground |
| `fight_rounds` | fight_id, round_number, fighter_id, same 24 stat columns |
| `bookmaker_moneyline` | fight_id, bookmaker, fighter1/2_id, fighter1/2_odds, fighter1/2_implied, last_updated |
| `bookmaker_rounds` | fight_id, bookmaker, line, over/under_price, over/under_implied, last_updated |

**NULL-safe upserts:** all ON DUPLICATE KEY UPDATE uses `IF(VALUES(col) IS NULL, col, VALUES(col))` — re-runs never overwrite good data with NULLs.

---

## Step 3: Build Raw DataFrame

**Files:** `models/build_raw_df_all_fights_with_rounds.py`

- Queries all fights + fighter history, builds pre-fight snapshots
- Filters fights where either fighter has < 1 prior fight
- **Output:** `fight_snapshots_all_with_rounds.parquet`

---

## Step 4: Engineer Features

**Files:** `models/fight_features.py`

- Recency-weighted win rate (exp decay, half-life 365d)
- Recent form (last 3/5/7 fights), finish quality score, momentum trend
- Strike + TD differentials, physical differentials (height, reach, age, stance)
- ~90-110 differential features per fight

---

## Step 5: Load Model & Predict

**Files:** `models/load_models_and_predict.py`, `models/build_df_create_predictions.py`

- Loads `scaler.pkl`, `logistic_model.pkl`, `xgboost_model.pkl`, `gradient_boost_model.pkl`
- Generates predictions + win probabilities, ensemble averages
- AlgoPicks generated separately via `simple_predictions.py`

---

## Step 6: Save Predictions to DB

**DB Tables Written:**
| Table | Key Columns |
|---|---|
| `predictions` | fight_id, logistic/xgboost/gradient_pred + probs, ensemble_*, correct |
| `prediction_simplified` | fight_id, algopick_prediction, algopick_probability, correct |
| `model_accuracies` | model_name, accuracy, avg_confidence |

---

## Status (2026-04-21)

| Component | Status |
|---|---|
| Tapology/UFCStats scrape (`scrape.py`) | Working — 12 events, Claude batch matching confirmed |
| Claude name matching (`claude_match.py`) | Working — Song Yadong, Darya Zheleznyakova matched correctly |
| Odds API daily pull (`oddsapi.py`) | Working — 27/48 matched, 21 skipped, 0 errors |
| `bookmaker_moneyline` + `bookmaker_rounds` | Created and populated |
| `run_ufc.sh` daily/Sunday split | In place |
| ML pipeline (Steps 3-6) | Not started |

**Name matching decision:** Claude Code CLI via subprocess (`claude -p --no-session-persistence`, prompt via stdin). Gemini dropped — both models hit daily quota repeatedly during testing.

---

## Known Gaps / Next Steps

| Area | Issue | Priority |
|---|---|---|
| ML section | `make_predictions()` and downstream steps still commented out in `main.py` — not yet wired into new pipeline | High |
| Claude name matching | Not yet tested end-to-end — needs a run with Gemini quota reset or fresh key | High |
| Feature set | RnD used 301 outcome-conditional features; production uses ~90 differentials | Medium |
| Model retraining | Manual only — no automated trigger | Medium |
| Debut fighters | `min_prior_fights ≥ 1` excludes debuts, no fallback | Low |
| Model versioning | `.pkl` files overwritten on retrain | Low |
