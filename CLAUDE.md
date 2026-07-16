# PMSports

Sports prediction site. Two independent pipelines (UFC, NCAAMB) train ML models and write predictions to MySQL; a Flask API serves them; a React frontend displays them under the "AlgoPicks" brand.

## Workspaces

- **`backend/`** — Flask API (`flask==3.0.0`). Serves UFC + NCAAMB routes from MySQL via `backend/routes/`. Entry point `backend/app.py`. Dockerized. Depends on `backend/requirements.txt`.
- **`react-site/`** — Vite + React 19 + TypeScript + TailwindCSS frontend. Pages under `src/pages/{about,ncaamb,ufc}`. Dockerized. Uses `rolldown-vite`, `react-router-dom` v7, `recharts`.
- **`Cron/`** — Python data pipelines. Runs on a VM at `/home/caramey/prod/` via system cron. Two independent sports:
  - `Cron/ufc/` — UFC MMA. Entry `run_ufc.sh`. See `Cron/ufc/PIPELINE.md` and `Cron/ufc/UFC_V2_STATUS.md`.
  - `Cron/ncaamb/` — NCAA men's basketball. Entry `run_ncaamb.sh`. Season-gated Nov 1 – Apr 15.
- **`.github/workflows/`** — ops-only workflows today (`deploy_prod`, `rebuild`, `restart`, `diagnose`, `restart_cloudflared`). No test/lint CI yet — see roadmap below.
- **`compose.yml` + `nginx.conf`** — full stack orchestration (api + ui + flaresolverr + cloudflared + nginx).

## What is AlgoPicks

The site's user-facing prediction brand. Two independent implementations, one per sport:

- **UFC:** `Cron/ufc/models/simple_predictions.py` picks the single best-accuracy model per event (dynamic per-event choice from logistic / xgboost / gradient / homemade / three ensemble variants), then **calibrates** the raw probability against empirical accuracy in a local ±0.0065 probability window. Stored in `ufc.prediction_simplified` (`algopick_model`, `algopick_prediction`, `algopick_probability`).
- **NCAAMB:** `Cron/ncaamb/ui_jobs/update_ui_games.py` is hard-wired to the LightGBM model's `gbm_prob_*` / `lgb_pred` output — no per-game model selection, no calibration. Surfaced as `team_1_prob_algopicks` / `team_2_prob_algopicks` in the UI-games table.

Distinct from the raw ensemble outputs (`ufc.predictions`, NCAAMB per-model probability columns), which store every model's prediction. AlgoPicks = the single curated recommendation. On the frontend, NCAAMB even treats "AlgoPicks" as a pseudo-sportsbook in the ModelPerformance panel for accuracy comparison.

## Cron pipelines

- **Entry points:** `Cron/ufc/run_ufc.sh`, `Cron/ncaamb/run_ncaamb.sh` (both invoked by system cron on the VM).
- **Failure model:** non-`set -e`, per-stage isolation via a `FAILED=()` array and a `run_stage()` helper. One flaky stage doesn't skip the rest of the day. See either `.sh` for the pattern.
- **Season gate:** `run_ncaamb.sh` exits early May–Oct and after Apr 15. `run_ufc.sh` runs year-round with Sunday-only stages for scrape/predict/homepage.
- **Directory conventions:**
  - Files at pipeline root = actually invoked by cron (or transitively imported by something that is).
  - `<pipeline>/tools/` = one-off analyses, backfills, exports, smoke tests. Not called by cron.
  - `<pipeline>/training/` = retrain-only scripts that produce `.pkl` model artifacts. Not called by cron.
- **Reference docs:** `Cron/ufc/PIPELINE.md`, `Cron/ufc/UFC_V2_STATUS.md`.

## Environment

- Each pipeline has its own gitignored `.env`. Templates: `Cron/.env.example` (UFC), `Cron/ncaamb/.env.example` (NCAAMB).
- MySQL host is shared; per-sport DB names differ (`UFC_DB=ufc`, `NCAAMB_DB=ncaamb`).
- Python: plain pip + per-workspace `requirements.txt`. `backend/` and `Cron/ncaamb/` have manifests; **`Cron/ufc/` currently does not** (roadmap item).

---

## Roadmap (not iter-1 — placeholders to flesh out over time)

### uv migration
Greenfield migration from pip-in-venv to `uv`. No `pyproject.toml`, `uv.lock`, or `.python-version` exists anywhere in the repo today. Target: one `pyproject.toml` per app + a shared workspace root, `uv.lock` committed, `requirements.txt` retired.

### App / sub-app / lib split
Long-term structure `Cron/` should evolve toward:
- `apps/ufc/` — subapps: `kalshi`, `bookies`, `events`, `fighter-data`
- `apps/ncaamb/` — equivalent subapp breakdown
- `libs/db/` — shared MySQL connection handling (currently duplicated across `Cron/ufc/{bets,scrapes,models,bookmakers}/utils.py` and multiple NCAAMB modules)
- `libs/utils/` — shared helpers (dates, name matching, env loading)

Consolidation of the ~half-dozen near-identical `get_db_connection()` implementations is the concrete first step.

### CI + tests
Only ops workflows exist today (deploy / restart / rebuild / diagnose). Need:
- Unit tests for pure functions (feature engineering, odds conversion, name matching)
- Regression tests for the pipeline stages (fixture-driven, real DB — no mocks; production migration failures are the reason)
- Test doubles for external HTTP (Tapology, UFCStats, The Odds API)
- Run on PR

### Continued cleanup
- **Iter-2 target:** model / parquet / CSV artifact strategy. Options: Cloudflare R2 free tier, git-lfs, HuggingFace hub. Currently ~470M of binaries tracked in git as the de facto cross-device sync mechanism.
- Add `Cron/ufc/requirements.txt`.
- Decompose `Cron/ncaamb/main.py` (currently 2,525 lines).
- Wire the UFC v2 ML pipeline (steps 3–6 in `Cron/ufc/PIPELINE.md` — `make_predictions()` and downstream still commented out in `Cron/ufc/scrape.py`).

## Conventions

_User will extend over time._
