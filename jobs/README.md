# jobs/

Greenfield home for PMSports data pipelines and shared libraries, managed as a `uv` workspace.

Will eventually replace `Cron/`. Nothing in `jobs/` is wired into cron or the VM yet — this is scaffolding only. `Cron/` remains the source of truth for production pipelines until each app is migrated over and cut across.

## Layout

- `apps/` — data pipelines. One directory per pipeline (future: `ufc/`, `ncaamb/`, ...).
- `libs/` — shared packages consumed by apps.
  - `libs/db-io/` — MySQL connection handling.
  - `libs/utils/` — small shared helpers (dates, env, name matching).

Each app and lib is its own uv workspace member with its own `pyproject.toml`. The root `jobs/pyproject.toml` is a virtual workspace — it declares members and dev tooling but ships no code itself.

## Working in the workspace

```bash
cd jobs
uv sync                    # install all workspace members + dev deps into .venv
uv run pytest              # run tests
uv run ruff check          # lint
```

Adding a new lib or app: create the directory, add its `pyproject.toml`, and it's picked up automatically by the `libs/*` / `apps/*` glob in the workspace root.

## Migration plan

See root `CLAUDE.md` roadmap. Order will roughly be: consolidate `get_db_connection()` into `libs/db-io`, then migrate one pipeline at a time (likely UFC first, since it has fewer files).
