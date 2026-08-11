# Project Spec: Deploy gridfw as a Sparse Portfolio Replication Tool

## What we're actually building

A publicly accessible web app where a user picks a target portfolio universe (e.g. S&P 500)
and a cardinality k, and the app returns the k stocks that best replicate the full
index's covariance structure — using CSSP via FW-Homotopy.

This is sparse ETF replication: find the minimum number of stocks needed to track
an index efficiently.

**The user flow:**

1. Select a universe (S&P 500 to start; extensible later)
2. Set k — number of stocks to select (slider, e.g. 10–100)
3. Click "Find portfolio"
4. See: which k stocks were selected, their sectors, how well they cover the index

**Not:** the core `/replicate` tool is not a benchmark comparison tool, not FW vs
Greedy, not academic metrics. A separate, clearly-labeled `/performance` page
(see Phase 2.5) links out to the existing walk-forward backtest research
(`examples/market/backtest.py`) for anyone curious how CSSP-selected portfolios
performed historically — but it's static, secondary, and never part of the
k-selection flow above.

---

## Why this is the right product

CSSP solves the sparse index replication problem exactly:

```text
max Tr(X^T P_S X)  s.t. |S| = k
```

This selects the k columns (stocks) whose covariance structure best spans the full
index. That is the principled solution to "which 20 stocks do I need to replicate
the S&P 500?" — directly relevant to ETF construction, proxy hedging, and
cost-efficient passive investing.

---

## Tech Stack (fixed)

| Layer      | Choice                  | Reason                               |
| ---------- | ----------------------- | ------------------------------------ |
| API        | FastAPI                 | Python, matches existing codebase    |
| Frontend   | Plain HTML + vanilla JS | No build step                        |
| Charts     | Chart.js via CDN        | Single script tag                    |
| Container  | Docker                  | Required keyword for recruiters      |
| Deployment | Railway or Render       | Free tier, simplest path to live URL |

No React, Vue, TypeScript, database, Redis, or Celery.

---

## Critical constraint: execution time

Running FW-Homotopy on S&P 500 (p ≈ 472) with production params takes ~2–3 min.
That is too slow for a synchronous HTTP request.

**Strategy: precompute a grid of results.**

For each supported universe × k combination, precompute and store the result as JSON.
Serve instantly on request. The user chooses from a discrete slider (k = 10, 20, 30,
40, 50) rather than arbitrary k. Results are labelled "precomputed" in the UI.

This keeps the architecture simple (no async jobs, no queues) while making the demo
feel instant.

Precomputed grid for launch:

- Universe: S&P 500
- k values: 10, 20, 30, 40, 50
- Total: 5 JSON files, generated once and committed to the repo

**Status: superseded.** k is now a free-form value (bounded `k_min`-`n_stocks`
per universe) instead of a fixed discrete set, since the frontend now lets the
user pick any k directly. This is fine while the solver runs at lightweight
dev settings (`n_steps=800` fixed, not scaled with k), but it reopens the
timing question for deploy: if solver settings are made stricter later,
either reintroduce a discrete/capped k range, or make caching key off
`(universe, k)` pairs generated on demand (still needs a warm-cache pass for
common values before going live, per the note below).

**Status: deferred.** During initial dev, `universe_registry.get_replication()`
computes live on every call (no on-disk cache) to keep iteration simple while
building out the API. Before Phase 4 (deploy), add cache-or-compute logic:
check `app/precomputed/{universe}_k{k}.json` first, compute + write only on a
miss, and warm the cache for the full grid above before going live. Without
this, a live request at production solver settings (~2-3 min) will time out
against Railway/Render's HTTP proxy.

---

## API Endpoints

### GET /health

```json
{"status": "ok"}
```

### GET /universes

Lists available universes and their metadata.

```json
[
  {
    "id": "sp500",
    "label": "S&P 500",
    "n_stocks": 472,
    "k_options": [10, 20, 30, 40, 50],
    "description": "S&P 500 constituents, 2018–2026 daily returns"
  }
]
```

### POST /replicate

Request:

```json
{ "universe": "sp500", "k": 20 }
```

Response:

```json
{
  "universe": "sp500",
  "k": 20,
  "selected": [
    { "ticker": "AAPL", "sector": "Technology", "weight": 0.05 },
    { "ticker": "XOM",  "sector": "Energy",     "weight": 0.05 }
  ],
  "cssp_objective": 0.234,
  "coverage_pct": 84.2,
  "precomputed": true
}
```

`coverage_pct` is what the user sees: "these 20 stocks cover 84% of the index's
variance structure". Computed as `Tr(A_SS^-1 A^2_SS) / Tr(A^-1 A^2) * 100`.

---

## Frontend

Single `index.html` served by FastAPI. No build step.

Layout (top to bottom):

1. Header: "Sparse Index Replication" — one sentence explaining what it does
2. Universe selector (dropdown, populated from `GET /universes`)
3. k slider — discrete steps from the universe's `k_options`
4. "Find portfolio" button
5. Results section (hidden until run):
   - Headline: "20 stocks covering 84% of S&P 500 variance"
   - Bar chart: sector breakdown of selected stocks (Chart.js)
   - Table: ticker, sector, weight for each selected stock
   - Small note: "Results precomputed. Algorithm: FW-Homotopy (CSSP)"

---

## File structure

```text
app/
├── main.py                  # FastAPI app
├── schemas.py               # Pydantic models
├── universe_registry.py     # universe metadata + precomputed result loader
├── precomputed/
│   ├── sp500_k10.json
│   ├── sp500_k20.json
│   ├── sp500_k30.json
│   ├── sp500_k40.json
│   └── sp500_k50.json
└── static/
    └── index.html

scripts/
└── precompute.py            # run once to generate the JSON files
```

---

## Phases

### Phase 0: Precompute results (do first)

Run FW-Homotopy for each (sp500, k) combination. Store JSON. Commit.
This is the data pipeline that everything else serves.

Script: `scripts/precompute.py`

### Phase 1: FastAPI backend

- `/health`, `/universes`, `/replicate` endpoints
- Load precomputed JSON, validate k, return structured response
- Pydantic models, clean error messages, presentable `/docs`

Acceptance criteria:

- [ ] `uvicorn app.main:app` starts without errors
- [ ] All endpoints return correct responses
- [ ] Invalid k returns 422 with a readable message
- [ ] `/docs` is clean

### Phase 2: Frontend

- `index.html` with dropdown, slider, button, results table, sector chart
- Chart.js from CDN
- Loading and error states

Acceptance criteria:

- [ ] Dropdown populates from API
- [ ] Slider updates to universe's k_options
- [ ] Run returns table of selected stocks and sector chart
- [ ] Coverage % is prominent
- [ ] Works on mobile

### Phase 2.5: Performance/research page (secondary, non-core)

A `/performance` page, linked from the main tool but not part of its flow,
surfacing `examples/market/backtest.py`'s walk-forward comparison (CSSP vs
market-cap/random/full-universe baselines, momentum and MVO variants) as
static content:

- Regenerate reports by rerunning `backtest.py` (not live per-request - this
  is a genuinely heavy walk-forward computation, ~75 rolling windows)
- Serve the resulting PNGs + quantstats HTML tearsheets from `app/reports/`
- `app/static/performance.html`: cumulative-return and rolling-Sharpe charts
  embedded inline, links to open each quantstats tearsheet
- Clearly labeled as historical research, not a live feature of the k-selector

Acceptance criteria:

- [ ] `/performance` loads and links/embeds the regenerated reports
- [ ] Main page (`index.html`) links to it, but `/replicate` is unaffected

### Phase 3: Docker

- `Dockerfile` (python:3.11-slim)
- Precomputed JSON files bundled in image
- `docker-compose.yml` for local dev
- `.dockerignore`

Acceptance criteria:

- [ ] `docker build` succeeds
- [ ] `docker run` serves working app on localhost

### Phase 4: Deploy

- Railway or Render free tier
- Public URL
- README updated with live link at top

---

## Non-goals

- Custom ticker input (scope creep — preloaded universes only)
- Real-time prices (precomputed is fine)
- Backtesting or performance charts **in the core `/replicate` tool** — a
  static, secondary `/performance` page is in scope (Phase 2.5), the
  interactive k-selector is not
- User accounts
- Showing FW vs Greedy comparison **in the core tool** (the `/performance`
  page's existing research does compare strategies, that's fine there)
- Rewriting the algorithm

---

## Definition of done

1. Public URL loads a working page
2. User selects S&P 500 + k → sees k stocks with sectors in under 2 seconds
3. README has live link and plain-language explanation
4. `docker build && docker run` works for anyone who clones it
