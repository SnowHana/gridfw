# CSSP Portfolio Optimizer — Project Brief

## What we're building

A web app that takes a stock universe as input, runs a CSSP-based sparse portfolio optimization algorithm, and outputs an optimized portfolio with visual breakdown. Target users are retail investors interested in quant-driven portfolio construction.

---

## Core Features (MVP)

### 1. Stock Universe Input

- User inputs a list of stock tickers (e.g. AAPL, MSFT, TSLA)
- OR selects from a preset universe (e.g. S&P 500, ASX 200)
- Set number of stocks to select (cardinality constraint k)

### 2. CSSP Optimization Engine (Python backend)

- Core algorithm already exists — integrate from /algorithm/cssp.py
- Input: returns matrix from historical price data
- Output: selected subset of k stocks + optimal weights
- Use yfinance to fetch historical price data (default: 2 years daily)

### 3. Results Dashboard

- Selected portfolio: ticker, weight %, sector
- Key metrics: Expected return, Volatility, Sharpe Ratio
- Comparison vs equal-weight benchmark and S&P 500
- Efficient frontier chart (portfolio vs random portfolios)
- Correlation heatmap of selected stocks

### 4. Export

- Download results as CSV
- Shareable link to results (stretch goal)

---

## Tech Stack

### Frontend

- React + Tailwind CSS
- Recharts for charts
- Clean, minimal UI — dark mode preferred

### Backend

- FastAPI (Python)
- yfinance for market data
- NumPy, SciPy for optimization
- pandas for data handling

### Deployment (MVP)

- Frontend: Vercel
- Backend: Railway or Render (free tier)

---

## File Structure

```
cssp-portfolio/
├── CLAUDE.md
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── TickerInput.jsx       # stock universe input
│   │   │   ├── ResultsTable.jsx      # portfolio weights table
│   │   │   ├── MetricsCard.jsx       # sharpe, return, vol
│   │   │   ├── EfficientFrontier.jsx # scatter chart
│   │   │   └── HeatMap.jsx           # correlation heatmap
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── index.html
│   └── package.json
├── backend/
│   ├── main.py                       # FastAPI entry point
│   ├── algorithm/
│   │   └── cssp.py                   # PASTE YOUR CSSP CODE HERE
│   ├── data/
│   │   └── fetch.py                  # yfinance data fetching
│   ├── optimizer/
│   │   └── portfolio.py              # weights, metrics calculation
│   └── requirements.txt
└── README.md
```

---

## API Endpoints

### POST /optimize

Request:

```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
  "k": 3,
  "lookback_years": 2
}
```

Response:

```json
{
  "selected": ["AAPL", "MSFT", "GOOGL"],
  "weights": [0.45, 0.35, 0.2],
  "metrics": {
    "expected_return": 0.142,
    "volatility": 0.187,
    "sharpe_ratio": 1.23
  },
  "benchmark_sharpe": 0.89
}
```

### POST /universe

- Returns price data + correlation matrix for a given ticker list

---

## Algorithm Notes (for Claude)

- The CSSP algorithm selects k columns (stocks) from a returns matrix to best approximate the full covariance structure
- This is mathematically equivalent to sparse minimum-variance portfolio selection
- The existing Python code uses continuous relaxation of the NP-hard cardinality constraint
- Key parameters: k (cardinality), lookback period, regularization lambda
- DO NOT rewrite the core algorithm — only wrap it with data I/O and API layer

---

## Design Guidelines

- Dark mode, minimal, data-forward
- Primary color: #6366f1 (indigo)
- Font: Inter
- Mobile responsive but desktop-first
- No unnecessary animations — keep it fast and clean

---

## MVP Success Criteria

- User can input tickers → get optimized portfolio in < 10 seconds
- Charts render correctly
- Deployed and publicly accessible via URL
- Works on mobile

---

## Out of Scope (for now)

- User accounts / login
- Saving portfolios
- Real-time prices (daily close is fine)
- Paid tier / paywall
- Backtesting engine

---

## First Task for Claude Code

1. Scaffold the full file structure above
2. Set up FastAPI backend with a /health endpoint
3. Set up React + Tailwind frontend with a basic ticker input form
4. Connect frontend to backend with a test API call
5. I will then paste in my CSSP algorithm code into /backend/algorithm/cssp.py
