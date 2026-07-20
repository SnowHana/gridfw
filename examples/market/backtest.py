"""Walk-forward backtest: CSSP-reduced momentum vs all baselines, plus MVO experiment.

Momentum strategies:
  1. cssp_momentum       — CSSP selects k stocks, momentum picks top_n within them
  2. full_momentum       — momentum on all available stocks, picks top_n
  3. marketcap_momentum  — top-k by market cap, momentum picks top_n
  4. random_momentum     — random k stocks, momentum picks top_n (avg over seeds)
  5. equal_weight        — passive benchmark: equal weight all stocks

MVO strategies (Path B — where covariance conditioning actually matters):
  6. cssp_mvo            — CSSP selects k stocks, MVO finds min-variance weights on those k
  7. marketcap_mvo       — top-k by market cap, MVO weights
  8. full_mvo            — MVO on all p stocks (demonstrates instability from ill-conditioning)

Why MVO amplifies conditioning differences:
  MVO inverts the covariance matrix Σ directly. When cond(Σ) ~ 2M (full universe),
  the inversion amplifies noise → weights concentrate on 1-2 spurious stocks.
  When cond(Σ) ~ 592 (CSSP), the inversion is stable → weights are diversified and
  out-of-sample variance matches in-sample. Momentum only needs signal rankings so
  conditioning barely matters there. MVO depends on it entirely.

Diagnostics:
  1. Statistical significance  — monthly t-test, CSSP-MVO vs each baseline
  2. Subsample Sharpe          — pre/post 2022 drawdown
  3. Condition number          — is the CSSP submatrix better conditioned?
  4. Selection stability       — Jaccard similarity across consecutive windows
  5. Sector attribution        — which sectors does CSSP consistently pick?
  6. Quantstats tearsheets     — full HTML report for CSSP-MVO and market-cap-MVO
  7. Weight concentration      — how many stocks get non-trivial weight in each MVO solution

Run:
    cd examples/market && python backtest.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from grad_fw import FWHomotopySolver
from sp500_load_data import load_sp500_prices_df, load_sp500_companies_index_df


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_company_maps():
    """Return (marketcap_map, sector_map) dicts keyed by ticker.

    Market cap is static (current), not historical — noted as a limitation.
    """
    dfs = load_sp500_companies_index_df()
    companies = dfs["companies"].copy()
    companies["Symbol"] = companies["Symbol"].str.replace(".", "-", regex=False)
    marketcap_map = dict(zip(companies["Symbol"], companies["Marketcap"]))
    sector_map = dict(zip(companies["Symbol"], companies["Sector"]))
    return marketcap_map, sector_map


# ---------------------------------------------------------------------------
# Walk-forward windows
# ---------------------------------------------------------------------------

def generate_windows(prices_df, train_years=2, test_months=1):
    """Yield (train_df, test_df) pairs rolling forward by test_months.

    Each window includes only stocks with complete data in both periods.
    """
    prices_df = prices_df.sort_index()
    n = len(prices_df)
    train_days = int(train_years * 252)
    test_days = int(test_months * 21)

    i = train_days
    while i + test_days <= n:
        train_df = prices_df.iloc[i - train_days : i]
        test_df = prices_df.iloc[i : i + test_days]
        valid = train_df.columns[train_df.notna().all() & test_df.notna().all()]
        if len(valid) >= 10:
            yield train_df[valid], test_df[valid]
        i += test_days


# ---------------------------------------------------------------------------
# Signal and portfolio helpers
# ---------------------------------------------------------------------------

def momentum_signal(prices_df, lookback_days=252, skip_days=21):
    """12-1 momentum: return from lookback_days ago to skip_days ago."""
    lb = min(lookback_days, len(prices_df) - skip_days - 1)
    if lb <= 0:
        return pd.Series(0.0, index=prices_df.columns)
    return prices_df.iloc[-skip_days] / prices_df.iloc[-lb] - 1


def _ew_returns(tickers, test_log_returns):
    valid = [t for t in tickers if t in test_log_returns.columns]
    if not valid:
        return pd.Series(0.0, index=test_log_returns.index)
    return test_log_returns[valid].mean(axis=1)


def _log_returns(prices_df):
    return np.log(prices_df / prices_df.shift(1)).dropna()


def _weighted_returns(weights_dict, test_log_returns):
    """Portfolio return series from a {ticker: weight} dict."""
    valid = [t for t in weights_dict if t in test_log_returns.columns]
    if not valid:
        return pd.Series(0.0, index=test_log_returns.index)
    w = np.array([weights_dict[t] for t in valid])
    w = np.maximum(w, 0.0)
    if w.sum() == 0:
        return pd.Series(0.0, index=test_log_returns.index)
    w /= w.sum()
    return pd.Series(test_log_returns[valid].values @ w, index=test_log_returns.index)


def mvo_min_variance(cov_matrix, tickers, reg_scale=1e-3):
    """Analytical long-only minimum-variance portfolio.

    Closed-form solution: w* = Σ⁻¹1 / (1ᵀΣ⁻¹1), then clip negatives.
    Uses np.linalg.solve (O(n³) but constant-time, no iterative solver).
    This is ~100x faster than scipy SLSQP and sufficient for our purposes.

    Tikhonov regularisation (reg_scale * mean_eigenvalue * I) prevents
    numerical blow-up on near-singular matrices — same reg applied to all
    three universes so the comparison is fair.
    """
    n = len(tickers)
    reg = reg_scale * np.trace(cov_matrix) / n
    cov_reg = cov_matrix + reg * np.eye(n)

    ones = np.ones(n)
    # Solve cov_reg @ x = ones  (more stable than explicit inv)
    try:
        x = np.linalg.solve(cov_reg, ones)
    except np.linalg.LinAlgError:
        x = ones  # fallback to equal weight

    w = np.maximum(x, 0.0)   # long-only clip
    total = w.sum()
    if total <= 0:
        w = ones / n
    else:
        w /= total
    return dict(zip(tickers, w))


def _effective_n(weights_dict):
    """Effective number of stocks = 1 / sum(w²)  (inverse Herfindahl).

    Equal-weight portfolio of n stocks → effective_n = n.
    Single-stock portfolio → effective_n = 1.
    """
    w = np.array(list(weights_dict.values()))
    w = w[w > 0]
    return float(1.0 / (w ** 2).sum()) if len(w) > 0 else 0.0


def _condition_number(A, idx):
    """Condition number of the submatrix A[idx, idx]."""
    if not idx:
        return np.nan
    sub = A[np.ix_(idx, idx)]
    evals = np.linalg.eigvalsh(sub)
    evals = evals[evals > 1e-10]
    return float(evals[-1] / evals[0]) if len(evals) > 1 else np.nan


# ---------------------------------------------------------------------------
# Single window
# ---------------------------------------------------------------------------

def run_window(train_prices, test_prices, k, top_n, fw_kwargs, marketcap_map):
    """Execute one walk-forward window.

    Returns:
        returns_dict: {strategy: daily log-return Series}
        meta: {cssp_tickers, cap_tickers, condition_*, test_start}
    """
    tickers = list(train_prices.columns)
    p = len(tickers)
    actual_k = min(k, p)

    log_ret_train = _log_returns(train_prices)
    log_ret_test = _log_returns(test_prices)
    A = np.cov(log_ret_train.values.T)

    # 1. CSSP-reduced momentum
    s = FWHomotopySolver(A, actual_k, **fw_kwargs).solve(verbose=False)
    cssp_idx = FWHomotopySolver.selected_indices(s)
    cssp_tickers = [tickers[i] for i in cssp_idx]
    top_cssp = momentum_signal(train_prices[cssp_tickers]).nlargest(min(top_n, len(cssp_tickers))).index.tolist()
    cssp_ret = _ew_returns(top_cssp, log_ret_test)

    # 2. Full-universe momentum
    top_full = momentum_signal(train_prices).nlargest(top_n).index.tolist()
    full_ret = _ew_returns(top_full, log_ret_test)

    # 3. Market-cap-filtered momentum  (key confound test)
    caps = {t: marketcap_map.get(t, 0.0) for t in tickers}
    cap_tickers = sorted(caps, key=caps.get, reverse=True)[:actual_k]
    top_cap = momentum_signal(train_prices[cap_tickers]).nlargest(min(top_n, len(cap_tickers))).index.tolist()
    cap_ret = _ew_returns(top_cap, log_ret_test)

    # 4. Random-k momentum (averaged over 5 seeds for stability)
    rng = np.random.default_rng(0)
    rand_rets = []
    for seed in range(5):
        rand_tickers = list(np.random.default_rng(seed).choice(tickers, size=actual_k, replace=False))
        top_rand = momentum_signal(train_prices[rand_tickers]).nlargest(min(top_n, len(rand_tickers))).index.tolist()
        rand_rets.append(_ew_returns(top_rand, log_ret_test))
    rand_ret = pd.concat(rand_rets, axis=1).mean(axis=1)

    # 5. Equal-weight benchmark
    ew_ret = _ew_returns(tickers, log_ret_test)

    # --- MVO strategies ---
    # Covariance submatrices (same A used for CSSP, so no lookahead)
    def _sub_cov(selected_tickers):
        idx = [tickers.index(t) for t in selected_tickers]
        return A[np.ix_(idx, idx)], selected_tickers

    # 6. CSSP-MVO: min-variance on CSSP-selected submatrix (cond ~ 592)
    cssp_cov, _ = _sub_cov(cssp_tickers)
    cssp_mvo_w = mvo_min_variance(cssp_cov, cssp_tickers)
    cssp_mvo_ret = _weighted_returns(cssp_mvo_w, log_ret_test)

    # 7. Market-cap-MVO: min-variance on top-k-by-cap submatrix (cond ~ 8,587)
    cap_cov, _ = _sub_cov(cap_tickers)
    cap_mvo_w = mvo_min_variance(cap_cov, cap_tickers)
    cap_mvo_ret = _weighted_returns(cap_mvo_w, log_ret_test)

    # 8. Full-MVO: min-variance on all p stocks (cond ~ 2M → near-singular)
    # Same regularisation as the others — exposes how ill-conditioning survives reg.
    full_mvo_w = mvo_min_variance(A, tickers)
    full_mvo_ret = _weighted_returns(full_mvo_w, log_ret_test)

    # Condition numbers + effective-N for weight-concentration diagnostic
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}
    meta = {
        "cssp_tickers":    cssp_tickers,
        "cap_tickers":     cap_tickers,
        "condition_full":  _condition_number(A, list(range(p))),
        "condition_cssp":  _condition_number(A, [ticker_to_idx[t] for t in cssp_tickers]),
        "condition_cap":   _condition_number(A, [ticker_to_idx[t] for t in cap_tickers if t in ticker_to_idx]),
        "effn_cssp_mvo":   _effective_n(cssp_mvo_w),
        "effn_cap_mvo":    _effective_n(cap_mvo_w),
        "effn_full_mvo":   _effective_n(full_mvo_w),
        "test_start":      test_prices.index[0],
    }

    returns = {
        "cssp_momentum":      cssp_ret,
        "full_momentum":      full_ret,
        "marketcap_momentum": cap_ret,
        "random_momentum":    rand_ret,
        "equal_weight":       ew_ret,
        "cssp_mvo":           cssp_mvo_ret,
        "marketcap_mvo":      cap_mvo_ret,
        "full_mvo":           full_mvo_ret,
    }
    return returns, meta


# ---------------------------------------------------------------------------
# Full backtest
# ---------------------------------------------------------------------------

_STRATEGIES = (
    "cssp_momentum", "full_momentum", "marketcap_momentum", "random_momentum", "equal_weight",
    "cssp_mvo", "marketcap_mvo", "full_mvo",
)
_LABELS = {
    "cssp_momentum":      "CSSP-Momentum",
    "full_momentum":      "Full-Universe Momentum",
    "marketcap_momentum": "Market-Cap-Filtered Momentum",
    "random_momentum":    "Random-k Momentum",
    "equal_weight":       "Equal-Weight (benchmark)",
    "cssp_mvo":           "CSSP-MVO (cond≈592)",
    "marketcap_mvo":      "MarketCap-MVO (cond≈8K)",
    "full_mvo":           "Full-MVO (cond≈2M, unstable)",
}


def run_comparison(prices_df, k=50, top_n=20, verbose=True):
    """Walk-forward comparison across all windows.

    Returns:
        results_df: daily log-return DataFrame (one column per strategy)
        summary_df: Sharpe / ann.return / vol / max-drawdown per strategy
        metadata:   list of per-window dicts (selections, condition numbers)
    """
    fw_kwargs = dict(n_steps=50, n_mc_samples=10, alpha=0.1)
    marketcap_map, _ = _load_company_maps()
    windows = list(generate_windows(prices_df, train_years=2, test_months=1))

    if verbose:
        print(f"Walk-forward: {len(windows)} windows | k={k} | top_n={top_n} | FW={fw_kwargs}")

    all_returns = {s: [] for s in _STRATEGIES}
    metadata = []

    for i, (train_prices, test_prices) in enumerate(windows):
        if verbose and (i % 6 == 0 or i == len(windows) - 1):
            print(
                f"  [{i+1:3d}/{len(windows)}] p={len(train_prices.columns)} | "
                f"test: {test_prices.index[0].date()} → {test_prices.index[-1].date()}"
            )
        result, meta = run_window(train_prices, test_prices, k, top_n, fw_kwargs, marketcap_map)
        for strat in _STRATEGIES:
            all_returns[strat].append(result[strat])
        metadata.append(meta)

    results_df = pd.DataFrame(
        {s: pd.concat(series).sort_index() for s, series in all_returns.items()}
    )
    summary_df = _summarise(results_df)
    if verbose:
        print("\n--- Summary ---")
        print(summary_df.to_string())

    return results_df, summary_df, metadata


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _sharpe(s, periods=252):
    return float(s.mean() / s.std() * np.sqrt(periods)) if s.std() > 0 else 0.0


def _max_drawdown(s):
    cum = s.cumsum()
    return float((cum - cum.cummax()).min())


def _to_simple(log_ret):
    return np.expm1(log_ret)


def _summarise(results_df, periods=252):
    rows = []
    for col in results_df.columns:
        s = results_df[col]
        rows.append({
            "Strategy": _LABELS.get(col, col),
            "Ann. Return": f"{s.mean() * periods * 100:.1f}%",
            "Volatility":  f"{s.std() * np.sqrt(periods) * 100:.1f}%",
            "Sharpe":      f"{_sharpe(s):.3f}",
            "Max Drawdown": f"{_max_drawdown(s) * 100:.1f}%",
        })
    return pd.DataFrame(rows).set_index("Strategy")


# ---------------------------------------------------------------------------
# Diagnostic functions
# ---------------------------------------------------------------------------

def _diag_significance(results_df):
    """Monthly t-test: CSSP-MVO vs every other strategy (and CSSP-momentum vs its baselines)."""
    from scipy import stats

    def _monthly(col):
        s = _to_simple(results_df[col])
        return s.resample("ME").apply(lambda x: np.expm1(np.log1p(x).sum()))

    def _row(anchor_col, other_col):
        diff = (_monthly(anchor_col) - _monthly(other_col)).dropna()
        t, p = stats.ttest_1samp(diff, 0)
        delta_ann = diff.mean() * 12 * 100
        sig = "✓ p<0.05" if p < 0.05 else ("~ p<0.10" if p < 0.10 else "✗ not sig")
        anchor_lbl = _LABELS[anchor_col].split("(")[0].strip()
        print(f"  {anchor_lbl} vs {_LABELS[other_col]:<33} {delta_ann:>+9.2f}% {t:>8.2f} {p:>8.4f}  {sig}")

    print(f"\n  {'comparison':55s} {'Δ ann.ret':>10} {'t-stat':>8} {'p-val':>8}  sig?")
    print("  " + "-" * 90)
    print("  -- MVO strategies --")
    for col in ("marketcap_mvo", "full_mvo", "cssp_momentum", "marketcap_momentum", "equal_weight"):
        _row("cssp_mvo", col)
    print("  -- Momentum strategies --")
    for col in ("marketcap_momentum", "full_momentum", "random_momentum", "equal_weight"):
        _row("cssp_momentum", col)


def _diag_subsample(results_df):
    """Sharpe split at Jan 2022 (bull vs bear/recovery)."""
    split = pd.Timestamp("2022-01-01")
    pre = results_df[results_df.index < split]
    post = results_df[results_df.index >= split]

    print(f"\n  {'Strategy':<38} {'Pre-2022':>10} {'Post-2022':>10}")
    print("  " + "-" * 62)
    for col in _STRATEGIES:
        s_pre  = _sharpe(pre[col])  if len(pre)  > 30 else float("nan")
        s_post = _sharpe(post[col]) if len(post) > 30 else float("nan")
        print(f"  {_LABELS[col]:<38} {s_pre:>10.3f} {s_post:>10.3f}")


def _diag_condition_numbers(metadata, output_dir):
    """Plot and summarise covariance condition numbers per window."""
    dates     = [m["test_start"]      for m in metadata]
    cond_full = [m["condition_full"]  for m in metadata]
    cond_cssp = [m["condition_cssp"]  for m in metadata]
    cond_cap  = [m["condition_cap"]   for m in metadata]

    for label, vals in [("Full", cond_full), ("CSSP", cond_cssp), ("MarketCap", cond_cap)]:
        print(f"  {label:<12} condition number — mean: {np.nanmean(vals):>10.1f} | median: {np.nanmedian(vals):>10.1f}")

    _, ax = plt.subplots(figsize=(12, 4))
    ax.semilogy(dates, cond_full, color="gray",       lw=1.2, label=f"Full (mean={np.nanmean(cond_full):.0f})", alpha=0.7)
    ax.semilogy(dates, cond_cssp, color="darkorange", lw=1.8, label=f"CSSP (mean={np.nanmean(cond_cssp):.0f})")
    ax.semilogy(dates, cond_cap,  color="steelblue",  lw=1.4, label=f"Market-cap (mean={np.nanmean(cond_cap):.0f})")
    ax.set_ylabel("Condition number (log scale)")
    ax.set_title("Covariance condition number per window\n(lower = better conditioned)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    path = os.path.join(output_dir, "condition_numbers.png")
    plt.savefig(path, dpi=150)
    print(f"  Saved → {path}")
    plt.show()


def _diag_stability(metadata):
    """Jaccard similarity between consecutive CSSP selections."""
    scores = []
    for prev, curr in zip(metadata[:-1], metadata[1:]):
        a, b = set(prev["cssp_tickers"]), set(curr["cssp_tickers"])
        scores.append(len(a & b) / len(a | b) if (a | b) else 0.0)

    k = len(metadata[0]["cssp_tickers"])
    p = 484  # approximate
    random_j = (k**2 / p) / (2 * k - k**2 / p)

    print(f"  Mean Jaccard: {np.mean(scores):.3f}  (random baseline ≈ {random_j:.3f})")
    print(f"  Min: {min(scores):.3f}  Max: {max(scores):.3f}")
    print(f"  Mean monthly selection turnover: {(1 - np.mean(scores)) * 100:.1f}%")

    if np.mean(scores) > 2 * random_j:
        print("  → Selections are stable — CSSP consistently picks the same representative stocks")
    else:
        print("  → Selections rotate frequently — high implied turnover cost")


def _diag_sectors(metadata):
    """Sector distribution: CSSP vs market-cap filter."""
    _, sector_map = _load_company_maps()

    cssp_counts, cap_counts = {}, {}
    total = 0
    for m in metadata:
        for t in m["cssp_tickers"]:
            sec = sector_map.get(t, "Unknown")
            cssp_counts[sec] = cssp_counts.get(sec, 0) + 1
            total += 1
        for t in m["cap_tickers"]:
            sec = sector_map.get(t, "Unknown")
            cap_counts[sec] = cap_counts.get(sec, 0) + 1

    all_sectors = sorted(set(list(cssp_counts) + list(cap_counts)))
    print(f"\n  {'Sector':<30} {'CSSP %':>8} {'MktCap %':>9}")
    print("  " + "-" * 50)
    for sec in all_sectors:
        c = cssp_counts.get(sec, 0) / total * 100
        m = cap_counts.get(sec, 0) / total * 100
        diff = c - m
        marker = " ← CSSP overweight" if diff > 3 else (" ← CSSP underweight" if diff < -3 else "")
        print(f"  {sec:<30} {c:>7.1f}% {m:>8.1f}%{marker}")


def _diag_weight_concentration(metadata):
    """Effective-N of MVO solutions: how concentrated are the weights?

    Equal-weight k=50 → effective_n = 50.
    Single-stock → effective_n = 1.
    Lower effective_n = more concentrated = less diversified = more unstable.
    """
    effn_cssp = [m["effn_cssp_mvo"] for m in metadata]
    effn_cap  = [m["effn_cap_mvo"]  for m in metadata]
    effn_full = [m["effn_full_mvo"] for m in metadata]

    k = len(metadata[0]["cssp_tickers"])
    print(f"  (Equal-weight baseline: effective_n = k = {k})")
    print(f"\n  {'Strategy':<30} {'Mean eff-N':>12} {'Min eff-N':>10} {'Max eff-N':>10}")
    print("  " + "-" * 65)
    for label, vals in [("CSSP-MVO", effn_cssp), ("MarketCap-MVO", effn_cap), ("Full-MVO", effn_full)]:
        print(f"  {label:<30} {np.nanmean(vals):>12.1f} {np.nanmin(vals):>10.1f} {np.nanmax(vals):>10.1f}")

    print()
    if np.nanmean(effn_cssp) > np.nanmean(effn_cap):
        print("  → CSSP-MVO is MORE diversified than MarketCap-MVO (better conditioning → more stable weights)")
    else:
        print("  → MarketCap-MVO is more diversified — conditioning advantage not reflected in weights")
    if np.nanmean(effn_full) < 5:
        print(f"  → Full-MVO effective_n < 5: near-singular covariance caused severe weight concentration")


def _diag_rolling_sharpe(results_df, output_dir, window=252):
    """Two rolling Sharpe charts: one for momentum strategies, one for MVO strategies."""
    momentum_style = {
        "cssp_momentum":      ("CSSP-Momentum",               "darkorange", 2.2),
        "full_momentum":      ("Full-Universe Momentum",       "steelblue",  1.8),
        "marketcap_momentum": ("Market-Cap-Filtered Momentum", "green",      1.6),
        "random_momentum":    ("Random-k Momentum",            "purple",     1.2),
        "equal_weight":       ("Equal-Weight",                 "gray",       1.0),
    }
    mvo_style = {
        "cssp_mvo":      ("CSSP-MVO",           "darkorange", 2.4),
        "marketcap_mvo": ("MarketCap-MVO",       "steelblue",  1.8),
        "full_mvo":      ("Full-MVO (unstable)", "red",        1.4),
        "equal_weight":  ("Equal-Weight",        "gray",       1.0),
    }

    for title_suffix, style, fname in [
        ("momentum strategies", momentum_style, "rolling_sharpe_momentum.png"),
        ("MVO strategies",      mvo_style,      "rolling_sharpe_mvo.png"),
    ]:
        _, ax = plt.subplots(figsize=(13, 5))
        for col, (label, color, lw) in style.items():
            if col not in results_df.columns:
                continue
            s = results_df[col]
            rs = s.rolling(window).mean() / s.rolling(window).std() * np.sqrt(252)
            ax.plot(rs.index, rs.values, label=label, color=color, linewidth=lw)
        ax.axhline(0, color="black", lw=0.6, linestyle="--", alpha=0.4)
        ax.set_ylabel("Rolling 12-month Sharpe")
        ax.set_title(f"Rolling Sharpe (252-day window) — {title_suffix}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        path = os.path.join(output_dir, fname)
        plt.savefig(path, dpi=150)
        print(f"  Saved → {path}")
        plt.show()


def _diag_quantstats(results_df, output_dir):
    """Full HTML tearsheets via quantstats for key strategies."""
    import quantstats as qs
    ew = _to_simple(results_df["equal_weight"])
    targets = [
        ("cssp_mvo",      "cssp_mvo_tearsheet.html"),
        ("marketcap_mvo", "marketcap_mvo_tearsheet.html"),
        ("cssp_momentum", "cssp_momentum_tearsheet.html"),
    ]
    for col, fname in targets:
        ret = _to_simple(results_df[col])
        path = os.path.join(output_dir, fname)
        qs.reports.html(ret, benchmark=ew, output=path, title=_LABELS[col], download_filename=path)
        print(f"  Tearsheet → {path}")


# ---------------------------------------------------------------------------
# Master analyse function
# ---------------------------------------------------------------------------

def analyse(results_df, metadata, output_dir="examples/market/figures"):
    """Run all diagnostics in sequence."""
    os.makedirs(output_dir, exist_ok=True)
    sep = "=" * 65

    print(f"\n{sep}")
    print("DIAGNOSTIC 1: Statistical significance (monthly t-test)")
    print(sep)
    _diag_significance(results_df)

    print(f"\n{sep}")
    print("DIAGNOSTIC 2: Subsample Sharpe — pre/post 2022 drawdown")
    print(sep)
    _diag_subsample(results_df)

    print(f"\n{sep}")
    print("DIAGNOSTIC 3: Covariance condition number")
    print(sep)
    _diag_condition_numbers(metadata, output_dir)

    print(f"\n{sep}")
    print("DIAGNOSTIC 4: MVO weight concentration (effective-N)")
    print(sep)
    _diag_weight_concentration(metadata)

    print(f"\n{sep}")
    print("DIAGNOSTIC 5: CSSP selection stability (Jaccard)")
    print(sep)
    _diag_stability(metadata)

    print(f"\n{sep}")
    print("DIAGNOSTIC 6: Sector attribution")
    print(sep)
    _diag_sectors(metadata)

    print(f"\n{sep}")
    print("DIAGNOSTIC 7: Rolling Sharpe")
    print(sep)
    _diag_rolling_sharpe(results_df, output_dir)

    print(f"\n{sep}")
    print("DIAGNOSTIC 8: Quantstats tearsheets")
    print(sep)
    _diag_quantstats(results_df, output_dir)


# ---------------------------------------------------------------------------
# Cumulative return plot
# ---------------------------------------------------------------------------

def plot_comparison(results_df, output_dir="examples/market/figures"):
    """Two cumulative return charts: momentum strategies and MVO strategies."""
    panels = [
        (
            {
                "cssp_momentum":      ("CSSP-Momentum",               "darkorange", 2.2),
                "full_momentum":      ("Full-Universe Momentum",       "steelblue",  1.8),
                "marketcap_momentum": ("Market-Cap-Filtered Momentum", "green",      1.6),
                "random_momentum":    ("Random-k Momentum",            "purple",     1.2),
                "equal_weight":       ("Equal-Weight (benchmark)",     "gray",       1.0),
            },
            "Momentum strategies (2yr train, 1mo test, 12-1 signal)",
            "backtest_momentum.png",
        ),
        (
            {
                "cssp_mvo":      ("CSSP-MVO  (cond≈592)",        "darkorange", 2.4),
                "marketcap_mvo": ("MarketCap-MVO  (cond≈8K)",    "steelblue",  1.8),
                "full_mvo":      ("Full-MVO  (cond≈2M, unstable)","red",        1.4),
                "equal_weight":  ("Equal-Weight (benchmark)",     "gray",       1.0),
            },
            "MVO strategies — does better conditioning help? (2yr train, 1mo test)",
            "backtest_mvo.png",
        ),
    ]

    for style, title, fname in panels:
        _, ax = plt.subplots(figsize=(13, 6))
        for col, (label, color, lw) in style.items():
            if col not in results_df.columns:
                continue
            cum = results_df[col].cumsum() * 100
            ax.plot(cum.index, cum.values, label=label, color=color, linewidth=lw)
        ax.axhline(0, color="black", lw=0.5, linestyle="--", alpha=0.4)
        ax.set_ylabel("Cumulative log return (%)")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        path = os.path.join(output_dir, fname)
        plt.savefig(path, dpi=150)
        print(f"Saved → {path}")
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prices_df = load_sp500_prices_df()
    results_df, summary_df, metadata = run_comparison(prices_df, k=50, top_n=20, verbose=True)
    plot_comparison(results_df)
    analyse(results_df, metadata)

