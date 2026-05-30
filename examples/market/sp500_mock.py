from sp500 import (
    compute_covariance,
    compute_log_returns,
    solve_sp500_fw_homotopy,
    solve_sp500_greedy,
)
from sp500_load_data import load_sp500_prices_df, START_DATE, END_DATE

prices_df = load_sp500_prices_df(start=START_DATE, end=END_DATE)
stock_names = prices_df.columns.tolist()

log_return_matrix = compute_log_returns(prices_df)
A = compute_covariance(log_return_matrix)
k = 50

fw_indices = solve_sp500_fw_homotopy(A=A, k=k, stock_names=stock_names)
greedy_indices = solve_sp500_greedy(A=A, k=k, stock_names=stock_names)
