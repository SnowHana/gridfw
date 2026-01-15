import pandas as pd
import os
import time


class GradientTestLogger:
    """Handles logging of test results, summary generation, and file saving."""

    def __init__(self, output_dir="logs"):
        self.logs = []
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def log_entry(
        self, test_name, p, cond_num, n_samples, max_rel_err, mean_rel_err, status
    ):
        self.logs.append(
            {
                "Test": test_name,
                "Dim": p,
                "Cond(A)": f"{cond_num:.1e}",
                "Samples": n_samples,
                "Max Rel": f"{max_rel_err:.2e}",
                "Mean Rel": f"{mean_rel_err:.2e}",
                "Status": status,
            }
        )

    def display_summary(self):
        if not self.logs:
            print("No logs to display.")
            return

        df = pd.DataFrame(self.logs)
        print("\n" + "=" * 100)
        print(f"VERIFICATION SUMMARY REPORT")
        print("=" * 100)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print(df.to_string(index=False))
        print("=" * 100 + "\n")

    def save_logs(self, filename_prefix="grad_test"):
        if not self.logs:
            return
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        pd.DataFrame(self.logs).to_csv(filepath, index=False)
        print(f"Logs saved to: {filepath}")
