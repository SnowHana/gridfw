import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import subprocess
import tempfile

# Try to import seaborn for better heatmaps, fallback to matplotlib
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Configuration
LOG_FILE = "logs/param_sweep_log.csv"
PLOT_DIR = "logs/plots"

def compile_to_pdf(latex_snippet, filename):
    """Wraps a LaTeX snippet in a document and compiles it to PDF."""
    template = r"""
\documentclass[varwidth=\maxdimen]{standalone}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{graphicx}
\usepackage{xcolor}
\begin{document}
%s
\end{document}
""" % latex_snippet

    os.makedirs(PLOT_DIR, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = os.path.join(tmpdir, "table.tex")
        with open(tex_path, "w") as f:
            f.write(template)
        
        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
            )
            pdf_path = os.path.join(tmpdir, "table.pdf")
            final_path = os.path.join(PLOT_DIR, filename)
            os.replace(pdf_path, final_path)
            print(f"Successfully exported PDF table to: {final_path}")
        except Exception as e:
            print(f"Warning: Failed to compile PDF for {filename}. (Error: {e})")

def load_and_clean_data():
    if not os.path.exists(LOG_FILE):
        print(f"Error: Log file not found at {LOG_FILE}")
        sys.exit(1)
    
    df = pd.read_csv(LOG_FILE)
    df.columns = df.columns.str.strip()
    
    # Filter outliers (same logic as plot_graph.py)
    # Deduplicate: keep fastest run for each (Dataset, Experiment, k)
    df = df.loc[df.groupby(['Dataset', 'Experiment', 'k'])['Greedy_Time_s'].idxmin()]
    
    # Growth filter (simplified for global analysis)
    # We'll just remove the extreme outliers > 5000s for a cleaner correlation
    df = df[df['Greedy_Time_s'] < 5000]
    
    return df

def analyze_correlation(df):
    # Select numeric columns for correlation
    cols = ['p', 'k', 'Steps', 'Samples', 'Greedy_Obj', 'FW_Obj', 'Ratio', 'Greedy_Time_s', 'FW_Time_s', 'Speedup_x']
    # Ensure columns exist
    cols = [c for c in cols if c in df.columns]
    
    corr_matrix = df[cols].corr()
    
    # 1. Generate Heatmap
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.figure(figsize=(12, 10))
    
    if HAS_SEABORN:
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    else:
        plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
        plt.colorbar()
        # Add labels manually if no seaborn
        plt.xticks(range(len(cols)), cols, rotation=45)
        plt.yticks(range(len(cols)), cols)
        for i in range(len(cols)):
            for j in range(len(cols)):
                plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha='center', va='center')

    plt.title("Parameter Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "correlation_heatmap.pdf"))
    plt.close()
    print(f"Generated heatmap in {PLOT_DIR}/correlation_heatmap.pdf")
    
    # 2. Export LaTeX Table
    print("\n% --- PARAMETER CORRELATION TABLE ---")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Pearson Correlation Matrix: Parameters vs. Performance}")
    print("\\label{tab:correlation}")
    print("\\small")
    print("\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}")
    print("\\toprule")
    
    # Header
    header = " & " + " & ".join([f"\\rotatebox{{90}}{{{c.replace('_', '\\_')}}}" for c in cols]) + " \\\\ \\midrule"
    print(header)
    
    for i, row_name in enumerate(cols):
        row_vals = [f"{corr_matrix.iloc[i, j]:.2f}" for j in range(len(cols))]
        # Bold the diagonal
        row_vals[i] = f"\\textbf{{{row_vals[i]}}}"
        print(f"{row_name.replace('_', '\\_')} & " + " & ".join(row_vals) + " \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Capture for PDF
    latex_snippet = "\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}\n\\toprule\n"
    latex_snippet += " & " + " & ".join([f"\\rotatebox{{90}}{{{c.replace('_', '\\_')}}}" for c in cols]) + " \\\\ \\midrule\n"
    for i, row_name in enumerate(cols):
        row_vals = [f"{corr_matrix.iloc[i, j]:.2f}" for j in range(len(cols))]
        row_vals[i] = f"\\textbf{{{row_vals[i]}}}"
        latex_snippet += f"{row_name.replace('_', '\\_')} & " + " & ".join(row_vals) + " \\\\\n"
    latex_snippet += "\\bottomrule\n\\end{tabular}"
    compile_to_pdf(latex_snippet, "correlation_matrix_table.pdf")

if __name__ == "__main__":
    print("Analyzing correlations...")
    df = load_and_clean_data()
    analyze_correlation(df)
