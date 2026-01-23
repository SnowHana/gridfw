import pandas as pd
import os
import subprocess
import tempfile

# Configuration
GRAD_LOG = "logs/grad_test_log.csv"
SWEEP_LOG = "logs/param_sweep_log.csv"
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
            # Run pdflatex twice for proper sizing/labels if needed
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
            print("Make sure 'pdflatex' and 'standalone' package are installed.")

def export_grad_table():
    if not os.path.exists(GRAD_LOG):
        print(f"Warning: {GRAD_LOG} not found.")
        return

    df = pd.read_csv(GRAD_LOG)
    
    # Clean up test names for display
    # e.g. tests/grad/test_f_grad.py::test_f_stress_ill_conditioned[1.0-50] -> test_f_stress_ill_conditioned[1.0-50]
    df['Test_Name'] = df['Test_Name'].str.split('::').str[-1]
    
    # Group by the base test name (before the bracket)
    df['Category'] = df['Test_Name'].str.split('[').str[0]
    
    print("\n% --- GRADIENT VERIFICATION TABLE ---")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Gradient Verification Results}")
    print("\\label{tab:grad_verif}")
    print("\\begin{tabular}{@{}llc@{}}")
    print("\\toprule")
    print("Test Category & Specific Case & Outcome \\\\ \\midrule")
    
    current_cat = ""
    for _, row in df.iterrows():
        cat = row['Category']
        name = row['Test_Name']
        outcome = f"\\textbf{{{row['Outcome']}}}" if row['Outcome'] == 'PASSED' else row['Outcome']
        
        if cat != current_cat:
            if current_cat != "":
                print("\\addlinespace")
            display_cat = cat.replace('_', '\\_')
            current_cat = cat
        else:
            display_cat = ""
            
        display_name = name.replace('_', '\\_').replace('[', ' (').replace(']', ')')
        print(f"{display_cat} & \\texttt{{{display_name}}} & {outcome} \\\\")
        
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Capture for PDF
    latex_snippet = f"\\begin{{tabular}}{{@{{}}llc@{{}}}}\n\\toprule\nTest Category & Specific Case & Outcome \\\\ \\midrule\n"
    current_cat = ""
    for _, row in df.iterrows():
        cat = row['Category']
        name = row['Test_Name']
        outcome = f"\\textbf{{{row['Outcome']}}}" if row['Outcome'] == 'PASSED' else row['Outcome']
        if cat != current_cat:
            if current_cat != "": latex_snippet += "\\addlinespace\n"
            display_cat = cat.replace('_', '\\_')
            current_cat = cat
        else:
            display_cat = ""
        display_name = name.replace('_', '\\_').replace('[', ' (').replace(']', ')')
        latex_snippet += f"{display_cat} & \\texttt{{{display_name}}} & {outcome} \\\\\n"
    latex_snippet += "\\bottomrule\n\\end{tabular}"
    compile_to_pdf(latex_snippet, "grad_verification_table.pdf")

def export_sweep_summary():
    if not os.path.exists(SWEEP_LOG):
        print(f"Warning: {SWEEP_LOG} not found.")
        return

    df = pd.read_csv(SWEEP_LOG)
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Take the last run for each (Dataset, Experiment, k) to get the most recent results
    df = df.sort_values('Timestamp').groupby(['Dataset', 'Experiment', 'k']).last().reset_index()
    
    # Filter for a concise summary (e.g. just Secom vary_k)
    summary_df = df[df['Experiment'] == 'vary_k'].copy()
    
    if summary_df.empty:
        return

    print("\n% --- PARAMETER SWEEP SUMMARY (vary_k) ---")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Performance Summary: Greedy vs. FW-Homotopy (Secom)}")
    print("\\label{tab:sweep_summary}")
    print("\\begin{tabular}{@{}ccccc@{}}")
    print("\\toprule")
    print("$k$ & Greedy Obj. & FW Obj. & Ratio & Speedup \\\\ \\midrule")
    
    # Select a subset of k values to keep the table concise
    k_to_show = sorted(summary_df['k'].unique())
    # Show every 4th point or so
    k_to_show = k_to_show[::4] if len(k_to_show) > 10 else k_to_show

    for k in k_to_show:
        row = summary_df[summary_df['k'] == k].iloc[0]
        print(f"{k} & {row['Greedy_Obj']:.2f} & {row['FW_Obj']:.2f} & {row['Ratio']:.4f} & {row['Speedup_x']:.1f}$\\times$ \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Capture for PDF
    latex_snippet = f"\\begin{{tabular}}{{@{{}}ccccc@{{}}}}\n\\toprule\n$k$ & Greedy Obj. & FW Obj. & Ratio & Speedup \\\\ \\midrule\n"
    for k in k_to_show:
        row = summary_df[summary_df['k'] == k].iloc[0]
        latex_snippet += f"{k} & {row['Greedy_Obj']:.2f} & {row['FW_Obj']:.2f} & {row['Ratio']:.4f} & {row['Speedup_x']:.1f}$\\times$ \\\\\n"
    latex_snippet += "\\bottomrule\n\\end{tabular}"
    compile_to_pdf(latex_snippet, "performance_summary_table.pdf")

if __name__ == "__main__":
    print("% === COPY THIS TO YOUR OVERLEAF PREAMBLE ===")
    print("% \\usepackage{booktabs}")
    print("% \\usepackage{caption}")
    print("% ===========================================")
    
    export_grad_table()
    export_sweep_summary()
