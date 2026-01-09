import json
import pandas as pd
import argparse
import os


def load_metrics(json_path):
    """Loads metrics from a JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f'Could not find metrics file: {json_path}')

    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def format_latex_table(data, caption='Evaluation Metrics', label='tab:metrics'):
    """
    Converts a dictionary of metrics into a Booktabs-formatted LaTeX table.
    """
    df = pd.DataFrame.from_dict(data, orient='index')

    # 1. Sort/Select Columns
    desired_cols = ['MSE', 'PSNR', 'SSIM']
    # Keep only columns that exist, preserve desired order, append extras at the end
    cols = [c for c in desired_cols if c in df.columns] + [
        c for c in df.columns if c not in desired_cols
    ]
    df = df[cols]

    # 2. Define Directionality (True = Higher is Better)
    metric_direction = {col: True for col in df.columns}

    # False = Lower is Better (Error metrics)
    lower_is_better_metrics = ['MSE', 'MAE', 'RMSE', 'SAM']
    for m in lower_is_better_metrics:
        if m in df.columns:
            metric_direction[m] = False

    # 3. Format Columns (4 decimals + Bold Best)
    def format_column(s, is_higher_better):
        if is_higher_better:
            best_val = s.max()
        else:
            best_val = s.min()

        formatted_list = []
        for val in s:
            # Check for best value (using epsilon for float comparison)
            if abs(val - best_val) < 1e-9:
                formatted_list.append(f'\\textbf{{{val:.4f}}}')
            else:
                formatted_list.append(f'{val:.4f}')
        return formatted_list

    formatted_data = {}
    for col in df.columns:
        formatted_data[col] = format_column(df[col], metric_direction.get(col, True))

    df_formatted = pd.DataFrame(formatted_data, index=df.index)

    # 4. Construct LaTeX String (Booktabs Style)
    # Note: No vertical bars '|' in the column definition
    latex_str = '% Requires \\usepackage{booktabs} in your preamble\n'
    latex_str += '\\begin{table}[h]\n'
    latex_str += '    \\centering\n'
    latex_str += '    \\caption{' + caption + '}\n'
    latex_str += '    \\label{' + label + '}\n'

    # Column setup: Left align 'Model', Center align metrics
    # e.g., {lccc} NOT {|l|c|c|c|}
    col_def = 'l' + 'c' * len(cols)
    latex_str += '    \\begin{tabular}{' + col_def + '}\n'

    latex_str += '        \\toprule\n'

    # Header Row
    latex_str += '        Model & ' + ' & '.join(cols) + ' \\\\\n'
    latex_str += '        \\midrule\n'

    # Data Rows
    for idx, row in df_formatted.iterrows():
        # Escape underscores for LaTeX safety (e.g., model_v1 -> model\_v1)
        clean_name = str(idx).replace('_', '\\_')
        row_str = ' & '.join(row.values)
        latex_str += f'        {clean_name} & {row_str} \\\\\n'

    latex_str += '        \\bottomrule\n'
    latex_str += '    \\end{tabular}\n'
    latex_str += '\\end{table}'

    return latex_str


def main():
    parser = argparse.ArgumentParser(description='Convert JSON metrics to LaTeX table.')
    parser.add_argument('--input_file', type=str, help='Path to the metrics.json file')
    parser.add_argument(
        '--output', type=str, default=None, help='Output .tex file path'
    )
    parser.add_argument(
        '--caption', type=str, default='Performance Comparison', help='Table caption'
    )
    args = parser.parse_args()

    data = load_metrics(args.input_file)
    latex_table = format_latex_table(data, caption=args.caption)

    print('\n' + '=' * 30 + ' GENERATED LATEX ' + '=' * 30 + '\n')
    print(latex_table)
    print('\n' + '=' * 77 + '\n')

    if args.output:
        with open(args.output, 'w') as f:
            f.write(latex_table)
        print(f'Table saved to: {args.output}')


if __name__ == '__main__':
    main()
