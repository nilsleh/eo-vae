import argparse
import json
import os


def load_results(results_dir):
    """Load all metric JSON files from the results directory."""
    results = {}

    modalities = ['S2L2A', 'S2RGB', 'S1RTC']

    for modality in modalities:
        json_path = os.path.join(results_dir, f'metrics_{modality}.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                results[modality] = json.load(f)
        else:
            print(f'Warning: {json_path} not found, skipping {modality}')

    return results


def combine_results(results):
    """Combine results from all modalities into a single structure."""
    combined = {}

    # Get all unique model names
    model_names = set()
    for modality_results in results.values():
        model_names.update(modality_results.keys())

    # Organize by model -> modality -> metrics
    for model in sorted(model_names):
        combined[model] = {}
        for modality, modality_results in results.items():
            if model in modality_results:
                combined[model][modality] = modality_results[model]

    return combined


def generate_latex_table(combined_results, output_path):
    """Generate a LaTeX table from combined results."""
    # Only include modalities that have been computed
    all_modalities = set()
    for model_data in combined_results.values():
        all_modalities.update(model_data.keys())

    # Sort modalities for consistent ordering
    modalities = sorted(all_modalities)

    if not modalities:
        print('Warning: No modalities found in results!')
        return ''

    def get_metrics_for_modality(mod):
        base = ['RMSE', 'PSNR', 'SSIM', 'SAM']
        if 'S2' in mod:
            return base + ['NDVI_MAE']
        return base

    # Start building the LaTeX table
    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Model Performance Across Modalities}')
    lines.append(r'\label{tab:model_performance}')
    lines.append(r'\resizebox{\textwidth}{!}{%')

    # Create column specification
    col_specs = []
    for mod in modalities:
        col_specs.append('c' * len(get_metrics_for_modality(mod)))
    col_spec = 'l|' + '|'.join(col_specs)

    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\hline')

    # Header row 1: Modality names
    header1 = r'\textbf{Model}'
    for modality in modalities:
        cols = len(get_metrics_for_modality(modality))
        header1 += r' & \multicolumn{' + str(cols) + r'}{c|}{\textbf{' + modality + '}}'
    header1 += r' \\'
    lines.append(header1)

    # Header row 2: Metric names
    header2 = ''
    for modality in modalities:
        for metric in get_metrics_for_modality(modality):
            header2 += r' & \textbf{' + metric.replace('_', r'\_') + r'}'
    header2 += r' \\'
    lines.append(header2)
    lines.append(r'\hline')
    lines.append(r'\hline')

    # Data rows
    for model_name in sorted(combined_results.keys()):
        model_data = combined_results[model_name]
        row = model_name.replace('_', r'\_')

        for modality in modalities:
            mod_metrics = get_metrics_for_modality(modality)
            if modality in model_data:
                metrics_data = model_data[modality]
                for metric in mod_metrics:
                    val = metrics_data.get(metric)
                    if isinstance(val, (int, float)):
                        fmt = '{:.2f}' if metric == 'PSNR' else '{:.4f}'
                        row += f' & {fmt.format(val)}'
                    else:
                        row += ' & -'
            else:
                row += ' & -' * len(mod_metrics)

        row += r' \\'
        lines.append(row)

    lines.append(r'\hline')
    lines.append(r'\end{tabular}')
    lines.append(r'}')
    lines.append(r'\end{table}')

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return '\n'.join(lines)


def generate_compact_latex_table(combined_results, output_path):
    """Generate a clean LaTeX table using booktabs and tabular* to fix Overfull \hbox."""
    all_modalities = sorted(
        set(
            mod for model_data in combined_results.values() for mod in model_data.keys()
        )
    )

    if not all_modalities:
        return ''

    def get_metrics_for_modality(mod):
        base = ['RMSE', 'PSNR', 'SSIM', 'SAM']
        if 'S2' in mod:
            return base + ['NDVI_MAE']
        return base

    # Find best values for bolding
    best_values = {mod: {} for mod in all_modalities}
    for modality in all_modalities:
        for metric in get_metrics_for_modality(modality):
            # Collect values for this (modality, metric) across all models
            values = []
            for model_name, model_data in combined_results.items():
                if modality in model_data and metric in model_data[modality]:
                    values.append(model_data[modality][metric])

            if values:
                # Determine if min or max is better
                if metric in ['RMSE', 'SAM', 'NDVI_MAE']:
                    best_values[modality][metric] = min(values)
                else:
                    best_values[modality][metric] = max(values)
            else:
                best_values[modality][metric] = None

    lines = []
    lines.append(r'\begin{table*}[h]')
    lines.append(r'\centering')
    lines.append(r'\resizebox{\linewidth}{!}{')

    # Use tabular instead of tabular* to let resizebox handle the width
    total_cols = sum(len(get_metrics_for_modality(mod)) for mod in all_modalities)
    # 1 label column + total_cols data columns
    num_cols = 1 + total_cols
    lines.append(r'\begin{tabular}{l' + 'c' * (num_cols - 1) + r'}')
    lines.append(r'\toprule')

    # Header row 1: Modalities
    header1 = r'\textbf{Model}'
    for modality in all_modalities:
        cols = len(get_metrics_for_modality(modality))
        header1 += r' & \multicolumn{' + str(cols) + r'}{c}{\textbf{' + modality + '}}'
    header1 += r' \\'
    lines.append(header1)

    # Dynamic cmidrules for clean separation
    midrules = []
    current_col = 2  # Start after Model column
    for modality in all_modalities:
        cols = len(get_metrics_for_modality(modality))
        end_col = current_col + cols - 1
        midrules.append(f'\\cmidrule(lr){{{current_col}-{end_col}}}')
        current_col += cols
    lines.append(' '.join(midrules))

    # Header row 2: Metrics
    header2 = ''
    for modality in all_modalities:
        for metric in get_metrics_for_modality(modality):
            arrow = (
                r'$\downarrow$'
                if metric in ['RMSE', 'SAM', 'NDVI_MAE']
                else r'$\uparrow$'
            )
            header2 += r' & ' + metric.replace('_', r'\_') + arrow
    header2 += r' \\'
    lines.append(header2)
    lines.append(r'\midrule')

    # Data rows
    for model_name in sorted(combined_results.keys()):
        model_data = combined_results[model_name]
        row = model_name.replace('_', r'\_')

        for modality in all_modalities:
            mod_metrics = get_metrics_for_modality(modality)
            if modality in model_data:
                metrics_data = model_data[modality]
                for metric in mod_metrics:
                    if metric in metrics_data:
                        val = metrics_data[metric]
                        # Check if best
                        is_best = False
                        best = best_values[modality].get(metric)
                        if best is not None:
                            is_best = abs(val - best) < 1e-6

                        fmt = f'{val:.4f}' if metric != 'PSNR' else f'{val:.2f}'
                        row += f' & \\textbf{{{fmt}}}' if is_best else f' & {fmt}'
                    else:
                        row += ' & -'
            else:
                row += ' & -' * len(mod_metrics)
        row += r' \\'
        lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}}')
    lines.append(r'\caption{Reconstruction Performance Across Modalities}')
    lines.append(r'\label{tab:reconstruction_performance}')
    lines.append(r'\end{table*}')

    compact_path = output_path.replace('.tex', '_compact.tex')
    with open(compact_path, 'w') as f:
        f.write('\n'.join(lines))

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Combine evaluation results and generate LaTeX table'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing the metric JSON files',
    )
    args = parser.parse_args()

    print(f'Loading results from: {args.results_dir}')

    # Load individual results
    results = load_results(args.results_dir)

    if not results:
        print('Error: No results found!')
        return

    print(f'Found results for {len(results)} modalities: {list(results.keys())}')

    # Combine results
    combined = combine_results(results)

    # Save combined results
    combined_path = os.path.join(args.results_dir, 'combined_results.json')
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=4)
    print(f'Saved combined results to: {combined_path}')

    # Generate LaTeX table
    latex_path = os.path.join(args.results_dir, 'results_table.tex')
    generate_latex_table(combined, latex_path)
    print(f'Generated LaTeX table: {latex_path}')

    # Generate compact LaTeX table with bold best values
    compact_latex = generate_compact_latex_table(combined, latex_path)
    compact_path = latex_path.replace('.tex', '_compact.tex')
    print(f'Generated compact LaTeX table: {compact_path}')

    # Print summary
    print('\n' + '=' * 50)
    print('SUMMARY')
    print('=' * 50)
    for model_name, model_data in combined.items():
        print(f'\n{model_name}:')
        for modality in sorted(model_data.keys()):
            metrics_str = ', '.join(
                [f'{k}: {v:.4f}' for k, v in model_data[modality].items()]
            )
            print(f'  {modality}: {metrics_str}')

    print('\n' + '=' * 50)
    print('Preview of LaTeX table (compact version):')
    print('=' * 50)
    print(compact_latex)


if __name__ == '__main__':
    main()
