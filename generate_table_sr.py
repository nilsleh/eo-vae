import json
import os


def main():
    # File path from your attachment
    file_path = '/mnt/SSD2/nils/eo-vae/results/sr-metrics/all_metrics.json'

    if not os.path.exists(file_path):
        print(f'Error: File not found at {file_path}')
        # Fallback to local if running elsewhere
        file_path = 'all_metrics.json'
        if not os.path.exists(file_path):
            return

    with open(file_path) as f:
        data = json.load(f)

    # Define Metrics, Direction, and Column Order
    # (Name, Direction) -> Direction: 'max' for up arrow (higher is better), 'min' for down arrow
    metrics_config = [('PSNR', 'max'), ('SSIM', 'max'), ('RMSE', 'min'), ('SAM', 'min')]

    # Sort models alphabetically or define custom order
    models = sorted(data.keys())

    # --- 1. Find Best Values ---
    best_values = {}
    for metric, direction in metrics_config:
        # Collect valid values for this metric across all models
        values = []
        for model in models:
            if metric in data[model]:
                values.append(data[model][metric])

        if not values:
            best_values[metric] = None
        elif direction == 'max':
            best_values[metric] = max(values)
        else:  # min
            best_values[metric] = min(values)

    # --- 2. Generate LaTeX ---
    print('% Copy the following block into your LaTeX document')
    print(r'\begin{table}[h]')
    print(r'    \centering')
    print(r'    \begin{tabular}{lcccc}')
    print(r'    \toprule')

    # Header Row with Arrows
    header_cells = ['    Model']
    for metric, direction in metrics_config:
        arrow = r'$\uparrow$' if direction == 'max' else r'$\downarrow$'
        header_cells.append(f'{metric}{arrow}')

    header = ' & '.join(header_cells) + r' \\'
    print(header)
    print(r'    \midrule')

    # Data Rows
    for model_name in models:
        metrics = data[model_name]

        # Clean up Model Name (optional, e.g., escape underscores)
        display_name = model_name.replace('_', r'\_')

        row_cells = [display_name]

        for metric, direction in metrics_config:
            val = metrics.get(metric, 0.0)
            target = best_values.get(metric)

            # Formatting string
            if metric == 'PSNR':
                val_str = f'{val:.2f}'
            else:
                val_str = f'{val:.4f}'

            # Determine if bold
            is_best = False
            if target is not None:
                # Floating point comparison
                if abs(val - target) < 1e-9:
                    is_best = True

            if is_best:
                row_cells.append(r'\textbf{' + val_str + '}')
            else:
                row_cells.append(val_str)

        print('    ' + ' & '.join(row_cells) + r' \\')

    print(r'    \bottomrule')
    print(r'    \end{tabular}')
    print(
        r'    \caption{Super-Resolution Quantitative Results. Comparison of EO-VAE, FluxVAE and Pixel Diffusion baselines.}'
    )
    print(r'    \label{tab:sr_results}')
    print(r'\end{table}')


if __name__ == '__main__':
    main()
