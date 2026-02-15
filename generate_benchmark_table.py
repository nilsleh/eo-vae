"""Generate a clean, publication-ready benchmark comparison table.
Focuses on the most important metrics for the paper.

Compatible with the simplified benchmark output format (no FLOPs).
"""

import argparse
import json
import os
import sys


def load_results(json_path):
    """Load benchmark results from JSON."""
    with open(json_path) as f:
        data = json.load(f)

    # Handle both old format (list) and new format (dict with metadata)
    if isinstance(data, list):
        return data, None
    elif isinstance(data, dict):
        if 'results' in data:
            return data['results'], data.get('metadata', None)
        else:
            # Old format wrapped in dict somehow
            return list(data.values()), None
    else:
        raise ValueError('Unexpected JSON format')


def generate_main_table(results, output_path):
    """Generate main comparison table with key metrics.
    Focuses on: performance, memory, and model size.
    """
    lines = []

    # Header
    lines.append('=' * 100)
    lines.append('BENCHMARK COMPARISON TABLE - Main Results')
    lines.append('=' * 100)
    lines.append('')

    # Column headers
    header = f'{"Model":<25} {"Total Time":<15} {"Throughput":<15} {"Peak Memory":<15} {"Parameters":<15}'
    lines.append(header)
    lines.append(f'{"":25} {"(ms)":<15} {"(imgs/sec)":<15} {"(GB)":<15} {"(M)":<15}')
    lines.append('-' * 100)

    # Find baseline (pixel model) for relative metrics
    baseline = None
    for r in results:
        if r['model_type'] == 'pixel':
            baseline = r
            break

    if not baseline:
        baseline = results[-1]  # Use last model as baseline if no pixel model

    baseline_time = baseline['timing_ms']['total']
    baseline_memory = baseline['memory_gb']['peak_memory']

    # Data rows
    for r in results:
        name = r['name']
        total_time = r['timing_ms']['total']
        throughput = r['throughput_imgs_per_sec']
        peak_mem = r['memory_gb']['peak_memory']
        total_params = r['parameters']['total'] / 1e6
        sr_params = r['parameters']['sr_model'] / 1e6

        # Format: Total (SR)
        params_str = f'{total_params:.2f} ({sr_params:.2f})'

        line = f'{name:<25} {total_time:<15.2f} {throughput:<15.2f} {peak_mem:<15.3f} {params_str:<15}'
        lines.append(line)

    lines.append('-' * 100)
    lines.append('')

    # Relative improvements vs baseline
    lines.append('Relative to Baseline (Pixel Model):')
    lines.append('-' * 100)

    header = f'{"Model":<25} {"Speed Improvement":<20} {"Memory Reduction":<20} {"Speedup Factor":<20}'
    lines.append(header)
    lines.append('-' * 100)

    for r in results:
        name = r['name']
        total_time = r['timing_ms']['total']
        peak_mem = r['memory_gb']['peak_memory']

        speedup = baseline_time / total_time if total_time > 0 else 0
        mem_ratio = baseline_memory / peak_mem if peak_mem > 0 else 0

        if r == baseline:
            line = f'{name:<25} {"1.00x (baseline)":<20} {"1.00x (baseline)":<20} {"-":<20}'
        else:
            line = f'{name:<25} {f"{speedup:.2f}x":<20} {f"{mem_ratio:.2f}x":<20} {f"{speedup:.1f}x faster":<20}'
        lines.append(line)

    lines.append('=' * 100)
    lines.append('')

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return lines


def generate_detailed_table(results, output_path):
    """Generate detailed table with timing breakdown."""
    lines = []

    # Header
    lines.append('=' * 120)
    lines.append('DETAILED BENCHMARK TABLE - Timing Breakdown')
    lines.append('=' * 120)
    lines.append('')

    # Timing breakdown
    lines.append('TIMING BREAKDOWN (milliseconds):')
    lines.append('-' * 120)
    header = f'{"Model":<25} {"Encode":<12} {"SR Forward":<12} {"Decode":<12} {"Total":<12} {"% SR":<10} {"% VAE":<10}'
    lines.append(header)
    lines.append('-' * 120)

    for r in results:
        name = r['name']
        enc_time = r['timing_ms']['encode']
        sr_time = r['timing_ms']['sr_forward']
        dec_time = r['timing_ms']['decode']
        total_time = r['timing_ms']['total']

        pct_sr = (sr_time / total_time * 100) if total_time > 0 else 0
        pct_vae = ((enc_time + dec_time) / total_time * 100) if total_time > 0 else 0

        line = f'{name:<25} {enc_time:<12.2f} {sr_time:<12.2f} {dec_time:<12.2f} {total_time:<12.2f} {pct_sr:<10.1f} {pct_vae:<10.1f}'
        lines.append(line)

    lines.append('')
    lines.append('')

    # Parameter breakdown
    lines.append('PARAMETERS (Millions):')
    lines.append('-' * 120)
    header = f'{"Model":<25} {"Encoder":<12} {"SR Model":<12} {"Decoder":<12} {"Total":<12} {"Model Type":<20}'
    lines.append(header)
    lines.append('-' * 120)

    for r in results:
        name = r['name']
        enc_params = r['parameters']['encoder'] / 1e6
        sr_params = r['parameters']['sr_model'] / 1e6
        dec_params = r['parameters']['decoder'] / 1e6
        total_params = r['parameters']['total'] / 1e6
        model_type = r['model_type']

        line = f'{name:<25} {enc_params:<12.2f} {sr_params:<12.2f} {dec_params:<12.2f} {total_params:<12.2f} {model_type:<20}'
        lines.append(line)

    lines.append('=' * 120)
    lines.append('')

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return lines


def generate_latex_table(results, metadata, output_path, custom_caption=None):
    """Generate a LaTeX table snippet that can be easily integrated into a paper.
    Uses booktabs style for professional appearance.
    """
    lines = []

    lines.append('% LaTeX table for paper - paste into your document')
    lines.append('% Requires: \\usepackage{booktabs}')
    lines.append('')

    # Generate caption
    if custom_caption:
        caption_text = custom_caption
    else:
        caption_text = 'Computational comparison of diffusion models for satellite image super-resolution. '
        caption_text += 'All models benchmarked on complete pixel-to-pixel pipeline. '
        caption_text += 'Latent models include VAE encoding and decoding. '
        if metadata:
            caption_text += f'Timing averaged over {metadata.get("n_iterations", "N/A")} iterations.'

    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append(f'\\caption{{{caption_text}}}')
    lines.append('\\label{tab:computational_comparison}')
    lines.append('\\begin{tabular}{lrrrr}')
    lines.append('\\toprule')
    lines.append(
        '\\textbf{Model} & \\textbf{Time (ms)} & \\textbf{Throughput} & \\textbf{Memory (GB)} & \\textbf{Params (M)} \\\\'
    )
    lines.append(
        '               &                     & (imgs/sec)           &                      & Total (Diffusion) \\\\'
    )
    lines.append('\\midrule')

    # Find baseline (pixel model)
    baseline = None
    for r in results:
        if r['model_type'] == 'pixel':
            baseline = r
            break
    if not baseline:
        baseline = results[-1]

    baseline_time = baseline['timing_ms']['total']

    # Data rows
    for r in results:
        name = r['name'].replace('_', '\\_')
        total_time = r['timing_ms']['total']
        throughput = r['throughput_imgs_per_sec']
        peak_mem = r['memory_gb']['peak_memory']
        total_params = r['parameters']['total'] / 1e6
        sr_params = r['parameters']['sr_model'] / 1e6

        # Add speedup annotation for non-baseline
        if r != baseline:
            speedup = baseline_time / total_time
            name_with_speedup = f'{name} ({speedup:.1f}$\\times$)'
        else:
            name_with_speedup = f'{name} (baseline)'

        # Format parameters: Total (SR)
        params_str = f'{total_params:.1f} ({sr_params:.1f})'

        line = f'{name_with_speedup} & {total_time:.1f} & {throughput:.2f} & {peak_mem:.2f} & {params_str} \\\\'
        lines.append(line)

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return lines


def generate_latex_detailed_table(results, metadata, output_path):
    """Generate a detailed LaTeX table with timing breakdown.
    Good for supplementary material.
    """
    lines = []

    lines.append('% Detailed LaTeX table with timing breakdown')
    lines.append('% Requires: \\usepackage{booktabs}')
    lines.append('')

    caption = 'Detailed timing breakdown for diffusion model inference. '
    caption += (
        'Encode and Decode times are for VAE operations (zero for pixel-space models).'
    )

    lines.append('\\begin{table}[t]')
    lines.append('\\centering')
    lines.append(f'\\caption{{{caption}}}')
    lines.append('\\label{tab:timing_breakdown}')
    lines.append('\\begin{tabular}{lrrrrr}')
    lines.append('\\toprule')
    lines.append(
        '\\textbf{Model} & \\textbf{Encode} & \\textbf{SR Forward} & \\textbf{Decode} & \\textbf{Total} & \\textbf{Speedup} \\\\'
    )
    lines.append('               & (ms) & (ms) & (ms) & (ms) & \\\\')
    lines.append('\\midrule')

    # Find baseline
    baseline = None
    for r in results:
        if r['model_type'] == 'pixel':
            baseline = r
            break
    if not baseline:
        baseline = results[-1]

    baseline_time = baseline['timing_ms']['total']

    for r in results:
        name = r['name'].replace('_', '\\_')
        enc = r['timing_ms']['encode']
        sr = r['timing_ms']['sr_forward']
        dec = r['timing_ms']['decode']
        total = r['timing_ms']['total']

        if r == baseline:
            speedup_str = '1.0$\\times$'
        else:
            speedup = baseline_time / total
            speedup_str = f'{speedup:.1f}$\\times$'

        line = f'{name} & {enc:.1f} & {sr:.1f} & {dec:.1f} & {total:.1f} & {speedup_str} \\\\'
        lines.append(line)

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\end{table}')

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return lines


def print_summary_stats(results):
    """Print key summary statistics to console."""
    print('\n' + '=' * 80)
    print('KEY FINDINGS')
    print('=' * 80)

    # Find baseline and latent models
    baseline = None
    latent_models = []

    for r in results:
        if r['model_type'] == 'pixel':
            baseline = r
        elif r['model_type'] in ['eo-vae', 'flux-vae']:
            latent_models.append(r)

    if not baseline and results:
        baseline = results[-1]

    if baseline:
        baseline_time = baseline['timing_ms']['total']
        baseline_mem = baseline['memory_gb']['peak_memory']

        print(f'\nBaseline ({baseline["name"]}):')
        print(f'  - Inference time: {baseline_time:.2f} ms')
        print(f'  - Throughput: {baseline["throughput_imgs_per_sec"]:.2f} imgs/sec')
        print(f'  - Peak memory: {baseline_mem:.3f} GB')

        for latent in latent_models:
            latent_time = latent['timing_ms']['total']
            latent_mem = latent['memory_gb']['peak_memory']

            speedup = baseline_time / latent_time if latent_time > 0 else 0
            mem_ratio = baseline_mem / latent_mem if latent_mem > 0 else 0

            print(f'\n{latent["name"]}:')
            print(f'  - Speed improvement: {speedup:.2f}x faster')
            print(
                f'  - Memory ratio: {mem_ratio:.2f}x {"less" if mem_ratio > 1 else "more"}'
            )
            print('  - Timing breakdown:')
            print(
                f'      Encode: {latent["timing_ms"]["encode"]:.2f} ms ({latent["timing_ms"]["encode"] / latent_time * 100:.1f}%)'
            )
            print(
                f'      SR:     {latent["timing_ms"]["sr_forward"]:.2f} ms ({latent["timing_ms"]["sr_forward"] / latent_time * 100:.1f}%)'
            )
            print(
                f'      Decode: {latent["timing_ms"]["decode"]:.2f} ms ({latent["timing_ms"]["decode"] / latent_time * 100:.1f}%)'
            )

    # Compare latent models to each other
    if len(latent_models) >= 2:
        print('\nLatent Model Comparison:')
        m1, m2 = latent_models[0], latent_models[1]
        ratio = (
            m1['timing_ms']['total'] / m2['timing_ms']['total']
            if m2['timing_ms']['total'] > 0
            else 0
        )
        print(f'  {m1["name"]} vs {m2["name"]}: {ratio:.2f}x')
        print('  (Values close to 1.0 indicate similar performance)')

    print('\n' + '=' * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Generate benchmark comparison tables from JSON results'
    )
    parser.add_argument(
        '--results', type=str, required=True, help='Path to benchmark_results.json'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: same as results file)',
    )
    parser.add_argument(
        '--caption',
        type=str,
        default=None,
        help='Custom caption for LaTeX table (optional)',
    )
    args = parser.parse_args()

    # Load results
    try:
        results, metadata = load_results(args.results)
    except Exception as e:
        print(f'Error loading results: {e}')
        sys.exit(1)

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.results)
        if not args.output_dir:
            args.output_dir = '.'

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate tables
    print(f'Generating benchmark tables from {args.results}')
    print(f'Output directory: {args.output_dir}\n')

    if metadata:
        print('Benchmark metadata:')
        print(f'  Date: {metadata.get("date", "N/A")}')
        print(f'  Batch size: {metadata.get("batch_size", "N/A")}')
        print(f'  Iterations: {metadata.get("n_iterations", "N/A")}')
        print(f'  Device: {metadata.get("device", "N/A")}')
        if 'note' in metadata:
            print(f'  Note: {metadata.get("note")}')
        print()

    # Main table (plain text)
    main_table_path = os.path.join(args.output_dir, 'table_main.txt')
    generate_main_table(results, main_table_path)
    print(f'Generated main table: {main_table_path}')

    # Detailed table (plain text)
    detailed_table_path = os.path.join(args.output_dir, 'table_detailed.txt')
    generate_detailed_table(results, detailed_table_path)
    print(f'Generated detailed table: {detailed_table_path}')

    # LaTeX main table
    latex_path = os.path.join(args.output_dir, 'table_latex.tex')
    generate_latex_table(results, metadata, latex_path, custom_caption=args.caption)
    print(f'Generated LaTeX table: {latex_path}')

    # LaTeX detailed table
    latex_detailed_path = os.path.join(args.output_dir, 'table_latex_detailed.tex')
    generate_latex_detailed_table(results, metadata, latex_detailed_path)
    print(f'Generated LaTeX detailed table: {latex_detailed_path}')

    # Print summary to console
    print_summary_stats(results)

    print(f'\nAll tables generated successfully in: {args.output_dir}')


if __name__ == '__main__':
    main()
