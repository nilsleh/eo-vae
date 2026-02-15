import datetime
import glob
import os
import tarfile
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import zarr
from tqdm import tqdm

# --- CONFIGURATION ---
LOCAL_DATA_ROOT = '/mnt/SSD2/nils/datasets/terramesh'
CUTOFF_DATE = datetime.datetime(2022, 1, 25)

# 12 Sentinel-2 Bands (Standard L2A ordering)
CHANNEL_NAMES = [
    'B01',
    'B02',
    'B03',
    'B04',
    'B05',
    'B06',
    'B07',
    'B08',
    'B8A',
    'B09',
    'B11',
    'B12',
]

SEARCH_PATTERNS = {
    'train': os.path.join(LOCAL_DATA_ROOT, 'train', 'S2L1C', 'majortom_shard_*.tar'),
    'val': os.path.join(LOCAL_DATA_ROOT, 'val', 'S2L1C', 'majortom_shard_*.tar'),
}

# Increased sample count to ensure we get enough representation from both eras
SAMPLES_TO_CHECK = 10000


class RunningStats:
    """Accumulates histograms and moments for a set of channels."""

    def __init__(self, name, n_channels=13, bins=1000, r_min=-2000, r_max=15000):
        self.name = name
        self.n_channels = n_channels
        self.count = 0
        self.pixel_count = 0

        # Moments
        self.sum = np.zeros(n_channels, dtype=np.float64)
        self.sq_sum = np.zeros(n_channels, dtype=np.float64)
        self.min = np.full(n_channels, np.inf)
        self.max = np.full(n_channels, -np.inf)

        # Histogram
        self.bins = np.linspace(r_min, r_max, bins + 1)
        self.histograms = np.zeros((n_channels, bins), dtype=np.int64)

    def update(self, img):
        """Img shape: (C, H, W)"""
        # if img.shape[0] != self.n_channels:
        #     # Handle edge cases (e.g. if some file has different bands), skip for now
        #     return

        B, C, H, W = img.shape
        pixels = H * W
        self.count += 1
        self.pixel_count += pixels

        # Flatten spatial dims
        flat = img.reshape(C, -1)

        # Update Min/Max
        batch_min = flat.min(axis=1)
        batch_max = flat.max(axis=1)
        self.min = np.minimum(self.min, batch_min)
        self.max = np.maximum(self.max, batch_max)

        # Update Sums (for Mean/Std)
        self.sum += flat.sum(axis=1)
        self.sq_sum += (flat**2).sum(axis=1)

        # Update Histograms
        for c in range(C):
            hist, _ = np.histogram(flat[c], bins=self.bins)
            self.histograms[c] += hist

    def compute_final(self):
        mean = self.sum / self.pixel_count
        var = (self.sq_sum / self.pixel_count) - (mean**2)
        std = np.sqrt(np.maximum(var, 0))  # Clip neg epsilon
        return mean, std


def check_local_data():
    # Initialize stats containers
    stats_pre = RunningStats('Pre-2022 (Legacy)')
    stats_post = RunningStats('Post-2022 (Baseline 04.00)')

    # Collect files
    all_files = []
    for split, pattern in SEARCH_PATTERNS.items():
        files = sorted(glob.glob(pattern))
        # Filter (Same logic as before)
        for f in files:
            try:
                shard_id = f.split('_')[-1].split('.')[0]
                if split == 'train' and '000001' <= shard_id <= '000025':
                    all_files.append(f)
                elif split == 'val' and '000001' <= shard_id <= '000005':
                    all_files.append(f)
            except:
                continue

    if not all_files:
        raise FileNotFoundError('No files found.')

    print(f'Scanning {len(all_files)} shards...')
    pbar = tqdm(total=SAMPLES_TO_CHECK)
    total_processed = 0

    for tar_path in all_files:
        if total_processed >= SAMPLES_TO_CHECK:
            break

        with tarfile.open(tar_path, 'r') as tar:
            for member in tar:
                if total_processed >= SAMPLES_TO_CHECK:
                    break

                if member.name.endswith('.zarr.zip'):
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    content = f.read()

                    with tempfile.NamedTemporaryFile(suffix='.zip') as tmp:
                        tmp.write(content)
                        tmp.flush()

                        with zarr.open(
                            store=zarr.ZipStore(tmp.name, mode='r'), mode='r'
                        ) as z:
                            # --- 1. Date Parsing ---
                            ts_str = None
                            if 'time' in z.attrs:
                                ts_str = z.attrs['time']
                            elif 'time' in z:
                                t_val = z['time'][...]
                                ts_str = (
                                    str(t_val[0])
                                    if (
                                        isinstance(t_val, (np.ndarray, list))
                                        and len(t_val) > 0
                                    )
                                    else str(t_val)
                                )

                            if not ts_str:
                                continue

                            clean_ts = str(ts_str).strip(" []'")
                            if clean_ts.isdigit():
                                ts = datetime.datetime.fromtimestamp(
                                    int(clean_ts) / 1e9
                                )
                            elif isinstance(ts_str, (np.datetime64, np.timedelta64)):
                                ts = ts_str.astype(datetime.datetime)
                            else:
                                clean_ts = clean_ts.replace('Z', '')
                                ts = datetime.datetime.fromisoformat(clean_ts)

                            # --- 2. Get Data ---
                            if 'bands' in z:
                                # Ensure float for stats
                                data = z['bands'][:].astype(np.float32)
                                # print(data.shape)

                                # --- 3. Split Logic ---
                                if ts < CUTOFF_DATE:
                                    stats_pre.update(data)
                                else:
                                    stats_post.update(data)

                                total_processed += 1
                                pbar.update(1)
    pbar.close()

    # --- REPORTING ---
    print('\n' + '=' * 60)
    print('COMPUTED STATISTICS')
    print('=' * 60)

    mean_pre, std_pre = stats_pre.compute_final()
    mean_post, std_post = stats_post.compute_final()

    print(
        f'{"Channel":<5} | {"Pre Mean":<10} {"Post Mean":<10} | {"Pre Min":<10} {"Post Min":<10} | {"Pre Std":<10} {"Post Std":<10} | {"Diff (Mean)":<10}'
    )
    print('-' * 90)
    for c in range(12):
        diff = mean_post[c] - mean_pre[c]
        print(
            f'{CHANNEL_NAMES[c]:<5} | {mean_pre[c]:<10.1f} {mean_post[c]:<10.1f} | {stats_pre.min[c]:<10.1f} {stats_post.min[c]:<10.1f} | {std_pre[c]:<10.1f} {std_post[c]:<10.1f} | {diff:<10.1f}'
        )

    # --- PLOTTING ---
    print('\nGenerating Histogram Plot...')

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    # Plot center of bins
    bin_centers = 0.5 * (stats_pre.bins[1:] + stats_pre.bins[:-1])

    for c in range(12):
        ax = axes[c]

        # Normalize histograms to density to account for different sample counts
        sum_pre = stats_pre.histograms[c].sum()
        sum_post = stats_post.histograms[c].sum()

        dens_pre = stats_pre.histograms[c] / (sum_pre + 1e-6)
        dens_post = stats_post.histograms[c] / (sum_post + 1e-6)

        # Plot
        ax.plot(
            bin_centers,
            dens_pre,
            color='blue',
            label='Pre-2022',
            alpha=0.8,
            linewidth=2,
        )
        ax.plot(
            bin_centers,
            dens_post,
            color='orange',
            label='Post-2022',
            alpha=0.8,
            linewidth=2,
        )

        # Add a shifted version of Post to see if they align
        # Heuristic: Shift Post by +1000 to see if it matches Pre
        ax.plot(
            bin_centers + 1000,
            dens_post,
            color='green',
            linestyle=':',
            label='Post + 1000',
            alpha=0.6,
        )

        ax.set_title(f'{CHANNEL_NAMES[c]}')
        ax.set_yscale('log')  # Log scale is crucial for S2 data
        ax.grid(True, alpha=0.3)
        if c == 0:
            ax.legend()

    plt.suptitle(
        f'Sentinel-2 Distribution Shift Analysis\nSample Count: Pre={stats_pre.count}, Post={stats_post.count}',
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig('s2_split_histograms.png')
    print("Saved plot to 's2_split_histograms.png'")


if __name__ == '__main__':
    check_local_data()
