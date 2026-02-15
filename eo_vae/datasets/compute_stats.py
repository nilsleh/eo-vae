import datetime
import glob
import os
import tarfile
import tempfile

import numpy as np
import zarr
from tqdm import tqdm

# --- CONFIGURATION ---
LOCAL_DATA_ROOT = '/mnt/SSD2/nils/datasets/terramesh'
CUTOFF_DATE = datetime.datetime(2022, 1, 25)
CLIP_MAX = 10000.0
SAMPLES_TO_CHECK = 10000

# Standard S2 L2A Bands
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
    'train': os.path.join(LOCAL_DATA_ROOT, 'train', 'S2L2A', 'majortom_shard_*.tar'),
    'val': os.path.join(LOCAL_DATA_ROOT, 'val', 'S2L2A', 'majortom_shard_*.tar'),
}


class RunningStats:
    def __init__(self, n_channels=12):
        self.n_channels = n_channels
        self.count = 0
        self.pixel_count = 0
        self.sum = np.zeros(n_channels, dtype=np.float64)
        self.sq_sum = np.zeros(n_channels, dtype=np.float64)

    def update(self, img):
        B, C, H, W = img.shape
        pixels = H * W
        self.count += 1
        self.pixel_count += pixels

        flat = img.reshape(C, -1)
        self.sum += flat.sum(axis=1)
        self.sq_sum += (flat**2).sum(axis=1)

    def compute_final(self):
        if self.pixel_count == 0:
            raise RuntimeError('No pixels processed! Cannot compute stats.')
        mean = self.sum / self.pixel_count
        var = (self.sq_sum / self.pixel_count) - (mean**2)
        std = np.sqrt(np.maximum(var, 0))
        return mean, std


def compute_stats():
    stats = RunningStats()

    # Collect files
    all_files = []
    for split, pattern in SEARCH_PATTERNS.items():
        files = sorted(glob.glob(pattern))
        for f in files:
            # Filter for shards 00-25
            shard_id = f.split('_')[-1].split('.')[0]
            if split == 'train' and '000001' <= shard_id <= '000025':
                all_files.append(f)
            elif split == 'val' and '000001' <= shard_id <= '000005':
                all_files.append(f)

    if not all_files:
        raise FileNotFoundError('No files found matching patterns.')

    print(f'Scanning {len(all_files)} shards for Unified Stats...')
    pbar = tqdm(total=SAMPLES_TO_CHECK)
    total_processed = 0

    for tar_path in all_files:
        if total_processed >= SAMPLES_TO_CHECK:
            break

        # Open Tar (No try/except)
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar:
                if total_processed >= SAMPLES_TO_CHECK:
                    break

                if member.name.endswith('.zarr.zip'):
                    f = tar.extractfile(member)
                    if f is None:
                        raise ValueError(f'Could not extract member: {member.name}')
                    content = f.read()

                    # Write to temp file for Zarr
                    with tempfile.NamedTemporaryFile(suffix='.zip') as tmp:
                        tmp.write(content)
                        tmp.flush()

                        # Open Zarr (No try/except)
                        with zarr.open(
                            store=zarr.ZipStore(tmp.name, mode='r'), mode='r'
                        ) as z:
                            # --- 1. STRICT DATE PARSING ---
                            ts_str = None
                            if 'time' in z.attrs:
                                ts_str = z.attrs['time']
                            elif 'time' in z:
                                t_val = z['time'][...]
                                # Handle scalar 0-d array vs 1-d array
                                if isinstance(t_val, (np.ndarray, list)):
                                    ts_str = (
                                        str(t_val[0]) if len(t_val) > 0 else str(t_val)
                                    )
                                else:
                                    ts_str = str(t_val)

                            if not ts_str:
                                # Crash if time is missing
                                print(f'Available keys: {list(z.keys())}')
                                print(f'Available attrs: {dict(z.attrs)}')
                                raise KeyError(f"Missing 'time' in {member.name}")

                            clean_ts = str(ts_str).strip(" []'")
                            ts = None

                            # Logic to determine format
                            if clean_ts.isdigit():
                                # Nanoseconds timestamp
                                ts = datetime.datetime.fromtimestamp(
                                    int(clean_ts) / 1e9
                                )
                            elif isinstance(ts_str, (np.datetime64, np.timedelta64)):
                                ts = ts_str.astype(datetime.datetime)
                            else:
                                # ISO Format
                                clean_ts = clean_ts.replace('Z', '')
                                try:
                                    ts = datetime.datetime.fromisoformat(clean_ts)
                                except ValueError:
                                    raise ValueError(
                                        f'Unknown date format: {ts_str} in {member.name}'
                                    )

                            # --- 2. LOAD DATA ---
                            data = None
                            if 'bands' in z:
                                data = z['bands'][:]
                            elif 's2:bands' in z:
                                data = z['s2:bands'][:]

                            if data is None:
                                raise KeyError(f"Missing 'bands' in {member.name}")

                            data = data.astype(np.float32)

                            # --- 3. HARMONIZATION LOGIC ---
                            # A. Shift Post-2022 Data
                            if ts >= CUTOFF_DATE:
                                data = data + 1000.0

                            # B. Clip to Valid Range (0 - 10000)
                            data = np.clip(data, 0.0, CLIP_MAX)

                            # C. Update Stats
                            stats.update(data)
                            total_processed += 1
                            pbar.update(1)

    pbar.close()

    # --- OUTPUT ---
    mean, std = stats.compute_final()

    print('\n' + '=' * 60)
    print('FINAL STATISTICS (Harmonized + Clipped)')
    print('=' * 60)

    # Formatted for copy-paste
    print('MEAN = [')
    print(', '.join([f'{m:.4f}' for m in mean]))
    print(']')

    print('\nSTD = [')
    print(', '.join([f'{s:.4f}' for s in std]))
    print(']')


if __name__ == '__main__':
    compute_stats()
