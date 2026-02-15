import json

import torch
from tqdm import tqdm

from eo_vae.datasets.terramesh_datamodule import TerraMeshDataModule


class RunningStatsButFast(torch.nn.Module):
    def __init__(self, shape, dims):
        """Initializes the RunningStatsButFast method.

        A PyTorch module that can be put on the GPU and calculate the multidimensional
        mean and variance of inputs online in a numerically stable way.

        Args:
            shape: The shape of resulting mean and variance (e.g., number of channels).
            dims: The dimensions of your input to calculate the mean and variance over.
        """
        super(RunningStatsButFast, self).__init__()
        self.register_buffer('mean', torch.zeros(shape))
        self.register_buffer('var', torch.ones(shape))
        self.register_buffer('std', torch.ones(shape))
        self.register_buffer('min', torch.ones(shape) * float('inf'))
        self.register_buffer('max', torch.ones(shape) * float('-inf'))
        self.register_buffer('count', torch.zeros(1))
        self.dims = dims

    def update(self, x):
        with torch.no_grad():
            # Calculate batch stats over the specified dimensions
            batch_mean = torch.mean(x, dim=self.dims)
            batch_var = torch.var(x, dim=self.dims, unbiased=False)

            # Calculate min/max over the reduction dimensions
            # We flatten the reduction dimensions to compute min/max efficiently
            # x shape: [B, C, H, W] -> permute to [C, B, H, W] -> flatten to [C, N]
            x_flat = x.permute(1, 0, 2, 3).flatten(1)

            batch_min = x_flat.min(dim=1)[0]
            batch_max = x_flat.max(dim=1)[0]

            # Total number of elements aggregated per channel in this batch
            # (B * H * W)
            batch_count = torch.tensor(
                x_flat.shape[1], device=x.device, dtype=torch.float
            )

            # Update Min/Max
            self.min = torch.minimum(self.min, batch_min)
            self.max = torch.maximum(self.max, batch_max)

            # Parallel algorithm for Mean/Var updates
            # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            delta = batch_mean - self.mean
            tot_count = self.count + batch_count

            new_mean = self.mean + delta * batch_count / tot_count
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
            new_var = M2 / tot_count

            self.mean = new_mean
            self.var = new_var
            self.count = tot_count
            self.std = torch.sqrt(self.var + 1e-8)

    def forward(self, x):
        self.update(x)
        return x


def compute_stats_online(data_path, modalities, num_batches=100):
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for mod in modalities:
        print(f'\n--- Processing modality: {mod} ---')

        dm = TerraMeshDataModule(
            data_path=data_path,
            modalities=[mod],
            num_workers=4,
            batch_size=4,  # Larger batch size is fine now
            normalize=False,
            target_size=None,
        )
        dm.setup()
        loader = dm.train_dataloader()

        runner = None

        with torch.no_grad():
            for i, batch in tqdm(enumerate(loader), total=num_batches):
                if i >= num_batches:
                    break

                img = batch['image'].to(device)  # [B, C, H, W]
                if img.dim() == 3:
                    img = img.unsqueeze(0)

                # Initialize runner if needed
                if runner is None:
                    C = img.shape[1]
                    # We want stats per channel. Input is (B, C, H, W).
                    # We reduce over B(0), H(2), W(3).
                    runner = RunningStatsButFast(shape=(C,), dims=[0, 2, 3]).to(device)

                # Convert to float64 for numerical stability during accumulation
                img = img.double()
                runner.update(img)

        print(f'Stats for {mod}:')
        # Convert to python list for JSON serialization
        results[mod] = {
            'mean': runner.mean.cpu().tolist(),
            'std': runner.std.cpu().tolist(),
            'min': runner.min.cpu().tolist(),
            'max': runner.max.cpu().tolist(),
        }

    return results


if __name__ == '__main__':
    # Configuration
    DATA_PATH = '/mnt/SSD2/nils/datasets/terramesh'
    MODALITIES_TO_CHECK = ['S2L2A', 'S1RTC', 'S2L1C']

    stats = compute_stats_online(
        data_path=DATA_PATH,
        modalities=MODALITIES_TO_CHECK,
        num_batches=500,  # 50 batches is plenty with subsampling
        # pixels_per_batch=10000 # 50k pixels per batch per channel
    )

    # Print in a format you can copy-paste into NORM_STATS
    import json

    print('\n--- Computed Stats ---')
    print(json.dumps(stats, indent=4))
