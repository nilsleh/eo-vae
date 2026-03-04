import torch
import math
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import AveragePrecision


class LinearProbeModule(LightningModule):
    """Single linear layer trained on frozen VAE latent features.

    Args:
        feat_dim: Input feature dimensionality (32 for global-avg-pooled latents).
        num_classes: Number of output classes.
        base_lr: Peak learning rate.
        final_lr: Minimum LR at end of cosine schedule.
        warmup_epochs: Epochs for linear LR warmup.
        max_epochs: Total training epochs (for cosine schedule end).
        weight_decay: AdamW weight decay.
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        base_lr: float = 1e-3,
        final_lr: float = 1e-5,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.SiLU(),
            nn.Linear(feat_dim // 2, num_classes),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.val_map = AveragePrecision(task='multilabel', num_labels=num_classes, average='macro')
        self.test_map = AveragePrecision(task='multilabel', num_labels=num_classes, average='macro')

    def forward(self, feature):
        return self.linear(feature)

    def training_step(self, batch, batch_idx):
        logits = self(batch['feature'])
        loss = self.loss_fn(logits, batch['label'])
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch['feature'])
        loss = self.loss_fn(logits, batch['label'])
        preds = torch.sigmoid(logits)
        self.val_map.update(preds, batch['label'].int())
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log('val_mAP', self.val_map.compute(), prog_bar=True)
        self.val_map.reset()

    def test_step(self, batch, batch_idx):
        logits = self(batch['feature'])
        preds = torch.sigmoid(logits)
        self.test_map.update(preds, batch['label'].int())

    def on_test_epoch_end(self):
        self.log('test_mAP', self.test_map.compute())
        self.test_map.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.base_lr,
            weight_decay=self.hparams.weight_decay,
        )

        # def lr_lambda(epoch):
        #     warmup = self.hparams.warmup_epochs
        #     total = self.hparams.max_epochs
        #     ratio = self.hparams.final_lr / self.hparams.base_lr

        #     if epoch < warmup:
        #         return (epoch + 1) / warmup
        #     # Cosine decay from base_lr to final_lr
        #     progress = (epoch - warmup) / max(total - warmup, 1)
        #     cosine = 0.5 * (1 + math.cos(math.pi * progress))
        #     return ratio + (1 - ratio) * cosine

        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }