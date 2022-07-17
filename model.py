from pl_bolts.models.self_supervised import SimCLR
from functools import partial
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import nn
import torch
from torch.nn import functional as F
from torchmetrics.functional import accuracy
from typing import Any, Callable, Dict, List, Sequence, Tuple


def static_lr(
        get_lr: Callable, param_group_indexes: Sequence[int], lrs_to_replace: Sequence[float]
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class MultiKernal(LightningModule):
    def __init__(self,
                 optimizer="sgd",
                 lr=0.4,
                 batch_size=256,
                 weight_decay=1e-5,
                 extra_optimizer_args={},
                 scheduler="warmup_cosine",
                 max_epochs=200,
                 warmup_epochs=10,
                 warmup_start_lr=0.003,
                 min_lr=0.0,
                 classifier_lr=0.1, fix_encoder=True, **kwargs):

        super().__init__()
        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
        simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
        self.encoder = simclr.encoder

        # change kernal
        # self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        # self.encoder.maxpool = nn.Identity()
        # self.encoder.to(device)

        # mlp
        dim_in = 2048
        dim_out = 100
        self.classifier = nn.Linear(dim_in, dim_out)

        # params
        self.optimizer = optimizer
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.extra_optimizer_args = extra_optimizer_args
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        self.classifier_lr = classifier_lr
        self.fix_encoder = fix_encoder

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """
        # conv1 weight encoder.conv1.weight
        # classifier weight
        all_params = tuple(self.encoder.parameters())
        updated_params = list()
        fixed_wd_params = list()
        for name, param in self.encoder.named_parameters():
            if name == "encoder.conv1.weight":
                updated_params.append(param)
            else:
                fixed_wd_params.append(param)
        print(len(updated_params), len(fixed_wd_params), len(all_params))
        assert len(updated_params) + len(fixed_wd_params) == len(all_params), "Sanity check failed."

        return [
            {"name": "encoder_fixed", "params": fixed_wd_params, "weight_decay": 0., "lr": 0.},
            {"name": "encoder_updated", "params": updated_params, "weight_decay": self.weight_decay, "lr": self.lr},
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": 0,
            },
        ]

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        # collect learnable parameters
        idxs_no_scheduler = [
            i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)
        ]

        # select optimizer
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam)")

        # create optimizer
        # optimizer = optimizer(
        #     self.learnable_params,
        #     lr=self.lr,
        #     weight_decay=self.weight_decay,
        #     **self.extra_optimizer_args,
        # )
        optimizer = optimizer(
            self.learnable_params,
            **self.extra_optimizer_args,
        )

        if self.scheduler == "none":
            return optimizer
        else:
            if self.scheduler == "warmup_cosine":
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=self.warmup_epochs,
                    max_epochs=self.max_epochs,
                    warmup_start_lr=self.warmup_start_lr,
                    eta_min=self.min_lr,
                )
            elif self.scheduler == "cosine":
                scheduler = CosineAnnealingLR(optimizer, self.max_epochs, eta_min=self.min_lr)
            elif self.scheduler == "step":
                scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
            else:
                raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

            if idxs_no_scheduler:
                partial_fn = partial(
                    static_lr,
                    get_lr=scheduler.get_lr,
                    param_group_indexes=idxs_no_scheduler,
                    lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
                )
                scheduler.get_lr = partial_fn

            return [optimizer], [scheduler]

    def forward(self, x):
        if self.fix_encoder:
            with torch.no_grad():
                z = self.encoder(x)
        else:
            z = self.encoder(x)
        y = self.classifier(z[-1])
        return F.log_softmax(y, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        if stage:
            self.log(f"{stage}_loss", loss, on_epoch=True, sync_dist=True)
            self.log(f"{stage}_acc", acc, on_epoch=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")


if __name__ == '__main__':
    from args import parse_args

    args = parse_args()
    m = MultiKernal(args)
    # print(m.encoder)
    # for name, param in m.named_parameters():
    #     print(name)
    a = torch.rand([2, 3, 32, 32])
    b = m(a)
