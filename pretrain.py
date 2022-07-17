import torchvision
from torchvision import transforms
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.nn import functional as F
from torch import nn
import torch
from torchmetrics.functional import accuracy
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from args import parse_args
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from model import MultiKernal



class MLP(LightningModule):
    def __init__(self, dim_in=2048,dim_out=100):
        super().__init__()
        self.dim_in=dim_in
        self.dim_out=dim_out
        self.model = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(
        #     self.parameters(),
        #     lr=LR,
        #     momentum=0.9,
        #
        # )
        optimizer=Adam(self.parameters(), lr=1e-3)
        # weight_decay = 5e-4,
        return {"optimizer": optimizer}



if __name__=='__main__':
    args=parse_args()
    # cifar100
    # mean = [0.5071, 0.4867, 0.4408]
    # std = [0.2675, 0.2565, 0.2761]
    # imagenet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_transform = transforms.Compose([transforms.Resize(args.image_size), transforms.ToTensor(),transforms.Normalize(mean, std)])
    # transforms.CenterCrop(size=96)
    if args.dataset=='cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, transform=img_transform,download=True)
        test_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False,transform=img_transform,download=True)

        # train_dataset = torchvision.datasets.CIFAR100(root='home/admin/torch_ds/cifar100', train=True, transform=img_transform,
        #                                               download=True)
        # test_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, transform=img_transform,
        #                                              download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,pin_memory=True)
    model = MultiKernal(**args.__dict__)

    callbacks = []
    if args.wandb:
        wandb_logger = WandbLogger(
            name=f"{args.name}",
            project=args.project,
            entity=args.entity,
            offline=False,
            reinit=True,
        )
        wandb_logger.watch(model, log="gradients", log_freq=100)
        wandb_logger.log_hyperparams(args)
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    trainer = Trainer.from_argparse_args(
        args,
        gpus=[7],
        logger=wandb_logger if args.wandb else None,
        callbacks=callbacks,
        checkpoint_callback=False,
        terminate_on_nan=True,
    )
    trainer.fit(model,train_loader,test_loader)
