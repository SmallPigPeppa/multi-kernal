import argparse
import pytorch_lightning as pl


def parse_args():
    parser = argparse.ArgumentParser()
    # max_epochs, gpus, precision, num_workers
    parser = pl.Trainer.add_argparse_args(parser)

    # dataset
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--image_size", type=int,default=32)


    # optimizer
    SUPPORTED_OPTIMIZERS = ["sgd", "adam"]
    # general train
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--classifier_lr", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--optimizer", choices=SUPPORTED_OPTIMIZERS, type=str, required=True)
    parser.add_argument("--lars", action="store_true")
    parser.add_argument("--grad_clip_lars", action="store_true")
    parser.add_argument("--eta_lars", default=1e-3, type=float)
    parser.add_argument("--exclude_bias_n_norm", action="store_true")

    # scheduler
    SUPPORTED_SCHEDULERS = [
        "reduce",
        "cosine",
        "warmup_cosine",
        "step",
        "exponential",
        "none",
    ]
    parser.add_argument("--scheduler", choices=SUPPORTED_SCHEDULERS, type=str, default="reduce")
    parser.add_argument("--lr_decay_steps", default=None, type=int, nargs="+")
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--warmup_start_lr", default=0.003, type=float)
    parser.add_argument("--warmup_epochs", default=10, type=int)

    # wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--name", type=str)
    parser.add_argument("--project", type=str)
    parser.add_argument("--entity", type=str)

    # fix encoder
    parser.add_argument("--fix_encoder", action="store_true")

    args = parser.parse_args()
    return args

if __name__=='__main__':
    args=parse_args()
    print(args)