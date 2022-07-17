python3 pretrain.py \
    --dataset cifar100 \
    --data_dir ~/torch_ds \
    --image_size 224 \
    --max_epochs 200 \
    --gpus [4,5] \
    --precision 16 \
    --optimizer sgd \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.4 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 256 \
    --num_workers 5 \
    --name cifar100 \
    --project multi-resolution \
    --entity pigpeppa \
    --wandb