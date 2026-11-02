import os
import warnings
import argparse
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.distributed as dist
import json
from pathlib import Path
from typing import Optional
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ChainedScheduler,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from toolbox.toolbox import (
    train1,
    test,
    plot_history,
    disable_user_warnings,
    DataSet,
    EarlyStopping,
    get_time_str,
    DelayedReduceLROnPlateau,
    is_main_process,
)

warnings.filterwarnings("ignore", category=FutureWarning)


class Discriminator(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.2):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)

        self.conv0 = nn.Conv2d(1, 32, 2, padding="valid")
        # self.conv1 = nn.Conv2d(32, 64, 2, padding="valid")
        # self.conv2 = nn.Conv2d(64, 128, 2, padding="valid")
        # self.conv3 = nn.Conv2d(128, 256, 2, padding="valid")
        # self.conv4 = nn.Conv2d(256, 512, 2, padding="valid")

        self.flatten = nn.Flatten()
        self.fc4 = nn.Linear(514080, 1024)
        self.fc5 = nn.Linear(1024, 512)
        # self.fc6 = nn.Linear(128, 64)
        # self.fc7 = nn.Linear(64, 32)
        self.fc66 = nn.Linear(512, self.num_classes)
        # self.fc77 = nn.Linear(2000, self.num_classes)

        # lets a tiny, non-zero signal pass when input is negative
        # use with very deep networks, GANs, high learning rates
        # do not use if you actually need 0-value outputs, limited resources, or when relu works
        self.relu = nn.LeakyReLU()

        # good for very deep neural networks, perfectly smooth gradient, differentiable at zero
        # use with unstable training or with gradient spikes
        # do not use on mobile or edge hardware, or with transformers or LLM, or if batchnorm works
        # self.celu = nn.CELU()

        # allows negative inputs
        # good for computer vision, squeezing out more accuracy
        # don't use with small, limited datasets, more computationally taxing, don't use with weight decay
        self.prelu = nn.PReLU()

        # incorporates normalization internally, good with tabular data and autoencoders
        # works well with deep fully-connected networks with many dense layers
        # do not use with CNN or transformers, or with dropout, or with transfer learning
        # self.selu = nn.SELU()

        # good for very deep neural networks such as LLMs, deep computer vision with CNN, etc.
        # do not use with shallow or small models, transformers, or very edge/low-latency devices
        # self.silu = nn.SiLU()

    def forward(self, x):
        x = self.prelu(self.conv0(x))
        x = self.flatten(x)
        x = self.prelu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.dropout(x)
        y = self.fc66(x)
        return y


def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank
    )


def cleanup():
    dist.destroy_process_group()


def train_model(
        rank: int,
        world_size: int,
        save_model: bool = False,
        test_model: bool = True,
        start_method: Optional[str] = None,
        num_epochs: int = -1,
):
    setup(rank, world_size)
    disable_user_warnings(globally=True)

    path = "dataset/"
    # path = "test"
    features = torch.load(path + "features_us.pt")
    labels = torch.load(path + "labels_us.pt")
    dataset = DataSet(features, labels)

    plot = True
    print_classification_metrics = True

    num_classes = max(labels) + 1
    dropout = 0.2
    train_prop = 0.8
    val_prop = 0.85

    batch_size = int((len(dataset) * train_prop) // 100)
    val_batch_size = int(((len(dataset) * (1 - train_prop)) * val_prop) // 100)
    # test_batch_size = int((len(dataset) * (1 - train_prop)) * (1 - val_prop))
    # batch_size = 512
    # val_batch_size = 256
    # test_batch_size = int(((1 - len(dataset) * train_prop) * (1 - val_prop)) // 1)
    dist_sampler = True

    # Prevent all workers from printing at once
    if is_main_process():
        print(f"Total samples indexed: {len(dataset)}")

    train_dataloader, val_dataloader, test_dataloader = dataset.get_dataloader(
        train_prop=train_prop,
        val_prop=val_prop,
        train_batch_size=batch_size,
        val_batch_size= val_batch_size,
        # test_batch_size=test_batch_size,
        shuffle=True,
        drop_last=True,
        # dual_dataset=True,
        single_dataset=False,
        dist_sampler=dist_sampler,
        sampler=dist_sampler,
        world_size=world_size,
        rank=rank,
        fix_batch_size=True,
        num_workers=4*world_size,
        pin_memory=True,
        multiprocessing_context=start_method,
    )

    device = torch.device(f"cuda:{rank}")
    use_amp = True
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_precision = torch.bfloat16  # or torch.float16 but bf16 is better
    scaler = False if amp_precision == torch.bfloat16 else scaler

    # Initialize model directly on the assigned GPU rank and wrap in DDP
    discriminator = Discriminator(num_classes=num_classes, dropout=dropout).to(rank)
    discriminator = DDP(discriminator, device_ids=[rank])

    beta1 = 0.9
    beta2 = 0.999
    lr = 1e-3  # 3
    num_epochs = 1100 if num_epochs < 0 else num_epochs  # 500
    # num_epochs = 1

    es = EarlyStopping(patience=10, min_delta=1e-3, threshold=50, verbose=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2), fused=use_amp)

    # scheduler_rlrop = ReduceLROnPlateau(optimizer=optimizer, factor=1e-1, patience=25, cooldown=10, min_lr=1e-6)
    # scheduler_calr = CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs, eta_min=1e-6)
    # scheduler_cawr = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=num_epochs//10, T_mult=1, eta_min=1e-6)
    scheduler_drlop = DelayedReduceLROnPlateau(
        optimizer=optimizer,
        delay_epochs=200,
        factor=1e-1,
        mode="min",
        patience=12,
        cooldown=10,
        min_lr=1e-6,
    )
    # scheduler_chained = ChainedScheduler(schedulers=[scheduler_rlrop, scheduler_cawr], optimizer=optimizer)
    # scheduler = scheduler_calr
    scheduler = None

    sch_name = {
        ReduceLROnPlateau: "Reduce LR On Plateau",
        CosineAnnealingLR: "Cosine Annealing LR",
        CosineAnnealingWarmRestarts: "Cosine Annealing WarmRestarts",
        DelayedReduceLROnPlateau: "Delayed Reduce LR On Plateau",
    }
    name = "BirdID"
    fig_path = "figures/us/"

    training_history = train1(
        model=discriminator,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        rank=rank,
        ddp_flag=True,
        epochs=num_epochs,
        lr=lr,
        opt=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        name=f"Discriminator - {name}",
        use_amp=use_amp,
        amp_precision=amp_precision,
        scaler=scaler,
        is_dist=dist_sampler,
        verbose_train=(is_main_process()),
        verbose=False,
        status_bar=True,
        val_status_bar=True,
        multiclass=True,
        # early_stopping=es,
        monitor_gpu_metrics=0.1,
        byte_conv = "mb",
        monitor_gpu_mp=False,
        print_classification_metrics=print_classification_metrics,
        # temp=True,
        # # temp_op="reduce",  # Automatically uses "reduce" but can also be "broadcast" if "broadcast", 1, or True
        # temp_epoch_threshold=100,
        # temp_metric_threshold=0.99,
    )

    cm_path = f"{fig_path}confusion_matrix/"
    Path(cm_path).mkdir(parents=True, exist_ok=True)
    cm_filename = f"{cm_path}/Discriminator - {name}_{get_time_str()}.png"

    if test_model:
        results = test(
            model=discriminator,
            test_dataloader=test_dataloader,
            device=device,
            rank=rank,
            multiclass=True,
            verbose=True,
            ddp_flag=True,
            val_status_bar=True,
            last_epoch=True,
            print_classification_metrics=print_classification_metrics,
            cm_save=True,
            cm_heatmap=True,
            cm_path=cm_filename,
            cm_title="Bird ID Confusion Matrix",
            eval_mode=True,
            cm_force_numpy=True,
            cm_arg_max=True,
            cm_include_axes_labels=False,
        )

        if is_main_process():
            preds, cm = results
        else:
            preds = results[0]


    if is_main_process() and isinstance(training_history, tuple):
        history, gpu_avgs = training_history
        print("\nAverage GPU Metrics During Training:")
        print(json.dumps(gpu_avgs, indent=4))
    else:
        history = training_history
    if is_main_process():
        # Ensure only the main process attempts to create directories and save plots
        if plot:
            disc_path = f"{fig_path}discriminator/"
            Path(disc_path).mkdir(parents=True, exist_ok=True)
            include_lr = True
            include_epochs = True
            if include_lr and include_epochs:
                name = f"{name}_{str(lr).replace(".", "%")}_{num_epochs}"
            elif include_lr:
                name = f"{name}_{str(lr).replace(".", "%")}"
            elif include_epochs:
                name = f"{name}_{num_epochs}"
            if scheduler is not None:
                plot_name = f"{sch_name[type(scheduler)]} - {name}"
            else:
                plot_name = f"Discriminator - {name}"
            plot_history(
                history=history,
                fig_path=disc_path,
                name=plot_name,
                # separate=True,
                twin_axes=True,
                legend_loc="upper center",
                fancybox=True,
                shadow=True,
                include_metrics=True,
                figtext_color=True,
            )


        if save_model:
            output_dir = "models"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_name = f"{output_dir}/{name}_{get_time_str()}"
            torch.save(discriminator.module.state_dict(), f"{output_name}.pt")
            if print_classification_metrics:
                with open(f"models/results_{output_name.replace("models/", "")}.txt", "w") as f:
                    f.write(f"train_acc:\t{history["train"]["accuracy"][-1]:.4f}\n")
                    f.write(f"val_acc:\t{history["test"]["accuracy"][-1]:.4f}\n")
                    f.write(f"train_loss:\t{history["train"]["loss"][-1]:.4f}\n")
                    f.write(f"val_loss:\t{history["test"]["loss"][-1]:.4f}\n")
                    f.write(f"epochs:\t\t{num_epochs}\n")
                    f.write(f"lr:\t\t{lr}\n")
                    f.write(f"act. fns:\tprelu, prelu, leaky_relu\n")

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple args to adjust training configuration")
    # Optional argument with a default value
    parser.add_argument("-e", "--epochs", type=int, default=-1, help="The number of epochs (default: -1)")

    # Optional boolean flag (stores True if passed, False otherwise)
    # parser.add_argument("-m", "--multiclass", action="store_true", help="Multiclass flag, otherwise Binary")
    parser.add_argument("-s", "--save", action="store_true", help="Save model flag")
    parser.add_argument("-t", "--test", action="store_true", help="Test model flag")

    # 3. Parse the arguments
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)

    world_size = torch.cuda.device_count()
    save_flag = (False or args.save)
    test_flag = (False or args.test)
    num_epochs = args.epochs
    start_method = "spawn"
    try:
        if world_size < 2:
            print("DDP requires at least 2 GPUs.")
        else:
            if start_method == "fork" or start_method == "forkserver":
                mp.set_start_method(start_method)
                if start_method == "forkserver":
                    os.environ["TORCH_MP_PARALLEL_START"] = "1"
                mp.start_processes(
                    train_model,
                    args=(world_size, save_flag, test_flag, start_method, num_epochs,),
                    nprocs=world_size,
                    start_method=start_method,
                )
            else:
                mp.spawn(
                    train_model,
                    args=(world_size, save_flag, test_flag, None, num_epochs,),
                    nprocs=world_size,
                )

    except KeyboardInterrupt:
        print("Training is interrupted. Killing child processes...")
        cleanup()
