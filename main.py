import json
import os
import torch
import dnnlib
import tempfile
import sys
import argparse
from training.training_loop import training_loop
from torch_utils import training_stats
from torch_utils import custom_ops


def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(
        file_name=os.path.join(c["run_dir"], "log.txt"),
        file_mode="a",
        should_flush=True,
    )

    if c["num_gpus"] > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, ".torch_distributed_init"))
        init_method = f"env://{init_file}"
        torch.distributed.init_process_group(
            backend="nccl", init_method=init_method, rank=rank, world_size=c["num_gpus"]
        )

    sync_device = torch.device("cuda", rank) if c["num_gpus"] > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = "none"
    json.dump(c, open(f'{c["run_dir"]}/conf.json', "w+"))

    training_loop(rank=rank, **c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True, type=str)
    args = parser.parse_args()
    c = json.load(open(args.conf, "r"))
    if not os.path.exists(c["run_dir"]):
        os.makedirs(c["run_dir"])
        json.dump(c, open(os.path.join(c["run_dir"], "config.json"), "w+"), indent=4)
    with tempfile.TemporaryDirectory() as temp_dir:
        if c["num_gpus"] == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(
                fn=subprocess_fn, args=(c, temp_dir), nprocs=c["num_gpus"]
            )
