

"""Main training loop."""

import os
import time
import copy
import json
import psutil
import PIL.Image
import numpy as np
import torch
import random
import dnnlib
import pickle
import torch.nn.functional as F
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix


def training_loop(
    run_dir='.',
    training_set_kwargs={},
    data_loader_kwargs={},
    G_kwargs={},
    D_kwargs={},
    Ckpt_kwargs = {},
    G_opt_kwargs={},
    D_opt_kwargs={},
    augment_kwargs=None,
    loss_kwargs={},
    random_seed=0,
    num_gpus=1,
    rank=0,
    batch_size=4,
    batch_gpu=4,
    ema_kimg=10,
    ema_rampup=0.05,
    G_reg_interval=None,
    D_reg_interval=16,
    augment_p=0,
    do_ada_aug=False,
    ada_target=None,
    ada_interval=4,
    ada_kimg=500,
    total_kimg=25000,
    kimg_per_tick=4,
    image_snapshot_ticks=50,
    resume_kimg=0,
    cudnn_benchmark=True,
    abort_fn=None,
    progress_fn=None,
):

    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True
    grid_sample_gradfix.enabled = True

    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_sampler = misc.InfiniteSampler(
        dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(
        dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print()

    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim,
                         img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    ckpt_step = dnnlib.util.construct_class_by_name(**Ckpt_kwargs)
    G = dnnlib.util.construct_class_by_name(
        **G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    D = dnnlib.util.construct_class_by_name(
        **D_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval()


    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(
            **augment_kwargs).train().requires_grad_(False).to(device)
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')
    if not do_ada_aug:
        print("Disabling ADA augmentation")
        augment_pipe = None
        ada_stats = None

    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(
        device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs)
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(
                params=module.parameters(), **opt_kwargs)
            phases += [dnnlib.EasyDict(name=name+'both',
                                       module=module, opt=opt, interval=1)]
        else:
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(
                module.parameters(), **opt_kwargs)
            phases += [dnnlib.EasyDict(name=name+'main',
                                       module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg',
                                       module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        with torch.autograd.profiler.record_function('data_fetch'):
            if training_set.variable_constraint_res:
                training_set.constraint_res = random.randint(64, 256)
                training_set_iterator = iter(torch.utils.data.DataLoader(
                    dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
            phase_real_img, phase_real_c, phase_cls = next(
                training_set_iterator)
            print(phase_real_c.shape, training_set.constraint_res)

            phase_real_img = (phase_real_img.to(device).to(torch.float32)).split(batch_gpu)
            phase_real_c = (phase_real_c).to(device).split(batch_gpu)
            
            all_gen_z = torch.randn(
                [len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(
                batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]

            phase_cls = phase_cls[:, 0].to(device).long().split(batch_gpu)

        for phase, phase_gen_z in zip(phases, all_gen_z):
            phase_gen_c = phase_real_c
            if batch_idx % phase.interval != 0:
                continue
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))

            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for real_img, real_c, gen_z, gen_c, clz in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c, phase_cls):

                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c,
                                          gen_z=gen_z, gen_c=gen_c, clz=clz, gain=phase.interval, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)

            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters()
                          if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten()
                                     for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5,
                                    neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()

            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        cur_nimg += batch_size
        batch_idx += 1

        if (rank == 0) and batch_idx % 100 == 0:
            print('Predicting...')
            ckpt_step(
                f'{run_dir}/pred/latest', 
                phase_real_c,
                training_set.n_classes,
                G_ema,
                device
            )
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * \
                (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_(
                (augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        tick_end_time = time.time()
        fields = []
        fields += [
            f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [
            f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours',
                               (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days',
                               (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            print('Predicting...')
            ckpt_step(
                f'{run_dir}/pred/t{cur_tick}', 
                phase_real_c,
                training_set.n_classes,
                G_ema,
                device
            )
            ckpt_step.save(f'{run_dir}/ckpt/t={cur_tick}', G, G_ema, D)

        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(
                    name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(
                    f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break
