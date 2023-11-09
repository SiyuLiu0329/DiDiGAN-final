"""Loss functions."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
import PIL
import cv2
from torch_utils.manifold import get_lin_ws
import random


def compute_per_channel_dice(
    input_tensor: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
    weight: float = None,
    reduce_mean: bool = True,
) -> torch.Tensor:
    assert (
        input_tensor.size() == target.size()
    ), "'input_tensor' and 'target' must have the same shape"
    dscs = []
    for c in range(target.shape[1]):
        t1, t2 = input_tensor[:, c, :, :], target[:, c, :, :]
        t1 = t1.flatten(1)
        t2 = t2.flatten(1)
        t2 = t2.float()
        intersect = (t1 * t2).sum(-1)
        if weight is not None:
            intersect = weight * intersect

        denominator = (t1 * t1).sum(-1) + (t2 * t2).sum(-1)
        dsc = (2 * (intersect / denominator.clamp(min=epsilon))).mean()
        dscs.append(dsc)
    return torch.mean(torch.stack(dscs)) if reduce_mean else dscs


class Loss:
    def accumulate_gradients(
        self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg
    ):
        raise NotImplementedError()


def save_as_rgb_img(path: str, arr) -> None:
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    arr = PIL.Image.fromarray(arr)
    arr.save(path)


def save_ncwh_tensor_as_png(
    prefix: str, tensor: torch.Tensor, postfix: str = None
) -> None:
    imgs = tensor.permute(0, 2, 3, 1)
    for i, img in enumerate(imgs):
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        if postfix is not None:
            save_as_rgb_img(f"{prefix}_{i}_{postfix}.png", img.data.cpu().numpy())
        else:
            save_as_rgb_img(f"{prefix}_{i}.png", img.data.cpu().numpy())


class DiDiGANLoss(Loss):
    def __init__(
        self,
        device,
        G,
        D,
        augment_pipe=None,
        r1_gamma=10,
        style_mixing_prob=0,
        pl_weight=0,
        pl_batch_shrink=2,
        pl_decay=0.01,
        pl_no_weight_grad=False,
        blur_init_sigma=0,
        blur_fade_kimg=0,
        fake_clz_val=-5,
    ):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.augment_pipe = augment_pipe
        self.r1_gamma = r1_gamma
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_no_weight_grad = pl_no_weight_grad
        self.pl_mean = torch.zeros([], device=device)
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg
        self.fake_clz_val = fake_clz_val

    def run_G(self, z, c, clz, update_emas=False):
        ws = self.G.mapping(z, clz.long(), update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function("style_mixing"):
                cutoff = torch.empty([], dtype=torch.int256, device=ws.device).random_(
                    1, ws.shape[1]
                )
                cutoff = torch.where(
                    torch.rand([], device=ws.device) < self.style_mixing_prob,
                    cutoff,
                    torch.full_like(cutoff, ws.shape[1]),
                )
                ws[:, cutoff:] = self.G.mapping(
                    torch.randn_like(z), clz.long(), update_emas=False
                )[:, cutoff:]

        img = self.G.synthesis(ws, c, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function("blur"):
                f = (
                    torch.arange(-blur_size, blur_size + 1, device=img.device)
                    .div(blur_sigma)
                    .square()
                    .neg()
                    .exp2()
                )
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits, seg, p_clz = self.D(img, update_emas=update_emas)
        return logits, seg, p_clz

    def accumulate_gradients(
        self, phase, real_img, real_c, gen_z, gen_c, clz, gain, cur_nimg
    ):
        assert phase in ["Gmain", "Greg", "Gboth", "Dmain", "Dreg", "Dboth"]
        if self.pl_weight == 0:
            phase = {"Greg": "none", "Gboth": "Gmain"}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {"Dreg": "none", "Dboth": "Dmain"}.get(phase, phase)
        blur_sigma = (
            max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma
            if self.blur_fade_kimg > 0
            else 0
        )

        if phase in ["Gmain", "Gboth"]:
            with torch.autograd.profiler.record_function("Gmain_forward"):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, clz)
                gen_img2, _ = self.run_G(
                    torch.randn(gen_z.shape, device=gen_z.device), gen_c, clz
                )
                loss_div = -F.l1_loss(gen_img, gen_img2.detach()).mean()

                gen_logits, seg, p_clz = self.run_D(
                    gen_img, gen_c, blur_sigma=blur_sigma
                )
                training_stats.report("Loss/scores/fake", gen_logits)
                training_stats.report("Loss/signs/fake", gen_logits.sign())
                loss_seg_g = 1 * F.mse_loss(
                    seg, F.interpolate(gen_c, (256, 256), mode="bicubic")
                )
                loss_clz_g = F.mse_loss(p_clz, clz.float().unsqueeze(1)).mean()
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report("Loss/G/loss", loss_Gmain)
            with torch.autograd.profiler.record_function("Gmain_backward"):
                (
                    loss_Gmain.mean().mul(gain)
                    + loss_seg_g.mean() * 1.0
                    + loss_div
                    + loss_clz_g
                ).backward()
                print("____G____")
                print(
                    f"loss_g_gdiv{loss_div:>40f}\nloss_g_main{loss_Gmain.mean():>40f}\nloss_g_segf{loss_seg_g.mean():>40f}\nloss_g_clas{loss_clz_g:>40f}"
                )
        count = 0
        loss_inter = 0
        if phase in ["Greg", "Gboth"]:
            with torch.autograd.profiler.record_function("Gpl_forward"):
                if count == 10:
                    w = get_lin_ws([0, 1], self.G, random.randint(3, 20), 50)
                    w = torch.cat(w, 0)
                    inter_c = gen_c
                    inter_img = self.G.synthesis(w, inter_c)
                    inter_logits, inter_seg, _ = self.run_D(
                        inter_img, inter_c, blur_sigma=blur_sigma
                    )
                    loss_Gmain_inter = torch.nn.functional.softplus(
                        -inter_logits
                    ).mean()
                    loss_seg_g_inter = (
                        1
                        * F.mse_loss(
                            inter_seg,
                            F.interpolate(inter_c, (256, 256), mode="bicubic"),
                        ).mean()
                    )
                    loss_inter = loss_Gmain_inter + loss_seg_g_inter
                    count = 0
                count += 1

                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(
                    gen_z[:batch_size], gen_c[:batch_size], clz[:batch_size]
                )
                pl_noise = torch.randn_like(gen_img) / np.sqrt(
                    gen_img.shape[2] * gen_img.shape[3]
                )
                with torch.autograd.profiler.record_function(
                    "pl_grads"
                ), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(
                        outputs=[(gen_img * pl_noise).sum()],
                        inputs=[gen_ws],
                        create_graph=True,
                        only_inputs=True,
                    )[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report("Loss/pl_penalty", pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report("Loss/G/reg", loss_Gpl)
            with torch.autograd.profiler.record_function("Gpl_backward"):
                (loss_Gpl.mean().mul(gain) + loss_inter).backward()

        loss_Dgen = 0
        if phase in ["Dmain", "Dboth"]:
            with torch.autograd.profiler.record_function("Dgen_forward"):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, clz, update_emas=True)
                gen_logits, seg, p_clz = self.run_D(
                    gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True
                )
                training_stats.report("Loss/scores/fake", gen_logits)
                training_stats.report("Loss/signs/fake", gen_logits.sign())
                loss_seg_d_fake = torch.zeros_like(
                    seg
                )  # 1 * F.mse_loss(seg, torch.ones_like(seg) * self.fake_clz_val)
                loss_clz_d_fake = 0

                # F.mse_loss(
                #     p_clz, (torch.ones_like(clz) * 2).long()
                # ).mean()
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function("Dgen_backward"):
                (
                    loss_Dgen.mean().mul(gain)
                    + loss_seg_d_fake.mean() * 1.0
                    + loss_clz_d_fake
                ).backward()
                print("____D(F)____")
                print(
                    f"loss_d_genf{loss_Dgen.mean():>40f}\nloss_d_segf{loss_seg_d_fake.mean():>40f}\nloss_d_clzf{loss_clz_d_fake:>40f}"
                )

        if phase in ["Dmain", "Dreg", "Dboth"]:
            name = (
                "Dreal"
                if phase == "Dmain"
                else "Dr1"
                if phase == "Dreg"
                else "Dreal_Dr1"
            )
            with torch.autograd.profiler.record_function(name + "_forward"):
                real_img_tmp = real_img.detach().requires_grad_(
                    phase in ["Dreg", "Dboth"]
                )
                real_logits, seg, p_clz = self.run_D(
                    real_img_tmp, real_c, blur_sigma=blur_sigma
                )
                training_stats.report("Loss/scores/real", real_logits)
                training_stats.report("Loss/signs/real", real_logits.sign())

                loss_Dreal = 0
                loss_seg_d_real = 0
                loss_clz_d_real = 0
                if phase in ["Dmain", "Dboth"]:
                    loss_seg_d_real = (
                        1
                        * F.mse_loss(
                            seg, F.interpolate(real_c, (256, 256), mode="nearest")
                        ).mean()
                    )
                    loss_clz_d_real = F.mse_loss(p_clz, clz.float().unsqueeze(1)).mean()
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report("Loss/D/loss", loss_Dgen + loss_Dreal)
                    print("____D(R)____")
                    print(
                        f"loss_d_real{loss_Dreal.mean():>40f}\nloss_d_segr{loss_seg_d_real:>40f}\nloss_d_clzr{loss_clz_d_real:>40f}"
                    )

                loss_Dr1 = 0
                if phase in ["Dreg", "Dboth"]:
                    with torch.autograd.profiler.record_function(
                        "r1_grads"
                    ), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(
                            outputs=[real_logits.sum()],
                            inputs=[real_img_tmp],
                            create_graph=True,
                            only_inputs=True,
                        )[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report("Loss/r1_penalty", r1_penalty)
                    training_stats.report("Loss/D/reg", loss_Dr1)

            with torch.autograd.profiler.record_function(name + "_backward"):
                (
                    (loss_Dreal + loss_Dr1).mean().mul(gain)
                    + loss_seg_d_real * 1.0
                    + loss_clz_d_real
                ).backward()
