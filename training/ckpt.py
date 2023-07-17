import torch.nn as nn
import os
import numpy as np
import PIL
import torch
from torch_utils.manifold import linviz_n_class
import matplotlib.pyplot as plt

class Ckpt(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, 
        pred_dir, 
        phase_real_c,
        n_classes,
        G_ema,
        device
    ):
        if not os.path.exists(f'{pred_dir}'):
            os.makedirs(f'{pred_dir}')
        grid_c = [phase_real_c[0][i:i+1] for i in range(3)]
        
        linviz_n_class(range(n_classes), pred_dir, G_ema, phase_real_c[0][0:1], n_neighbours=20, pts=24, w_dim=G_ema.w_dim)
        plt.clf()
        for a, c in enumerate(grid_c):
            for b in range(1):
                for clz in range(n_classes):
                    z = torch.randn([1, G_ema.z_dim], device=device)
                    img = G_ema(z=z, c=c, clz=(torch.ones(1) * clz).long().to(z.device), 
                        noise_mode='const').cpu()[0][0].numpy()
                    img = np.asarray(img, dtype=np.float32)
                    lo, hi = [-1, 1]
                    img = (img - lo) * (255 / (hi - lo))
                    img = np.rint(img).clip(0, 255).astype(np.uint8)
                    PIL.Image.fromarray(img).save(
                        f'{pred_dir}/c_in={a}_clz={clz}.png')
                    ci = (c - lo) * (255 / (hi - lo))
                    PIL.Image.fromarray((ci[0, 0, :, :].cpu().numpy()).astype(
                        'uint8')).save(f'{pred_dir}/c_in={a}_src.png')

    def save(self, ckpt_dir, G, G_ema, D):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        for name, m in zip(['G', 'D', 'G_ema'], [G, D, G_ema]):
            torch.save(m.state_dict(), os.path.join(
                ckpt_dir, f'{name}.pth'))