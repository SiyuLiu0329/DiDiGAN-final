import numpy as np
import umap
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import shutil
import random
import imageio
from PIL import ImageDraw, ImageFont

def get_center(group):
    x1, x2 = group
    m1, m2 = np.mean(x1), np.mean(x2)
    return m1, m2

def get_closest_point(coord, group):
    x1, x2 = group
    c = np.array(coord)

    dist = 9999999
    index = None
    count = 0
    for g1, g2 in zip(x1, x2):
        g = np.array([g1, g2])
        d = np.linalg.norm(
            g - c 
        )
        if d < dist:
            dist = d
            index = count
        count += 1
    return index

def get_path1d(n1, n2, pts):
    p = np.linspace(n1, n2, pts)
    return p


def viz_groups(
        clusters,
        n_neighbors, 
        overlay_points=None
    ):
    plt.clf()
    ax = plt.axes()
    ax.set_facecolor('black')
    print(f"{n_neighbors=}")
    all_points = []
    segs = []
    count = 0
    for c in clusters:
        c = c.detach().cpu().numpy()
        all_points.append(c)
        start = count
        count = count + len(c)
        segs.append([start, count])
    

    
    
    all_points = np.concatenate(all_points, 0)
    n_overlay_points = 0
    if overlay_points is not None:
        n_overlay_points = overlay_points.shape[0]
        all_points = np.concatenate([all_points, overlay_points], 0)

    embedding = umap.UMAP(n_neighbors, n_components=2).fit_transform(all_points)
    groupall = (embedding[:, 0], embedding[:, 1])
    group_overlap = (embedding[-n_overlay_points:, 0], embedding[-n_overlay_points:, 1])
    groups = []
    centres =[]

    for s, e in segs:
        g = (embedding[s:e, 0], embedding[s:e, 1])
        groups.append(g)
        plt.scatter(*g)
        c = get_center(g)
        c = get_closest_point(c, groupall)
        centres.append(
            c
        )
        plt.scatter(groupall[0][c], groupall[1][c])
    if overlay_points is not None:
        plt.scatter(*group_overlap)
    
    plt.axis('off')
    return tuple(centres)

@torch.no_grad()
def get_lin_ws(
    classes,
    G, n_neighbours, pts, k=16, w_dim=512
):
    w_all = []
    w_combined = []
    for clz in classes:
        w = []
        for i in tqdm(range(2000)):
            z = torch.normal(0, 1, size=(1, w_dim)).cuda()
            ws = G.mapping(z, (torch.ones(1) * clz).long().to(z.device))
            w.append(ws[:, 0, :])
            w_combined.append(ws)
        w = torch.cat(w, 0)
        w_all.append(w)
    centres = viz_groups(w_all, n_neighbours)
    ws = [w_combined[i] for i in centres]
    frames = []
    text = f'{classes[0]}'
    
    w_inter = []
    for w1, w2, clz in zip(ws, ws[1:], classes[1:]):
        text += f'->{clz}'
        imgs = []
        for n, i in enumerate( np.arange(0, 1, float(1./pts))):
            w = torch.lerp(w1.cuda(), w1.cuda(), i)
            w_inter.append(w)
    return random.choices(w_inter, k=k)

def sample_conditional(G, classes, w_dim, is_discrete):
    w_all = []
    w_combined = []
    for clz in classes:
        w = []
        for i in tqdm(range(10000)):
            z = torch.normal(0, 1, size=(1, w_dim)).cuda()
            if not is_discrete:
                cc = (torch.ones(1).float() * clz).to(z.device)
            else:
                cc = (torch.ones(1) * clz).long().to(z.device)
            ws = G.mapping(z, cc)
            w.append(ws[:, 0, :])
            
            w_combined.append(ws)
        w = torch.cat(w, 0)
        w_all.append(w)
    return w_all, w_combined

def sample_unconditional(G, D, c, w_dim, is_discrete):
    w_all = []
    w_combined0 = []
    w_combined1 = []
    w0, w1 = [], []
    for i in tqdm(range(2000)):
        z = torch.normal(0, 1, size=(16, w_dim)).cuda()
        img, ws = G.uncond_gen(z, c.cuda())
        _, _, p_clz = D(img)
        clz = p_clz.argmax(-1).cpu().numpy()
        
        for cl, w in zip(clz, ws):
        
            
            if cl == 0:
                w0.append(w[0])
                w_combined0.append(w[None, ...])
            elif cl == 1:
                w1.append(w[0])
                w_combined1.append(w[None, ...])
            else:
                raise ValueError()
    w0 = torch.stack(w0, 0)
    w1 = torch.stack(w1, 0)
    
           
    

    w_all = [w0, w1]
    w_combined = w_combined0 + w_combined1

    return w_all, w_combined

@torch.no_grad()
def linviz_n_class(
    classes,
    out_dir, 
    G, c, n_neighbours, pts, w_dim, is_discrete=False, D=None
):  


    if D is not None:
        # use this to colour
        w_all, w_combined = sample_unconditional(G, D, c, w_dim, is_discrete)
    else:
        w_all, w_combined = sample_conditional(G, classes, w_dim, is_discrete)

    centres = viz_groups(w_all, n_neighbours)
    ws = [w_combined[i] for i in centres]
    frames = []
    text = f'{"CN" if classes[0] == 0 else "AD"}'
    w_inters = []
    for w1, w2, clz in zip(ws, ws[1:], classes[1:]):
        text += f'->{"CN" if clz == 0 else "AD"}'
        imgs, w_inter = animate_transition(G, c, w1, w2, pts, text)
        frames.extend(imgs)
        w_inters.extend(w_inter.detach().cpu().numpy())
    # viz_groups(w_all, n_neighbours, overlay_points=np.concatenate(w_inters, 0))
    viz_groups(w_all, n_neighbours, overlay_points=None)
    imageio.mimsave(f'{out_dir}/interp.gif', frames)
    imageio.imsave(f'{out_dir}/f_start.png', frames[0])
    imageio.imsave(f'{out_dir}/f_end.png', frames[-1])
    plt.savefig(f'{out_dir}/manifold.png')
    return [w.detach().cpu().numpy() for w in w_all], w_inters

def animate_transition(G, c, start, end, pts, text):
    imgs = []
    w_inter = []
    for n, i in enumerate( np.arange(0, 1, float(1./pts))):
        
        w = torch.lerp(start.cuda(), end.cuda(), i)
        w_inter.append(w)
        c = c.cuda()
        cn = c + torch.nn.functional.interpolate(torch.normal(0, 0.05, size=(c.shape[0], c.shape[1], 20, 20)), (64, 64)).cuda()
        cn[c==-1] = -1
        c = cn
        img = G.synthesis(w, c[0: 1]).detach().cpu()[0][0].numpy()
        img = np.asarray(img, dtype=np.float32)
        lo, hi = [-1, 1]
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)
        im = Image.fromarray(img).rotate(90)
        dr = ImageDraw.Draw(im)
        fnt = ImageFont.load_default()
        dr.text((10, 10), text, font=fnt, fill=255)

        imgs.append(np.array(im))
    return imgs, torch.cat(w_inter, 0)
    
