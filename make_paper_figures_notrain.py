"""
Paper figures that need ONLY the VisDrone images (no training, no trained weights).  Run on Kaggle in the SAME notebook
session as the 60-epoch experiment (or with that notebook's saved output attached as an input).

Produces in /kaggle/working/paper_figs_notrain/:
  fig_dataset_samples.pdf   six training-split images with ground-truth boxes coloured by object size
  fig_dataset_stats.pdf     object-size histogram per split, objects per class, objects per image
  fig_assign_real.pdf       real training images: positive anchors of the task-aligned assigner vs the entropic-OT assigner
  fig_zero_pos.pdf          fraction of objects without any positive anchor, and positives per object, by object size
  figs_notrain_stats.json   every number shown in these figures
  (a .png copy of every figure for quick viewing)
The assigners are compared at the first training iteration: YOLOv8n built for 10 classes and initialised from the
COCO-pretrained weights exactly as at the start of training (yolov8n.pt is downloaded automatically).
Runtime: a few minutes on a Kaggle GPU (about 15 minutes on CPU).
Then download paper_figs_notrain.zip from the Output panel.
"""
import os, sys, json, glob, random, math, shutil, gc
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path('/kaggle/working/paper_figs_notrain') if Path('/kaggle').exists() else Path(os.environ.get('PAPER_FIGS_OUT', 'paper_figs'))
OUT.mkdir(parents=True, exist_ok=True)
INPUT_ROOTS = [Path(p) for p in os.environ.get('PAPER_FIGS_INPUTS', '/kaggle/input:/kaggle/working').split(':')]
IMGSZ, SPLIT_SEED, CONF_VIS = 960, 0, 0.25
NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
BUCKETS = [(0, 8, '<8'), (8, 16, '8–16'), (16, 32, '16–32'), (32, 1e9, '≥32')]

# palette: categorical slots 1-4 of the validated reference palette (light)
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SIZE_COL = [ORANGE, YELLOW, AQUA, BLUE]
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 9, 'axes.titlesize': 9.5, 'legend.fontsize': 8, 'axes.edgecolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.axisbelow': True, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03, 'pdf.fonttype': 42})

def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf'); fig.savefig(OUT / f'{name}.png', dpi=200); plt.close(fig)
    print('saved', name)

# ------------------------------------------------------------------ locate data and weights
def find_dir(name):
    for root in INPUT_ROOTS:
        hits = [p for p in root.rglob(name) if p.is_dir()] if root.exists() else []
        if hits: return hits[0]
    raise FileNotFoundError(f'{name} not found under {INPUT_ROOTS}')

SRC = find_dir('VisDrone2019-DET-train')
print('dataset:', SRC)

# ------------------------------------------------------------------ sequence-level split (identical to the experiment)
imgs_all = sorted((SRC / 'images').glob('*.jpg'))
seqs = sorted({p.stem.split('_')[0] for p in imgs_all})
random.Random(SPLIT_SEED).shuffle(seqs)
n = len(seqs); a, b = int(0.70 * n), int(0.85 * n)
split_of = {q: 'train' for q in seqs[:a]} | {q: 'val' for q in seqs[a:b]} | {q: 'test' for q in seqs[b:]}
SPLIT = {sp: [p for p in imgs_all if split_of[p.stem.split('_')[0]] == sp] for sp in ('train', 'val', 'test')}
print({k: len(v) for k, v in SPLIT.items()})

def load_gt(img):
    rows = []
    ann = SRC / 'annotations' / f'{img.stem}.txt'
    for r in ann.read_text().strip().splitlines():
        x, y, w, h, score, cat = map(int, r.split(',')[:6])
        if score == 0 or cat in (0, 11) or w <= 0 or h <= 0: continue
        rows.append((x, y, x + w, y + h, cat - 1))
    return np.array(rows, float).reshape(-1, 5)

def bucket_of(wh):
    s = np.sqrt(np.maximum(wh, 0).prod(-1))
    return np.digitize(s, [8, 16, 32])

STATS = {}
# ------------------------------------------------------------------ dataset statistics
size_by_split, cls_by_split, per_img = {}, {}, {}
for sp, lst in SPLIT.items():
    sizes, clss, cnt = [], [], []
    for p in lst:
        g = load_gt(p)
        sizes += list(np.sqrt((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]))); clss += list(g[:, 4].astype(int)); cnt.append(len(g))
    size_by_split[sp], cls_by_split[sp], per_img[sp] = np.array(sizes), np.array(clss), np.array(cnt)
    bk = np.digitize(size_by_split[sp], [8, 16, 32])
    STATS[f'dataset_{sp}'] = dict(images=len(lst), objects=int(len(sizes)), objects_per_image=float(np.mean(cnt)),
                                  size_bucket_fraction={BUCKETS[i][2]: float(np.mean(bk == i)) for i in range(4)},
                                  median_side_px=float(np.median(sizes)),
                                  per_class={NAMES[c]: int(np.sum(np.array(clss) == c)) for c in range(10)})
print(json.dumps({k: (v['images'], v['objects']) for k, v in STATS.items()}))

fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw=dict(wspace=0.38, width_ratios=[1.3, 1.2, 1]))
bins = np.logspace(np.log10(1), np.log10(400), 50)
for sp, col, ls in (('train', BLUE, '-'), ('val', ORANGE, '--'), ('test', AQUA, ':')):
    h, e = np.histogram(size_by_split[sp], bins=bins); h = h / h.sum()
    ax[0].step(e[:-1], h, where='post', color=col, lw=1.4, ls=ls, label=f'{sp} ({len(size_by_split[sp]):,})')
for v in (8, 16, 32): ax[0].axvline(v, color=INK2, lw=0.6, ls=(0, (2, 2)))
ax[0].set_xscale('log'); ax[0].set_xlabel('Object size $\\sqrt{wh}$ (pixels, original image)'); ax[0].set_ylabel('Fraction of objects')
ax[0].set_title('(a) Object sizes', loc='left'); ax[0].legend(loc='best', fontsize=7)
ax[0].set_xticks([2, 8, 32, 128]); ax[0].set_xticklabels(['2', '8', '32', '128']); ax[0].minorticks_off()
y = np.arange(10); hgt = 0.27
for k, (sp, col) in enumerate((('train', BLUE), ('val', ORANGE), ('test', AQUA))):
    c = np.bincount(cls_by_split[sp], minlength=10) / max(1, len(cls_by_split[sp]))
    ax[1].barh(y + (k - 1) * hgt, c, height=hgt, color=col, label=sp)
ax[1].set_yticks(y); ax[1].set_yticklabels(NAMES, fontsize=7); ax[1].invert_yaxis(); ax[1].set_xlabel('Fraction of objects')
ax[1].set_title('(b) Classes', loc='left'); ax[1].grid(axis='y', visible=False)
for sp, col, ls in (('train', BLUE, '-'), ('val', ORANGE, '--'), ('test', AQUA, ':')):
    h, e = np.histogram(per_img[sp], bins=np.arange(0, 421, 20)); ax[2].step(e[:-1], h / max(1, h.sum()), where='post', color=col, lw=1.4, ls=ls)
ax[2].set_xlabel('Objects per image'); ax[2].set_ylabel('Fraction of images'); ax[2].set_title('(c) Crowding', loc='left')
save(fig, 'fig_dataset_stats')

# sample images: six training images from six different clips, chosen at random with a fixed seed
rng = random.Random(1)
clips = sorted({p.stem.split('_')[0] for p in SPLIT['train']}); rng.shuffle(clips)
samples = []
for q in clips:
    cand = [p for p in SPLIT['train'] if p.stem.startswith(q + '_')]
    samples.append(rng.choice(cand))
    if len(samples) == 6: break
fig, axs = plt.subplots(2, 3, figsize=(7.0, 2.95), gridspec_kw=dict(wspace=0.03, hspace=0.04))
for ax, p in zip(axs.ravel(), samples):
    im = Image.open(p).convert('RGB'); W0, H0 = im.size; g = load_gt(p)
    if W0 / H0 < 16 / 9:                                   # centre crop to 16:9 so the grid has no gaps
        h2 = int(W0 * 9 / 16); top = (H0 - h2) // 2; im = im.crop((0, top, W0, top + h2)); g[:, [1, 3]] -= top
    ax.imshow(im); ax.set_xlim(0, im.size[0]); ax.set_ylim(im.size[1], 0)
    bk = bucket_of(g[:, 2:4] - g[:, 0:2])
    for (x1, y1, x2, y2, c), k in zip(g, bk):
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=SIZE_COL[k], lw=0.5))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.text(0.01, 0.98, f'{W0}×{H0}, {len(g)} objects', transform=ax.transAxes, va='top', fontsize=6.5,
            color='white', bbox=dict(boxstyle='round,pad=0.15', fc='black', ec='none', alpha=0.55))
handles = [Rectangle((0, 0), 1, 1, fill=False, ec=c, lw=1.2) for c in SIZE_COL]
fig.legend(handles, [f'{b[2]} px' for b in BUCKETS], loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=7.5)
save(fig, 'fig_dataset_samples')
STATS['dataset_samples'] = [p.name for p in samples]


# ================================================================== part B: assigners on real images
import types, torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors, dist2bbox

OT_SRC = r"""'''
Entropic-OT label assignment with a Gaussian-Wasserstein (NWD) cost for Ultralytics YOLOv8.

Math (matches the paper draft):
  - box -> phi(b) = (cx, cy, w/2, h/2). For axis-aligned Gaussians, W2(N_i, N_j) = ||phi_i - phi_j||_2.
  - loc cost  c_loc = 1 - exp(-W2 / C)      (Lipschitz in box coordinates with constant 1/C, any object size)
  - cls cost  c_cls = -logit(p_ic)          (= BCE_gt - BCE_bg; row-constant terms cancel in Sinkhorn)
  - C_ij = cls_w * c_cls + loc_w * c_loc ;  background column has cost 0
  - plan  pi = argmin <C,pi> - eps*H(pi) over U(a,b)   (log-domain Sinkhorn)
  - gt j takes its round(k_j) anchors with the largest plan mass (>=1 each); ties between gts go to the larger mass
  - cost='iou' switches c_loc to 1 - IoU (ablation); target='iou' switches only the soft cls target to IoU
  - nwd_adapt=True uses C_j = max(nwd_c, rho*sqrt(w_j h_j)): scale-aware, Lipschitz constant still <= 1/nwd_c
'''
import torch
from torchvision.ops import box_iou
from ultralytics.utils.tal import TaskAlignedAssigner

OT_CFG = dict(
    ot_eps=0.1,          # entropic regularisation epsilon
    sink_iters=50,       # Sinkhorn iterations
    nwd_c=12.8,          # NWD normalising constant (tune; dataset dependent)
    cls_w=1.0,
    loc_w=3.0,
    center_radius=2.5,   # candidate anchors: centre within R strides of gt centre
    min_cand=9,          # every gt always gets at least this many nearest anchors
    max_k=10,            # cap on dynamic k (demand of each gt)
    strides=(8, 16, 32),
    cost="nwd",          # similarity used in the transport cost: "nwd" or "iou"
    target="nwd",        # similarity used for the soft classification target: "nwd" or "iou"
    nwd_adapt=False,     # scale-adaptive normaliser C_j = max(nwd_c, nwd_rho * sqrt(w_j h_j)); Lipschitz const <= 1/nwd_c
    nwd_rho=0.25,
    big=1e4,             # cost of non-candidate pairs
)
_DEFAULT_CFG = dict(OT_CFG)


def phi(b):
    '''xyxy -> (cx, cy, w/2, h/2)'''
    cx = (b[..., 0] + b[..., 2]) / 2
    cy = (b[..., 1] + b[..., 3]) / 2
    hw = (b[..., 2] - b[..., 0]) / 2
    hh = (b[..., 3] - b[..., 1]) / 2
    return torch.stack([cx, cy, hw, hh], -1)


def w2_matrix(pred_xyxy, gt_xyxy):
    '''(M,4),(G,4) -> (M,G) Wasserstein-2 distance between Gaussian boxes.'''
    return torch.cdist(phi(pred_xyxy), phi(gt_xyxy))


def sinkhorn_log(C, a, b, eps, iters):
    '''Log-domain Sinkhorn. C:(M,N), a:(M,), b:(N,) positive, each summing to 1. Returns plan (M,N).'''
    logK = -C / eps
    la, lb = a.log(), b.log()
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    for _ in range(iters):
        f = la - torch.logsumexp(logK + g[None, :], dim=1)
        g = lb - torch.logsumexp(logK + f[:, None], dim=0)
    return torch.exp(logK + f[:, None] + g[None, :])


def infer_stride(anc, strides=(8, 16, 32)):
    '''Anchor centres are (k+0.5)*s, so the stride is recoverable from the x coordinate.'''
    x = anc[:, 0]
    out = torch.full_like(x, float(strides[-1]))
    for s in reversed(strides):
        frac = (x / s) % 1.0
        out = torch.where((frac - 0.5).abs() < 1e-3, torch.full_like(x, float(s)), out)
    return out


class OTAssigner(TaskAlignedAssigner):
    '''Drop-in replacement for TaskAlignedAssigner inside v8DetectionLoss.'''

    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0, eps=1e-9, **kw):
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps)
        self.cfg = dict(OT_CFG)

    # ---- one image -------------------------------------------------------------------
    def _assign_one(self, sc, pb, anc, strd, gb, gl):
        cfg = self.cfg
        G, A = gb.shape[0], pb.shape[0]
        gc = torch.stack([(gb[:, 0] + gb[:, 2]) / 2, (gb[:, 1] + gb[:, 3]) / 2], -1)
        nd = torch.cdist(gc, anc) / strd[None, :]                       # (G,A) distance in strides
        cand = nd <= cfg["center_radius"]
        top = nd.topk(min(cfg["min_cand"], A), dim=1, largest=False).indices
        cand.scatter_(1, top, True)
        cidx = cand.any(0).nonzero().squeeze(-1)                         # candidate anchors
        M = cidx.numel()
        cand = cand[:, cidx].T                                           # (M,G)
        pbc = pb[cidx]
        p = sc[cidx].clamp(1e-4, 1 - 1e-4)

        W2 = w2_matrix(pbc, gb)
        if cfg["nwd_adapt"]:
            area = ((gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])).clamp(min=1.0)
            Cj = (cfg["nwd_rho"] * area.sqrt()).clamp(min=cfg["nwd_c"])          # (G,)
        else:
            Cj = torch.full((G,), float(cfg["nwd_c"]), device=gb.device)
        nwd = torch.exp(-W2 / Cj[None, :])
        iou_m = box_iou(pbc, gb)
        sim = iou_m if cfg["cost"] == "iou" else nwd                              # drives cost and dynamic k
        tsim = iou_m if cfg["target"] == "iou" else nwd                           # drives the soft cls target
        pc = p[:, gl]                                                    # (M,G) score of each gt's class
        c_cls = -(pc.log() - (1 - pc).log())
        C = cfg["cls_w"] * c_cls + cfg["loc_w"] * (1 - sim)
        C = torch.where(cand, C, torch.full_like(C, cfg["big"]))

        # dynamic k (OTA / SimOTA style, on the chosen similarity)
        simc = torch.where(cand, sim, torch.zeros_like(sim))
        k = simc.topk(min(cfg["max_k"], M), dim=0).values.sum(0).clamp(1.0, cfg["max_k"])
        if k.sum() > 0.9 * M:
            k = k * (0.9 * M / k.sum())

        a = torch.full((M,), 1.0 / M, device=C.device)
        b = torch.cat([(M - k.sum())[None], k]) / M                      # [background, gt_1..gt_G]
        C_full = torch.cat([torch.zeros(M, 1, device=C.device), C], 1)
        plan = sinkhorn_log(C_full, a, b, cfg["ot_eps"], cfg["sink_iters"])

        # positives: for every gt take its round(k_j) anchors with the largest plan mass (>=1 per gt);
        # an anchor claimed by several gts keeps the gt with the largest mass.
        col = plan[:, 1:]                                                # (M,G)
        kk = k.round().clamp(min=1).long()
        order = col.argsort(0, descending=True)                          # (M,G)
        rank_ok = torch.arange(M, device=C.device)[:, None] < kk[None, :]
        P = torch.zeros_like(col, dtype=torch.bool).scatter_(0, order, rank_ok)
        j = torch.where(P, col, torch.full_like(col, -1.0)).argmax(1)
        pos = P.any(1)
        return cidx[pos], j[pos], tsim[pos, j[pos]]

    # ---- batch -----------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        device = gt_bboxes.device
        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]).bool(),
                torch.zeros_like(pd_scores[..., 0]).long(),
            )
        A = pd_scores.shape[1]
        pd_scores, pd_bboxes, anc_points = pd_scores.float(), pd_bboxes.float(), anc_points.float()
        gt_bboxes = gt_bboxes.float()
        strd = infer_stride(anc_points, self.cfg["strides"])

        mask_pos = torch.zeros(self.bs, self.n_max_boxes, A, device=device)
        overlaps = torch.zeros_like(mask_pos)
        align = torch.zeros_like(mask_pos)

        for bi in range(self.bs):
            gidx = mask_gt[bi, :, 0].bool().nonzero().squeeze(-1)
            if gidx.numel() == 0:
                continue
            gl = gt_labels[bi, gidx, 0].long()
            ai, gj, sim = self._assign_one(pd_scores[bi], pd_bboxes[bi], anc_points, strd, gt_bboxes[bi, gidx], gl)
            gi = gidx[gj]
            mask_pos[bi, gi, ai] = 1.0
            overlaps[bi, gi, ai] = sim
            score = pd_scores[bi, ai, gl[gj]]
            align[bi, gi, ai] = score.pow(self.alpha) * sim.pow(self.beta)

        fg_mask = mask_pos.sum(-2)                                       # (b,A)
        target_gt_idx = mask_pos.argmax(-2)                              # (b,A)
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        pos_align = align.amax(-1, keepdim=True)
        pos_ov = overlaps.amax(-1, keepdim=True)
        norm = (align * pos_ov / (pos_align + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx


# ---- switching between default and OT assigner --------------------------------------
def use_ot(**overrides):
    import ultralytics.utils.loss as L
    OT_CFG.clear(); OT_CFG.update(_DEFAULT_CFG); OT_CFG.update(overrides)       # no leakage between runs
    L.TaskAlignedAssigner = OTAssigner


def use_default():
    import ultralytics.utils.loss as L
    L.TaskAlignedAssigner = TaskAlignedAssigner"""
OTmod = types.ModuleType('ot_assigner'); exec(OT_SRC, OTmod.__dict__)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N_IMAGES = int(os.environ.get('N_IMAGES', 300))          # training-split images used for the statistics
torch.manual_seed(0)
model = DetectionModel('yolov8n.yaml', nc=10, verbose=False)
model.load(YOLO('yolov8n.pt').model, verbose=False)       # same initialisation as the start of training
model = model.to(DEV).eval()
det = model.model[-1]; reg_max = det.reg_max; proj = torch.arange(reg_max, dtype=torch.float, device=DEV)
tal = TaskAlignedAssigner(topk=10, num_classes=10, alpha=0.5, beta=6.0)
OTmod.use_ot(cost='nwd', target='nwd'); ota = OTmod.OTAssigner(topk=10, num_classes=10)

def letterbox(img):
    W, H = img.size; r = IMGSZ / max(W, H); w, h = round(W * r), round(H * r)
    canvas = Image.new('RGB', (IMGSZ, IMGSZ), (114, 114, 114)); canvas.paste(img.resize((w, h), Image.BILINEAR), (0, 0))
    return canvas, r

@torch.no_grad()
def assign_both(p):
    img = Image.open(p).convert('RGB'); lb, r = letterbox(img); g = load_gt(p)
    if len(g) == 0: return None
    x = torch.from_numpy(np.asarray(lb)).permute(2, 0, 1).float().div(255)[None].to(DEV)
    feats = model(x)[1]
    no = det.nc + reg_max * 4
    pd, ps = torch.cat([f.view(1, no, -1) for f in feats], 2).split((reg_max * 4, det.nc), 1)
    ps = ps.permute(0, 2, 1).contiguous(); pd = pd.permute(0, 2, 1).contiguous()
    anc, st = make_anchors(feats, det.stride, 0.5)
    pb = dist2bbox(pd.view(1, -1, 4, reg_max).softmax(3).matmul(proj), anc, xywh=False)
    gtb = torch.tensor(g[:, :4] * r, dtype=torch.float, device=DEV)[None]
    gtl = torch.tensor(g[:, 4:5], dtype=torch.float, device=DEV)[None]
    mask = torch.ones(1, len(g), 1, device=DEV)
    args = (ps.sigmoid(), (pb * st).float(), (anc * st).float(), gtl, gtb, mask)
    out = {}
    for name, A in (('tal', tal), ('ot', ota)):
        _, _, _, fg, gi = A(*args)
        fg = fg[0].bool(); gi = gi[0]
        out[name] = dict(anchors=torch.nonzero(fg).squeeze(-1).cpu().numpy(), gt=gi[fg].cpu().numpy())
    return dict(img=lb, r=r, g=g, anc=(anc * st).cpu().numpy(), st=st.squeeze(-1).cpu().numpy(), out=out)

rng = random.Random(3)
pool = list(SPLIT['train']); rng.shuffle(pool); pool = pool[:N_IMAGES]
edges = np.array([2, 4, 6, 8, 11, 16, 23, 32, 45, 64, 128, 512]); mid = np.sqrt(edges[:-1] * edges[1:])
cnt = {k: {'none': np.zeros(len(mid)), 'pos': np.zeros(len(mid))} for k in ('tal', 'ot')}; tot = np.zeros(len(mid))
bucket = {k: {'none': np.zeros(4), 'n': np.zeros(4), 'pos': np.zeros(4)} for k in ('tal', 'ot')}
keep = []
for t_, p in enumerate(pool):
    res = assign_both(p)
    if res is None: continue
    g = res['g']; sz = np.sqrt((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])); bi = np.digitize(sz, edges) - 1; qb = np.digitize(sz, [8, 16, 32])
    for b_ in range(len(mid)): tot[b_] += np.sum(bi == b_)
    for k in ('tal', 'ot'):
        npos = np.bincount(res['out'][k]['gt'].astype(int), minlength=len(g))
        for b_ in range(len(mid)):
            sel = bi == b_; cnt[k]['none'][b_] += np.sum(npos[sel] == 0); cnt[k]['pos'][b_] += np.sum(npos[sel])
        for q in range(4):
            sel = qb == q; bucket[k]['none'][q] += np.sum(npos[sel] == 0); bucket[k]['n'][q] += np.sum(sel); bucket[k]['pos'][q] += np.sum(npos[sel])
    keep.append((int(np.sum(sz < 16)), p, res))
    if (t_ + 1) % 50 == 0: print('assigned', t_ + 1, 'images')

STATS['assign'] = {'images': len(keep), 'objects': int(tot.sum()), 'bin_edges_px': edges.tolist(), 'objects_per_bin': tot.tolist()}
for k in ('tal', 'ot'):
    STATS['assign'][k] = dict(no_positive_fraction=(cnt[k]['none'] / np.maximum(tot, 1)).tolist(),
                              positives_per_object=(cnt[k]['pos'] / np.maximum(tot, 1)).tolist(),
                              by_size={BUCKETS[q][2]: dict(objects=int(bucket[k]['n'][q]),
                                                           no_positive=int(bucket[k]['none'][q]),
                                                           no_positive_fraction=float(bucket[k]['none'][q] / max(1, bucket[k]['n'][q])),
                                                           positives_per_object=float(bucket[k]['pos'][q] / max(1, bucket[k]['n'][q])))
                                       for q in range(4)})
print(json.dumps({k: STATS['assign'][k]['by_size'] for k in ('tal', 'ot')}, indent=1))

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw=dict(wspace=0.3))
for k, col, mk, lab in (('tal', ORANGE, 's', 'Task-aligned (YOLOv8 default)'), ('ot', BLUE, 'o', 'Entropic OT (Algorithm 1)')):
    m_ = tot > 0
    axs[0].plot(mid[m_], (cnt[k]['none'] / np.maximum(tot, 1))[m_], color=col, marker=mk, ms=4, lw=1.5, label=lab)
    axs[1].plot(mid[m_], (cnt[k]['pos'] / np.maximum(tot, 1))[m_], color=col, marker=mk, ms=4, lw=1.5, label=lab)
for ax in axs:
    for v in (8, 16, 32): ax.axvline(v, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.set_xscale('log'); ax.set_xticks([2, 4, 8, 16, 32, 64, 128]); ax.set_xticklabels(['2', '4', '8', '16', '32', '64', '128']); ax.minorticks_off()
    ax.set_xlabel('Object size $\\sqrt{wh}$ (pixels, original image)')
axs[0].set_ylabel('Fraction of objects'); axs[0].set_ylim(0, 1); axs[0].set_title('(a) Objects with no positive anchor', loc='left')
axs[1].set_ylabel('Positive anchors per object'); axs[1].set_ylim(bottom=0); axs[1].set_title('(b) Positives per object', loc='left')
axs[0].legend(loc='upper right', fontsize=7.5)
save(fig, 'fig_zero_pos')

# qualitative: the two analysed images with the most objects under 16 px (different clips), crop with most such objects
keep.sort(key=lambda z: -z[0]); chosen, seen = [], set()
for nt, p, res in keep:
    if p.stem.split('_')[0] in seen: continue
    chosen.append((p, res)); seen.add(p.stem.split('_')[0])
    if len(chosen) == 2: break
CROP = 192
fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.75), gridspec_kw=dict(wspace=0.03, hspace=0.1))
STATS['assign_examples'] = {}
for r_, (p, res) in enumerate(chosen):
    g = res['g'] * np.array([res['r']] * 4 + [1]); im = res['img']
    tiny = g[np.sqrt((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])) / res['r'] < 16]
    cx = (tiny[:, 0] + tiny[:, 2]) / 2; cy = (tiny[:, 1] + tiny[:, 3]) / 2; best, bxy = -1, (0, 0)
    for x0 in range(0, IMGSZ - CROP, 16):
        for y0 in range(0, IMGSZ - CROP, 16):
            c = np.sum((cx >= x0) & (cx < x0 + CROP) & (cy >= y0) & (cy < y0 + CROP))
            if c > best: best, bxy = c, (x0, y0)
    x0, y0 = bxy
    inside = (g[:, 0] >= x0) & (g[:, 2] <= x0 + CROP) & (g[:, 1] >= y0) & (g[:, 3] <= y0 + CROP)
    STATS['assign_examples'][p.name] = {}
    for c_, (key, title) in enumerate(((None, 'Ground truth (crop)'), ('tal', 'Task-aligned (YOLOv8 default)'), ('ot', 'Entropic OT (Algorithm 1)'))):
        ax = axs[r_, c_]; ax.imshow(im.crop((x0, y0, x0 + CROP, y0 + CROP)), extent=(x0, x0 + CROP, y0 + CROP, y0))
        ax.set_xlim(x0, x0 + CROP); ax.set_ylim(y0 + CROP, y0); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if key is None:
            for bx in g[inside]: ax.add_patch(Rectangle(bx[:2], bx[2] - bx[0], bx[3] - bx[1], fill=False, ec='white', lw=0.8))
            lbl = f'{int(inside.sum())} objects'
        else:
            A = res['out'][key]; npos = np.bincount(A['gt'].astype(int), minlength=len(g))
            for a_, j in zip(A['anchors'], A['gt']):
                if not inside[j]: continue
                pt = res['anc'][a_]; gc = ((g[j, 0] + g[j, 2]) / 2, (g[j, 1] + g[j, 3]) / 2)
                ax.plot([pt[0], gc[0]], [pt[1], gc[1]], color=BLUE, lw=0.5, alpha=0.8)
                ax.scatter([pt[0]], [pt[1]], s=6 + 2 * res['st'][a_] / 8, color=BLUE, edgecolor='white', linewidth=0.3, zorder=3)
            for j in np.where(inside)[0]:
                bx = g[j]; ok = npos[j] > 0
                ax.add_patch(Rectangle(bx[:2], bx[2] - bx[0], bx[3] - bx[1], fill=False, ec='white' if ok else YELLOW, lw=0.8 if ok else 1.1, ls='-' if ok else (0, (2, 1.2))))
            nz = int(np.sum(inside & (npos == 0)))
            lbl = f'{nz} of {int(inside.sum())} objects without a positive'
            STATS['assign_examples'][p.name][key] = dict(objects=int(inside.sum()), without_positive=nz)
        ax.text(0.02, 0.02, lbl, transform=ax.transAxes, fontsize=6.5, color='white', va='bottom',
                bbox=dict(boxstyle='round,pad=0.15', fc='black', ec='none', alpha=0.6))
        if r_ == 0: ax.set_title(title, loc='left', fontsize=8.5)
handles = [plt.Line2D([], [], marker='o', color=BLUE, lw=0.6, ms=4), Rectangle((0, 0), 1, 1, fill=False, ec=YELLOW, lw=1.2, ls=(0, (2, 1.2)))]
fig.legend(handles, ['positive anchor (line to its object)', 'object with no positive anchor'], loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.01), fontsize=7.5)
save(fig, 'fig_assign_real')
STATS['assign_rule'] = f'{N_IMAGES} random training-split images (seed 3); model initialised as at the start of training; examples: the two analysed images with the most objects under 16 px (different clips), {CROP}x{CROP} crop of the 960x960 input with the most such objects'

json.dump(STATS, open(OUT / 'figs_notrain_stats.json', 'w'), indent=1)
shutil.make_archive(str(OUT), 'zip', OUT)
print('done ->', str(OUT) + '.zip')
