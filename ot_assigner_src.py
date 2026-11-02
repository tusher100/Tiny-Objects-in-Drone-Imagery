"""
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
"""
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
    """xyxy -> (cx, cy, w/2, h/2)"""
    cx = (b[..., 0] + b[..., 2]) / 2
    cy = (b[..., 1] + b[..., 3]) / 2
    hw = (b[..., 2] - b[..., 0]) / 2
    hh = (b[..., 3] - b[..., 1]) / 2
    return torch.stack([cx, cy, hw, hh], -1)


def w2_matrix(pred_xyxy, gt_xyxy):
    """(M,4),(G,4) -> (M,G) Wasserstein-2 distance between Gaussian boxes."""
    return torch.cdist(phi(pred_xyxy), phi(gt_xyxy))


def sinkhorn_log(C, a, b, eps, iters):
    """Log-domain Sinkhorn. C:(M,N), a:(M,), b:(N,) positive, each summing to 1. Returns plan (M,N)."""
    logK = -C / eps
    la, lb = a.log(), b.log()
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)
    for _ in range(iters):
        f = la - torch.logsumexp(logK + g[None, :], dim=1)
        g = lb - torch.logsumexp(logK + f[:, None], dim=0)
    return torch.exp(logK + f[:, None] + g[None, :])


def infer_stride(anc, strides=(8, 16, 32)):
    """Anchor centres are (k+0.5)*s, so the stride is recoverable from the x coordinate."""
    x = anc[:, 0]
    out = torch.full_like(x, float(strides[-1]))
    for s in reversed(strides):
        frac = (x / s) % 1.0
        out = torch.where((frac - 0.5).abs() < 1e-3, torch.full_like(x, float(s)), out)
    return out


class OTAssigner(TaskAlignedAssigner):
    """Drop-in replacement for TaskAlignedAssigner inside v8DetectionLoss."""

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
    L.TaskAlignedAssigner = TaskAlignedAssigner