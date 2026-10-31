"""Image quality control and suspicious-region prioritisation for a radiologist worklist.

Lab prototype on SYNTHETIC phantom images only (TRL 4). Classical image processing (no trained network);
NOT a diagnostic device: it only orders the worklist and flags images for re-acquisition.
"""
import numpy as np

SIZE = 64
DEFECTS = ["blur", "noise", "low_contrast", "cropped"]

CRITERIA = [
    ("qc_recall", ">=", 0.90, "defective images flagged by QC"),
    ("qc_false_reject", "<=", 0.10, "good images wrongly rejected by QC"),
    ("lesion_auc", ">=", 0.85, "ROC AUC of suspicion score (QC-passed images)"),
    ("top_quartile_precision", ">=", 0.90, "share of lesion images within the top 25% of the read order"),
]


def _conv1(img, k, axis):
    pad = len(k) // 2
    p = np.pad(img, [(pad, pad) if a == axis else (0, 0) for a in range(2)], mode="reflect")
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), axis, p)


def gauss(img, s):
    r = int(3 * s) + 1; x = np.arange(-r, r + 1); k = np.exp(-x * x / (2 * s * s)); k /= k.sum()
    return _conv1(_conv1(img, k, 0), k, 1)


def make_image(rng, lesion=False, defect=None):
    yy, xx = np.mgrid[:SIZE, :SIZE]
    cy, cx = SIZE / 2 + rng.normal(0, 2), SIZE / 2 + rng.normal(0, 2)
    ry, rx = rng.uniform(20, 24), rng.uniform(15, 19)
    body = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
    tex = gauss(rng.normal(0, 1, (SIZE, SIZE)), 3.0); tex = tex / tex.std() * 0.04
    img = np.where(body, 0.5, 0.05) + np.where(body, tex, 0)
    if lesion:
        for _ in range(int(rng.integers(1, 3))):
            ang = rng.uniform(0, 2 * np.pi); rr = rng.uniform(0, 0.6)
            ly, lx = cy + rr * ry * np.sin(ang), cx + rr * rx * np.cos(ang)
            s = rng.uniform(2.0, 3.5); amp = rng.uniform(0.18, 0.30)
            img = img + amp * np.exp(-((yy - ly) ** 2 + (xx - lx) ** 2) / (2 * s * s)) * body
    if defect == "blur":
        img = gauss(img, 2.2)
    elif defect == "low_contrast":
        img = 0.35 + (img - img.mean()) * 0.35
    elif defect == "cropped":
        shift = int(rng.integers(24, 32)); img = np.roll(img, shift, axis=1); img[:, :shift] = img[:, shift:shift + 1]
        body = np.roll(body, shift, axis=1); body[:, :shift] = False
    img = img + rng.normal(0, 0.012 if defect != "noise" else 0.09, img.shape)
    return np.clip(img, 0, 1), body


def _mask(img):
    lo, hi = np.percentile(img, 5), np.percentile(img, 95)
    return img > (lo + hi) / 2, hi - lo


def qc_metrics(img):
    mask, rng_ = _mask(img)
    hp = img - gauss(img, 1.0)
    bg = ~mask
    noise = float(1.4826 * np.median(np.abs(hp[bg] - np.median(hp[bg])))) if bg.sum() > 50 else float(hp.std())
    edge = np.hypot(*np.gradient(gauss(img, 1.0)))
    near = gauss(mask.astype(float), 2) > 0.1
    sharp = float(np.percentile(edge[near], 99)) / max(rng_, 1e-6) if near.any() else 0.0
    border = float(mask[:, 0].mean() + mask[:, -1].mean() + mask[0].mean() + mask[-1].mean())
    return {"sharpness": sharp, "noise": noise, "contrast": float(rng_), "border_touch": border}


QC_LIMITS = {"sharpness_min": 0.25, "noise_max": 0.03, "contrast_min": 0.30, "border_touch_max": 0.08}


def qc_decision(m):
    r = []
    if m["sharpness"] < QC_LIMITS["sharpness_min"]: r.append("blurred")
    if m["noise"] > QC_LIMITS["noise_max"]: r.append("noisy")
    if m["contrast"] < QC_LIMITS["contrast_min"]: r.append("low contrast")
    if m["border_touch"] > QC_LIMITS["border_touch_max"]: r.append("anatomy cut at border")
    return {"pass": not r, "reasons": r}


def suspicion(img):
    """Robust z-score of the strongest blob-like (difference-of-Gaussians) response inside the body, using masked smoothing."""
    mask, _ = _mask(img)
    m = mask.astype(float)
    sm = lambda s: gauss(img * m, s) / (gauss(m, s) + 1e-6)
    dog = sm(2.0) - sm(6.0)
    inner = gauss(m, 3.0) > 0.9
    ref = dog[inner]
    if ref.size < 50:
        return 0.0
    mad = 1.4826 * np.median(np.abs(ref - np.median(ref))) + 1e-6
    return float((ref.max() - np.median(ref)) / mad)


def worklist(items):
    """items: dicts with id, qc (dict) and score. Re-acquisition items are listed separately; the rest ordered by score."""
    rework = [i["id"] for i in items if not i["qc"]["pass"]]
    ordered = sorted([i for i in items if i["qc"]["pass"]], key=lambda i: -i["score"])
    return {"read_order": [i["id"] for i in ordered], "reacquire": rework}


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def simulate(seed=0, n=300):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        defect = DEFECTS[int(rng.integers(0, 4))] if rng.random() < 0.3 else None
        lesion = bool(rng.random() < 0.4)
        img, _ = make_image(rng, lesion, defect)
        qc = qc_decision(qc_metrics(img))
        out.append({"id": "IMG%04d" % i, "lesion": lesion, "defect": defect, "qc": qc, "qc_metrics": qc_metrics(img), "score": suspicion(img)})
    return out


def evaluate(seed=0):
    d = simulate(seed)
    bad = [x for x in d if x["defect"]]; good = [x for x in d if not x["defect"]]
    passed = [x for x in d if x["qc"]["pass"] and not x["defect"]]
    wl = worklist(d)
    ordered = {x["id"]: x for x in d}
    top = wl["read_order"][: max(1, len(wl["read_order"]) // 4)]
    total_l = sum(ordered[i]["lesion"] for i in wl["read_order"])
    return {"qc_recall": float(np.mean([not x["qc"]["pass"] for x in bad])), "qc_false_reject": float(np.mean([not x["qc"]["pass"] for x in good])),
            "lesion_auc": auc([x["lesion"] for x in passed], [x["score"] for x in passed]),
            "top_quartile_precision": float(sum(ordered[i]["lesion"] for i in top) / max(len(top), 1)), "n_images": len(d)}


def generate(seed=0):
    d = simulate(seed)
    return {"images": [dict(id=x["id"], lesion=int(x["lesion"]), defect=x["defect"] or "", qc_pass=int(x["qc"]["pass"]), qc_reasons=";".join(x["qc"]["reasons"]),
                            suspicion=round(x["score"], 2), **{k: round(v, 4) for k, v in x["qc_metrics"].items()}) for x in d]}


def demo(seed=0):
    d = simulate(seed, 40)
    w = worklist(d)
    return "read order (first 5): %s\nre-acquire: %s" % (w["read_order"][:5], w["reacquire"])
