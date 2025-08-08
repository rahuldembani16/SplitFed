"""
Privacy Evaluation Harness for FL, SL, and SFL on PlantVillage

Metrics implemented:
- Membership Inference Attack (MIA, loss-threshold) on final/global models.
- Activation Inversion (optimize input to match client cut-layer activations) for SL/SFL client models.

Usage (example):
  python privacy_eval.py --dataset ./dataset/plantvillage 
                         --check mia activation_inversion 
                         --methods FL SL SFL 
                         --samples 16

Outputs: prints metrics per method; higher MIA AUC/attack-acc = worse privacy; higher PSNR from inversion = worse privacy.
"""

import argparse
import math
import os
from typing import List, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


# -------------------------------
# Dataset utilities
# -------------------------------

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)


def collect_samples_from_dir(directory: str) -> List[Tuple[str, str]]:
    samples: List[Tuple[str, str]] = []
    if not os.path.exists(directory):
        return samples
    valid_ext = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    for cls in sorted(os.listdir(directory)):
        cls_dir = os.path.join(directory, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if any(fname.endswith(ext) for ext in valid_ext):
                samples.append((os.path.join(cls_dir, fname), cls))
    return samples


def setup_dataset(root_hint: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[str], Dict[str, int]]:
    candidates = [
        root_hint,
        "./dataset/plantvillage",
        "./plantvillage_dataset",
        "./PlantVillage",
    ]
    root = None
    for p in candidates:
        if p and os.path.exists(p):
            root = p
            break
    if root is None:
        raise FileNotFoundError(f"PlantVillage dataset not found. Tried: {candidates}")

    # Check for pre-split variants
    for tr, va in [("train", "val"), ("train", "test")]:
        train_dir = os.path.join(root, tr)
        val_dir = os.path.join(root, va)
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            train_samples = collect_samples_from_dir(train_dir)
            val_samples = collect_samples_from_dir(val_dir)
            labels = sorted(list({lbl for _, lbl in train_samples + val_samples}))
            class_to_idx = {c: i for i, c in enumerate(labels)}
            return train_samples, val_samples, labels, class_to_idx

    # Single-directory; auto split 80/20 stratified
    all_samples = collect_samples_from_dir(root)
    if not all_samples:
        raise RuntimeError(f"No image samples found under {root}")
    paths, lbls = zip(*all_samples)
    labels = sorted(list(set(lbls)))
    class_to_idx = {c: i for i, c in enumerate(labels)}

    from sklearn.model_selection import train_test_split

    tr_p, va_p, tr_l, va_l = train_test_split(
        list(paths), list(lbls), test_size=0.2, random_state=SEED, stratify=lbls
    )
    train_samples = list(zip(tr_p, tr_l))
    val_samples = list(zip(va_p, va_l))
    return train_samples, val_samples, labels, class_to_idx


class PlantVillageDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str]], class_to_idx: Dict[str, int], transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, lbl = self.samples[idx]
        image = Image.open(p).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[lbl]


def build_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return tfm


def make_loaders(train_samples, val_samples, class_to_idx, batch_size=32):
    tfm = build_transforms()
    ds_tr = PlantVillageDataset(train_samples, class_to_idx, tfm)
    ds_va = PlantVillageDataset(val_samples, class_to_idx, tfm)
    return (
        DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0),
    )


# -------------------------------
# Models (recreate architectures used in training scripts)
# -------------------------------

class MobileNetV2Client(nn.Module):
    """Client-side model used in SL/SFL scripts (features up to layer 14 + 96->128 conv)."""

    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.features = nn.Sequential(*list(mobilenet.features.children())[:14])
        for p in self.features.parameters():
            p.requires_grad = False
        self.custom_layer = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.custom_layer(x)
        return x


class MobileNetV2Server(nn.Module):
    """Server-side model used in SL/SFL scripts (128->320 conv -> GAP -> linear)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(128, 320, kernel_size=1, stride=1),
            nn.BatchNorm2d(320),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(320, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def build_fl_model(num_classes: int) -> nn.Module:
    net = models.mobilenet_v2(pretrained=True)
    in_f = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_f, num_classes)
    return net


# -------------------------------
# Metrics: Membership Inference (loss threshold)
# -------------------------------

@torch.no_grad()
def collect_losses(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    losses: List[float] = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        batch_losses = loss_fn(logits, y).detach().cpu().numpy().tolist()
        losses.extend(batch_losses)
    return np.array(losses, dtype=np.float32)


def mia_loss_threshold(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: torch.device):
    """
    Yeom-style loss-threshold MIA.
    Returns: dict with threshold, attack_acc, auc (if sklearn available), train_mean, test_mean.
    """
    tr_losses = collect_losses(model, train_loader, device)
    te_losses = collect_losses(model, test_loader, device)

    # Best threshold maximizing TPR - FPR (Youden's J statistic)
    all_vals = np.concatenate([tr_losses, te_losses])
    labels = np.concatenate([np.ones_like(tr_losses), np.zeros_like(te_losses)])
    order = np.argsort(all_vals)
    sorted_vals = all_vals[order]
    sorted_labels = labels[order]

    # Sweep thresholds between consecutive unique values
    P = sorted_labels.sum()
    N = len(sorted_labels) - P
    best_J = -1.0
    best_thr = sorted_vals[0] - 1e-6
    positives_seen = 0
    negatives_seen = 0
    for i in range(len(sorted_vals)):
        lbl = sorted_labels[i]
        if lbl == 1:
            positives_seen += 1
        else:
            negatives_seen += 1
        # Threshold between i and i+1
        tpr = (P - positives_seen) / max(P, 1)
        fpr = (N - negatives_seen) / max(N, 1)
        J = tpr - fpr
        if J > best_J:
            best_J = J
            best_thr = sorted_vals[i]

    # Compute attack accuracy at best_thr (predict member if loss <= thr)
    tr_pred = (tr_losses <= best_thr).astype(np.int32)
    te_pred = (te_losses <= best_thr).astype(np.int32)
    attack_acc = (tr_pred.mean() + (1 - te_pred).mean()) / 2.0

    # AUC if sklearn available
    auc = None
    try:
        from sklearn.metrics import roc_auc_score

        y_true = np.concatenate([np.ones_like(tr_losses), np.zeros_like(te_losses)])
        y_score = -np.concatenate([tr_losses, te_losses])  # lower loss => more member
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        pass

    return {
        "threshold": float(best_thr),
        "attack_acc": float(attack_acc),
        "auc": auc,
        "train_loss_mean": float(tr_losses.mean()),
        "test_loss_mean": float(te_losses.mean()),
    }


# -------------------------------
# Metrics: Activation Inversion (reconstruction)
# -------------------------------

def total_variation(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = x[:, :, :, 1:] - x[:, :, :, :-1]
    return (dx.abs().mean() + dy.abs().mean())


def psnr(mse: float, max_val: float = 1.0) -> float:
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10((max_val * max_val) / mse)


@torch.no_grad()
def denorm(x: torch.Tensor) -> torch.Tensor:
    # Inverse of ImageNet normalization
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x * std) + mean


def activation_inversion_metric(client_model: nn.Module, loader: DataLoader, device: torch.device, 
                                samples: int = 8, iters: int = 300, tv_lambda: float = 1e-4):
    client_model.eval()
    collected = 0
    mse_list: List[float] = []
    psnr_list: List[float] = []

    for x, _ in loader:
        x = x.to(device)
        with torch.no_grad():
            target_act = client_model(x)

        B = x.size(0)
        for b in range(B):
            if collected >= samples:
                break
            tgt = target_act[b : b + 1]
            z = torch.randn_like(x[b : b + 1], device=device, requires_grad=True)
            opt = torch.optim.Adam([z], lr=0.1)

            for _ in range(iters):
                opt.zero_grad()
                act = client_model(z)
                loss = nn.functional.mse_loss(act, tgt) + tv_lambda * total_variation(z)
                loss.backward()
                opt.step()
                # Keep z within roughly normalized image space bounds
                with torch.no_grad():
                    z.clamp_(min=-3.0, max=3.0)

            with torch.no_grad():
                x_den = denorm(x[b : b + 1]).clamp(0, 1)
                z_den = denorm(z).clamp(0, 1)
                mse = nn.functional.mse_loss(z_den, x_den).item()
                mse_list.append(mse)
                psnr_list.append(psnr(mse, 1.0))
            collected += 1
        if collected >= samples:
            break

    return {
        "mse_mean": float(np.mean(mse_list)) if mse_list else None,
        "psnr_mean": float(np.mean(psnr_list)) if psnr_list else None,
        "samples": collected,
    }


# -------------------------------
# Orchestration per method
# -------------------------------

def evaluate_method(method: str, dataset_root: str, device: torch.device, samples: int):
    train_samples, val_samples, classes, class_to_idx = setup_dataset(dataset_root)
    train_loader, val_loader = make_loaders(train_samples, val_samples, class_to_idx, batch_size=32)
    num_classes = len(classes)

    results = {"method": method}

    if method == "FL":
        model = build_fl_model(num_classes).to(device)
        # Try known save paths
        ckpts = [
            "./model/FL_MobileNetV2_PlantVillage_final.pth",
            "./model/FL_MobileNetV2_PlantVillage_best.pth",
        ]
        loaded = False
        for p in ckpts:
            if os.path.exists(p):
                model.load_state_dict(torch.load(p, map_location=device))
                loaded = True
                break
        if not loaded:
            raise FileNotFoundError(f"FL checkpoint not found in {ckpts}")

        results["mia"] = mia_loss_threshold(model, train_loader, val_loader, device)
        # No activation inversion (no split layer exposed)

    elif method == "SL":
        client = MobileNetV2Client().to(device)
        server = MobileNetV2Server(num_classes).to(device)
        ckp_client = "./model/SL_MobileNetV2_client_PlantVillage_final.pth"
        ckp_server = "./model/SL_MobileNetV2_server_PlantVillage_final.pth"
        if not (os.path.exists(ckp_client) and os.path.exists(ckp_server)):
            raise FileNotFoundError("SL checkpoints not found. Train SL script first.")
        client.load_state_dict(torch.load(ckp_client, map_location=device))
        server.load_state_dict(torch.load(ckp_server, map_location=device))

        # Compose full model for MIA
        full = nn.Sequential(client, server).to(device)
        results["mia"] = mia_loss_threshold(full, train_loader, val_loader, device)
        results["activation_inversion"] = activation_inversion_metric(client, val_loader, device, samples=samples)

    elif method == "SFL":
        client = MobileNetV2Client().to(device)
        server = MobileNetV2Server(num_classes).to(device)
        ckp_client = "./model/SFL_MobileNetV2_client_PlantVillage_final.pth"
        ckp_server = "./model/SFL_MobileNetV2_server_PlantVillage_final.pth"
        if not (os.path.exists(ckp_client) and os.path.exists(ckp_server)):
            raise FileNotFoundError("SFL checkpoints not found. Train SFL script first.")
        client.load_state_dict(torch.load(ckp_client, map_location=device))
        server.load_state_dict(torch.load(ckp_server, map_location=device))

        full = nn.Sequential(client, server).to(device)
        results["mia"] = mia_loss_threshold(full, train_loader, val_loader, device)
        results["activation_inversion"] = activation_inversion_metric(client, val_loader, device, samples=samples)

    else:
        raise ValueError("method must be one of: FL, SL, SFL")

    return results


def main():
    parser = argparse.ArgumentParser(description="Privacy evaluation for FL/SL/SFL")
    parser.add_argument("--dataset", type=str, default="./dataset/plantvillage")
    parser.add_argument("--methods", nargs="+", default=["FL", "SL", "SFL"], help="Methods to evaluate")
    parser.add_argument("--check", nargs="+", default=["mia", "activation_inversion"],
                        help="Which checks to run (mia, activation_inversion)")
    parser.add_argument("--samples", type=int, default=8, help="Inversion samples per method (SL/SFL)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Running privacy evaluation on device: {device}")
    print(f"Methods: {args.methods}")
    print(f"Checks: {args.check}")

    for m in args.methods:
        try:
            res = evaluate_method(m, args.dataset, device, samples=args.samples)
        except Exception as e:
            print(f"[{m}] ERROR: {e}")
            continue

        print(f"\n=== {m} ===")
        if "mia" in args.check and "mia" in res:
            mia = res["mia"]
            print(f"MIA (loss-threshold): attack_acc={mia['attack_acc']:.3f}, "
                  f"auc={mia['auc'] if mia['auc'] is not None else 'n/a'}, "
                  f"thr={mia['threshold']:.4f}, tr_loss_mean={mia['train_loss_mean']:.4f}, te_loss_mean={mia['test_loss_mean']:.4f}")
        if m in ("SL", "SFL") and "activation_inversion" in args.check and "activation_inversion" in res:
            inv = res["activation_inversion"]
            print(f"Activation inversion: samples={inv['samples']}, mse_mean={inv['mse_mean']:.6f} "
                  f"psnr_mean={inv['psnr_mean']:.2f} dB")


if __name__ == "__main__":
    main()
