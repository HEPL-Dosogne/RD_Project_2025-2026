"""
PatchCore-based anomaly detection pipeline.

This script implements a complete workflow for industrial anomaly detection,
including feature extraction using a pretrained WideResNet, memory bank
construction from normal samples, and evaluation through patch‑wise distance
analysis. It also provides visualization utilities and a command‑line interface
for training and inference.
"""

import os
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import matplotlib.pyplot as plt


# =========================
# 1. Dataset & transforms
# =========================

def make_dataloaders(data_dir, batch_size=16, num_workers=4):
    """
    data_dir/train/good
    data_dir/test/good
    data_dir/test/defect
    """

    common_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=common_transform,
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=common_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 1 image pour générer la carte d’anomalie
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, train_dataset, test_dataset


# =========================
# 2. Backbone & hooks
# =========================

class FeatureExtractor(nn.Module):
    """
    Wrapper sur WideResNet50_2 avec hooks sur les couches intermédiaires.
    """

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

        backbone = models.wide_resnet50_2(pretrained=True)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        self.backbone = backbone.to(self.device)

        # On veut les features de layer2 et layer3 (classique PatchCore)
        self.outputs = {}

        def get_hook(name):
            def hook(module, input, output):
                self.outputs[name] = output
            return hook

        self.backbone.layer2.register_forward_hook(get_hook("layer2"))
        self.backbone.layer3.register_forward_hook(get_hook("layer3"))

    @torch.no_grad()
    def forward(self, x):
        """
        x: [B,3,H,W]
        return: dict {"layer2": feat2, "layer3": feat3}
        """
        _ = self.backbone(x)  # on ne garde pas la sortie finale
        return self.outputs


def _embedding_concat(x, y):
    """
    Concatène les features patch-wise (PatchCore)
    x: [B, C1, H1, W1]
    y: [B, C2, H2, W2] avec H2/W2 = H1/2, W1/2 (par ex.)
    On interpole y pour matcher la taille de x.
    """

    B, C1, H1, W1 = x.shape
    B, C2, H2, W2 = y.shape

    if H1 != H2 or W1 != W2:
        y = F.interpolate(y, size=(H1, W1), mode="bilinear", align_corners=False)

    # concat channels
    z = torch.cat([x, y], dim=1)  # [B, C1+C2, H1, W1]
    return z


# =========================
# 3. Création de la memory bank
# =========================

def build_memory_bank(train_loader, feature_extractor, device="cuda",
                      max_patches=10000):
    """
    Parcourt toutes les images normales, extrait les embeddings patch-wise,
    et construit la memory bank (avec éventuellement un sous-échantillonnage).
    """
    feature_extractor.eval()

    all_patches = []

    with torch.no_grad():
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            feats = feature_extractor(imgs)
            f2 = feats["layer2"]  # [B, C2, H2, W2]
            f3 = feats["layer3"]  # [B, C3, H3, W3]
            emb = _embedding_concat(f2, f3)  # [B, C, H, W]

            B, C, H, W = emb.shape
            emb = emb.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
            all_patches.append(emb.cpu())

    all_patches = torch.cat(all_patches, dim=0)  # [N_patches, C]

    # Optionnel : coreset (ici, on fait au plus simple: random subset)
    if all_patches.shape[0] > max_patches:
        idx = torch.randperm(all_patches.shape[0])[:max_patches]
        all_patches = all_patches[idx]

    memory_bank = all_patches  # [M, C]
    return memory_bank


# =========================
# 4. Inference (score & map)
# =========================

def compute_anomaly_score_and_map(img, feature_extractor, memory_bank,
                                  device="cuda"):
    """
    img: [1,3,H,W]
    memory_bank: [M,C]
    return: score_image (float), anomaly_map (H,W numpy)
    """
    feature_extractor.eval()
    memory_bank = memory_bank.to(device)

    with torch.no_grad():
        img = img.to(device)
        feats = feature_extractor(img)
        f2 = feats["layer2"]
        f3 = feats["layer3"]
        emb = _embedding_concat(f2, f3)  # [1,C,H,W]
        B, C, H, W = emb.shape
        emb = emb.permute(0, 2, 3, 1).reshape(-1, C)  # [H*W, C]

        # distances L2 vers la memory bank
        # memory_bank: [M,C], emb: [N,C]
        # distance (N,M) = ||emb_i - mb_j||^2
        # pour éviter des matrices énormes, on peut faire par batch
        chunk_size = 2048
        dists_list = []
        for i in range(0, emb.shape[0], chunk_size):
            emb_chunk = emb[i:i+chunk_size]  # [c,C]
            # [c,1,C] - [1,M,C] -> [c,M,C] -> norme sur C
            dist = torch.cdist(emb_chunk.unsqueeze(0), memory_bank.unsqueeze(0), p=2).squeeze(0)  # [c,M]
            min_dist, _ = torch.min(dist, dim=1)  # [c]
            dists_list.append(min_dist)

        min_dists = torch.cat(dists_list, dim=0)  # [H*W]
        anomaly_map = min_dists.reshape(H, W)

        # score global image = max distance (comme PatchCore)
        score = anomaly_map.max().item()

        # normalisation pour la visualisation
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        anomaly_map = anomaly_map.cpu().numpy()

    return score, anomaly_map


# =========================
# 5. Sauvegarde / chargement
# =========================

def save_memory_bank(path, memory_bank):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(memory_bank, path)


def load_memory_bank(path, device="cuda"):
    memory_bank = torch.load(path, map_location=device)
    return memory_bank


# =========================
# 6. Visualisation
# =========================

def save_anomaly_visual(img_tensor, anomaly_map, out_path):
    """
    img_tensor: [3,H,W], normalisé ImageNet
    anomaly_map: [H',W'] numpy (déjà normalisée 0-1)
    """
    # denormaliser pour affichage
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()  # H,W,3

    # resize anomaly_map vers la taille de l’image si besoin
    H, W, _ = img.shape
    amap_t = torch.tensor(anomaly_map).unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
    amap_resized = F.interpolate(amap_t, size=(H, W), mode="bilinear", align_corners=False)
    amap_resized = amap_resized.squeeze().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img)
    ax[0].axis("off")
    ax[0].set_title("Image")

    ax[1].imshow(img)
    ax[1].imshow(amap_resized, alpha=0.5)
    ax[1].axis("off")
    ax[1].set_title("Anomaly map")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# =========================
# 7. Train mode (build memory bank)
# =========================

def train_patchcore(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    train_loader, _, train_ds, _ = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"[INFO] Nb images train: {len(train_ds)}")

    feature_extractor = FeatureExtractor(device=device)
    memory_bank = build_memory_bank(
        train_loader,
        feature_extractor,
        device=device,
        max_patches=args.max_patches,
    )

    print(f"[INFO] Memory bank size: {memory_bank.shape}")
    save_memory_bank(args.memory_path, memory_bank)
    print(f"[INFO] Memory bank saved to {args.memory_path}")


# =========================
# 8. Eval mode
# =========================

def eval_patchcore(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    _, test_loader, _, test_ds = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"[INFO] Nb images test: {len(test_ds)}")
    print(f"[INFO] Classes test (multi-classes originales): {test_ds.classes}")

    # indice de la classe 'good' dans ImageFolder
    idx_good = test_ds.class_to_idx["good"]
    print(f"[INFO] Index classe 'good' = {idx_good}")

    feature_extractor = FeatureExtractor(device=device)
    memory_bank = load_memory_bank(args.memory_path, device=device)

    all_scores = []
    all_labels_bin = []  # 0 = good, 1 = defect

    # pour visualiser quelques exemples
    n_vis = min(args.n_vis, len(test_ds))
    vis_indices = np.linspace(0, len(test_ds)-1, n_vis, dtype=int)
    vis_indices_set = set(vis_indices.tolist())

    for idx, (img, label) in enumerate(test_loader):
        score, amap = compute_anomaly_score_and_map(
            img,
            feature_extractor,
            memory_bank,
            device=device,
        )

        all_scores.append(score)

        # label ImageFolder multi-classes -> binaire good/defect
        label = label.item()
        binary_label = 0 if label == idx_good else 1
        all_labels_bin.append(binary_label)

        if idx in vis_indices_set:
            out_path = os.path.join(
                args.vis_dir,
                f"sample_{idx:04d}_label{binary_label}_score{score:.3f}.png"
            )
            save_anomaly_visual(img.squeeze(0), amap, out_path)

    all_scores = np.array(all_scores)
    all_labels_bin = np.array(all_labels_bin)

    # AUROC (1 = defect)
    roc = roc_auc_score(all_labels_bin, all_scores)
    print(f"\n[RESULT] ROC-AUC (defect=1): {roc:.4f}")

    # choix d’un seuil simple : percentile
    thresh = np.percentile(all_scores, args.threshold_percentile)
    print(f"[INFO] Threshold (percentile {args.threshold_percentile}): {thresh:.4f}")

    preds = (all_scores >= thresh).astype(int)

    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_labels_bin, preds))

    print("\nClassification report:")
    print(classification_report(all_labels_bin, preds, target_names=["good", "defect"]))

    # Sauvegarder les scores si besoin
    if args.save_scores:
        np.savez(
            args.scores_path,
            scores=all_scores,
            labels=all_labels_bin,
            threshold=thresh,
        )
        print(f"[INFO] Scores saved to {args.scores_path}")



# =========================
# 9. Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="PatchCore anomaly detection")

    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Répertoire racine des données (contient train/ et test/)")
    parser.add_argument("--memory-path", type=str, default="models/patchcore_memory.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-patches", type=int, default=10000,
                        help="Taille max de la memory bank (coreset random).")

    # eval
    parser.add_argument("--threshold-percentile", type=float, default=90.0,
                        help="Seuil sur les scores d’anomalie (percentile).")
    parser.add_argument("--vis-dir", type=str, default="results/vis")
    parser.add_argument("--n-vis", type=int, default=10,
                        help="Nombre d’images à sauvegarder avec carte d’anomalie.")
    parser.add_argument("--save-scores", action="store_true")
    parser.add_argument("--scores-path", type=str, default="results/scores_patchcore.npz")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if args.mode == "train":
        train_patchcore(args)
    elif args.mode == "eval":
        eval_patchcore(args)

