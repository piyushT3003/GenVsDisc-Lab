"""
Training script for the Generative vs. Discriminative Model Lab.

Trains two models on MNIST with PyTorch and exports their weights to plain
NumPy (.npz) files so the web app can run inference WITHOUT PyTorch:

  1. Discriminative model  -> small CNN classifier          models/classifier.npz
     learns p(y | x) directly: "which digit is this?"

  2. Generative model      -> Conditional VAE (CVAE)         models/cvae.npz
     learns p(x | y): "what does a digit y look like?" and can SAMPLE new digits.

It also writes models/metrics.json (accuracy, loss curves, confusion matrix,
ELBO statistics) and models/sample_digits.npz (a few real test digits for the UI).

You do NOT need to run this to use the app - trained weights are already included.
To retrain:
    pip install torch numpy
    python training/train_models.py --data path/to/mnist_folder

The data folder must contain the 4 standard MNIST files (optionally .gz):
    train-images-idx3-ubyte  train-labels-idx1-ubyte
    t10k-images-idx3-ubyte   t10k-labels-idx1-ubyte
If --data is omitted, torchvision is used to download MNIST.
"""
import argparse
import gzip
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "models")
LATENT = 2  # 2-D latent space so the "style map" can be plotted directly


# ----------------------------------------------------------------------------- data
def _read_idx(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def _find(folder, name):
    for cand in (name, name + ".gz"):
        p = os.path.join(folder, cand)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name} not found in {folder}")


def load_mnist(folder=None):
    if folder:
        xtr = _read_idx(_find(folder, "train-images-idx3-ubyte"))
        ytr = _read_idx(_find(folder, "train-labels-idx1-ubyte"))
        xte = _read_idx(_find(folder, "t10k-images-idx3-ubyte"))
        yte = _read_idx(_find(folder, "t10k-labels-idx1-ubyte"))
    else:
        from torchvision import datasets
        tr = datasets.MNIST(os.path.join(ROOT, "training", "data"), train=True, download=True)
        te = datasets.MNIST(os.path.join(ROOT, "training", "data"), train=False, download=True)
        xtr, ytr = tr.data.numpy(), tr.targets.numpy()
        xte, yte = te.data.numpy(), te.targets.numpy()
    f = lambda a: a.astype(np.float32) / 255.0
    return f(xtr), ytr.astype(np.int64), f(xte), yte.astype(np.int64)


# ----------------------------------------------------------------------------- models
class CNN(nn.Module):
    """Discriminative model: image -> 10 class scores."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # 28x28x16 -> pool -> 14x14
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 14x14x32 -> pool -> 7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.drop(F.relu(self.fc1(x.flatten(1))))
        return self.fc2(x)


class CVAE(nn.Module):
    """Generative model: learns p(x | y) through a latent variable z."""

    def __init__(self, hidden=512, latent=LATENT):
        super().__init__()
        self.enc1 = nn.Linear(784 + 10, hidden)
        self.enc2 = nn.Linear(hidden, hidden // 2)
        self.mu = nn.Linear(hidden // 2, latent)
        self.logvar = nn.Linear(hidden // 2, latent)
        self.dec1 = nn.Linear(latent + 10, hidden // 2)
        self.dec2 = nn.Linear(hidden // 2, hidden)
        self.dec3 = nn.Linear(hidden, 784)

    def encode(self, x, y1h):
        h = F.relu(self.enc1(torch.cat([x, y1h], 1)))
        h = F.relu(self.enc2(h))
        return self.mu(h), self.logvar(h)

    def decode(self, z, y1h):
        h = F.relu(self.dec1(torch.cat([z, y1h], 1)))
        h = F.relu(self.dec2(h))
        return self.dec3(h)  # logits of Bernoulli pixels

    def forward(self, x, y1h):
        mu, logvar = self.encode(x, y1h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z, y1h), mu, logvar


def elbo_terms(logits, x, mu, logvar):
    rec = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(1)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    return rec, kl


# ----------------------------------------------------------------------------- training
def augment(xb):
    """Small random shifts/rotations so the classifier copes with hand-drawn input."""
    n = xb.shape[0]
    ang = (torch.rand(n) - 0.5) * 2 * np.deg2rad(12)
    sc = 1 + (torch.rand(n) - 0.5) * 0.2
    tx, ty = (torch.rand(n) - 0.5) * 0.2, (torch.rand(n) - 0.5) * 0.2
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, xb.shape, align_corners=False)
    return F.grid_sample(xb, grid, align_corners=False)


def train_classifier(xtr, ytr, xte, yte, epochs):
    torch.manual_seed(0)
    model = CNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 2), gamma=0.3)
    Xtr, Ytr = torch.tensor(xtr).unsqueeze(1), torch.tensor(ytr)
    Xte, Yte = torch.tensor(xte).unsqueeze(1), torch.tensor(yte)
    hist = {"train_loss": [], "train_acc": [], "test_acc": []}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        tot, correct, loss_sum = 0, 0, 0.0
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            xb, yb = augment(Xtr[idx]), Ytr[idx]
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(idx); tot += len(idx)
            correct += (out.argmax(1) == yb).sum().item()
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(Xte[i:i + 1000]).argmax(1) for i in range(0, len(Xte), 1000)])
        te_acc = (pred == Yte).float().mean().item()
        hist["train_loss"].append(round(loss_sum / tot, 4))
        hist["train_acc"].append(round(correct / tot, 4))
        hist["test_acc"].append(round(te_acc, 4))
        print(f"[classifier] epoch {ep + 1}/{epochs}  loss {loss_sum / tot:.4f}  test acc {te_acc:.4f}")
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(yte, pred.numpy()):
        cm[t, p] += 1
    return model, hist, cm


def train_cvae(xtr, ytr, xte, yte, epochs):
    torch.manual_seed(0)
    model = CVAE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr, Ytr = torch.tensor(xtr).view(-1, 784), F.one_hot(torch.tensor(ytr), 10).float()
    hist = {"loss": [], "recon": [], "kl": []}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        s_rec = s_kl = 0.0
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            xb, yb = Xtr[idx], Ytr[idx]
            logits, mu, logvar = model(xb, yb)
            rec, kl = elbo_terms(logits, xb, mu, logvar)
            loss = (rec + kl).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            s_rec += rec.sum().item(); s_kl += kl.sum().item()
        n = len(Xtr)
        hist["loss"].append(round((s_rec + s_kl) / n, 2))
        hist["recon"].append(round(s_rec / n, 2))
        hist["kl"].append(round(s_kl / n, 2))
        print(f"[cvae] epoch {ep + 1}/{epochs}  -ELBO {(s_rec + s_kl) / n:.2f}  (recon {s_rec / n:.2f}, KL {s_kl / n:.2f})")
    return model, hist


def export(model, path):
    arrays = {k.replace(".", "_"): v.detach().cpu().numpy().astype(np.float32)
              for k, v in model.state_dict().items()}
    np.savez_compressed(path, **arrays)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="folder with MNIST idx files")
    ap.add_argument("--clf-epochs", type=int, default=6)
    ap.add_argument("--vae-epochs", type=int, default=40)
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    os.makedirs(OUT, exist_ok=True)

    xtr, ytr, xte, yte = load_mnist(a.data)
    print("train", xtr.shape, "test", xte.shape)

    t0 = time.time()
    clf, clf_hist, cm = train_classifier(xtr, ytr, xte, yte, a.clf_epochs)
    clf_time = time.time() - t0
    export(clf, os.path.join(OUT, "classifier.npz"))

    t0 = time.time()
    vae, vae_hist = train_cvae(xtr, ytr, xte, yte, a.vae_epochs)
    vae_time = time.time() - t0
    export(vae, os.path.join(OUT, "cvae.npz"))

    # Generative classification on the test set via Bayes' rule:
    # p(y|x) ∝ p(x|y) p(y),  with log p(x|y) approximated by the ELBO.
    vae.eval()
    with torch.no_grad():
        X = torch.tensor(xte).view(-1, 784)
        scores = []
        for c in range(10):
            yc = F.one_hot(torch.full((len(X),), c), 10).float()
            mu, logvar = vae.encode(X, yc)
            rec, kl = elbo_terms(vae.decode(mu, yc), X, mu, logvar)
            scores.append(-(rec + kl))
        scores = torch.stack(scores, 1)
        gen_pred = scores.argmax(1).numpy()
        best_elbo = scores.max(1).values.numpy()
    gen_acc = float((gen_pred == yte).mean())
    print(f"[cvae] generative (Bayes-rule) classification accuracy: {gen_acc:.4f}")

    n_clf = sum(p.numel() for p in clf.parameters())
    n_vae = sum(p.numel() for p in vae.parameters())
    metrics = {
        "classifier": {
            "name": "CNN (2 conv + 2 dense)", "params": n_clf, "epochs": a.clf_epochs,
            "test_accuracy": clf_hist["test_acc"][-1], "history": clf_hist,
            "confusion_matrix": cm.tolist(), "train_seconds": round(clf_time, 1),
        },
        "cvae": {
            "name": f"Conditional VAE (latent dim {LATENT})", "params": n_vae, "epochs": a.vae_epochs,
            "history": vae_hist, "train_seconds": round(vae_time, 1),
            "generative_classification_accuracy": round(gen_acc, 4),
            "test_elbo_percentiles": {str(p): round(float(np.percentile(best_elbo, p)), 2)
                                      for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
        },
        "dataset": {"name": "MNIST", "train": int(len(xtr)), "test": int(len(xte))},
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)

    # 5 real test digits per class for the "load sample" button in the UI
    rng = np.random.default_rng(1)
    pick = np.concatenate([rng.choice(np.where(yte == c)[0], 5, replace=False) for c in range(10)])
    np.savez_compressed(os.path.join(OUT, "sample_digits.npz"),
                        images=(xte[pick] * 255).astype(np.uint8), labels=yte[pick])
    print("done. metrics ->", os.path.join(OUT, "metrics.json"))


if __name__ == "__main__":
    main()
