"""
Pure-NumPy inference for both models (no PyTorch needed to run the app).

    Classifier  (discriminative):  x  ->  p(y | x)
    CVAE        (generative)    :  (z, y) -> x      i.e. samples from p(x | y)

The weights in classifier.npz / cvae.npz were trained with PyTorch by
training/train_models.py and exported as plain arrays.
"""
import json
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- helpers
def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def conv2d(x, w, b):
    """x: (C,H,W)  w: (O,C,3,3)  padding=1, stride=1 -> (O,H,W)  via im2col."""
    c, h, wd = x.shape
    o, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    cols = np.empty((c, kh, kw, h, wd), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            cols[:, i, j] = xp[:, i:i + h, j:j + wd]
    out = w.reshape(o, -1) @ cols.reshape(c * kh * kw, h * wd)
    return out.reshape(o, h, wd) + b[:, None, None]


def maxpool2(x):
    c, h, w = x.shape
    return x.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


# ----------------------------------------------------------------------------- preprocessing
def preprocess(img: Image.Image):
    """
    Turn any drawn/uploaded image into a 28x28 MNIST-style array in [0,1]:
    white digit on black, digit fitted in a 20x20 box, centred by centre of mass.
    Returns (array28, info_dict).
    """
    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    info = {"inverted": False}

    # MNIST is white-on-black. A photo/scan of ink on paper is the opposite -> invert.
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    if border.mean() > 0.5:
        g = 1.0 - g
        info["inverted"] = True

    # remove background haze and stretch contrast
    bg = np.percentile(g, 50)
    g = np.clip(g - bg, 0, None)
    if g.max() < 1e-3:
        info["empty"] = True
        return np.zeros((28, 28), np.float32), info
    g = np.clip(g / np.percentile(g[g > 0], 99), 0, 1)
    g[g < 0.1] = 0

    ys, xs = np.nonzero(g > 0.1)
    if len(ys) == 0:
        info["empty"] = True
        return np.zeros((28, 28), np.float32), info
    g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # fit longest side to 20 px (keep aspect ratio)
    h, w = g.shape
    s = 20.0 / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    small = Image.fromarray((g * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)
    small = np.asarray(small, dtype=np.float32) / 255.0

    canvas = np.zeros((28, 28), np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas[top:top + nh, left:left + nw] = small

    # shift so the centre of mass is at the image centre (as in original MNIST)
    tot = canvas.sum()
    if tot > 0:
        cy = (canvas.sum(1) * np.arange(28)).sum() / tot
        cx = (canvas.sum(0) * np.arange(28)).sum() / tot
        dy, dx = int(round(13.5 - cy)), int(round(13.5 - cx))
        canvas = np.roll(np.roll(canvas, dy, 0), dx, 1)
    info["empty"] = False
    return np.clip(canvas, 0, 1), info


# ----------------------------------------------------------------------------- models
class Classifier:
    """Discriminative model: small CNN that directly outputs p(y | x)."""

    def __init__(self, path=os.path.join(HERE, "classifier.npz")):
        w = np.load(path)
        self.p = {k: w[k] for k in w.files}

    def forward(self, x28):
        p = self.p
        x = x28[None].astype(np.float32)
        a1 = relu(conv2d(x, p["conv1_weight"], p["conv1_bias"]))      # 16x28x28
        h1 = maxpool2(a1)                                               # 16x14x14
        a2 = relu(conv2d(h1, p["conv2_weight"], p["conv2_bias"]))      # 32x14x14
        h2 = maxpool2(a2)                                               # 32x7x7
        f = relu(p["fc1_weight"] @ h2.reshape(-1) + p["fc1_bias"])     # 128
        logits = p["fc2_weight"] @ f + p["fc2_bias"]                    # 10
        return logits, a1

    def predict(self, x28):
        logits, a1 = self.forward(x28)
        probs = softmax(logits)
        return probs, logits, a1


class CVAE:
    """Generative model: Conditional Variational Autoencoder, learns p(x | y)."""

    def __init__(self, path=os.path.join(HERE, "cvae.npz")):
        w = np.load(path)
        self.p = {k: w[k] for k in w.files}
        self.latent = self.p["mu_weight"].shape[0]

    @staticmethod
    def onehot(y):
        v = np.zeros(10, np.float32)
        v[y] = 1
        return v

    def encode(self, x784, y):
        p = self.p
        h = relu(p["enc1_weight"] @ np.concatenate([x784, self.onehot(y)]) + p["enc1_bias"])
        h = relu(p["enc2_weight"] @ h + p["enc2_bias"])
        return p["mu_weight"] @ h + p["mu_bias"], p["logvar_weight"] @ h + p["logvar_bias"]

    def decode_logits(self, z, y):
        """z: (N, latent)  -> logits (N, 784). Batched."""
        p = self.p
        z = np.atleast_2d(z).astype(np.float32)
        yy = np.tile(self.onehot(y), (len(z), 1))
        h = relu(np.concatenate([z, yy], 1) @ p["dec1_weight"].T + p["dec1_bias"])
        h = relu(h @ p["dec2_weight"].T + p["dec2_bias"])
        return h @ p["dec3_weight"].T + p["dec3_bias"]

    def decode(self, z, y):
        return sigmoid(self.decode_logits(z, y)).reshape(-1, 28, 28)

    def sample(self, y, n=8, seed=None, temperature=1.0):
        """Generate n NEW digits of class y: z ~ N(0, I), x = decoder(z, y)."""
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, self.latent)).astype(np.float32) * temperature
        return self.decode(z, y), z

    def elbo(self, x28, y):
        """Evidence lower bound  ≈ log p(x | y)  (deterministic, using z = mu)."""
        x = x28.reshape(-1).astype(np.float32)
        mu, logvar = self.encode(x, y)
        logits = self.decode_logits(mu, y)[0]
        # Bernoulli log-likelihood, numerically stable
        rec = (x * logits - np.logaddexp(0, logits)).sum()
        kl = -0.5 * np.sum(1 + logvar - mu ** 2 - np.exp(logvar))
        return float(rec - kl), float(rec), float(kl), mu

    def reconstruct(self, x28, y):
        mu, _ = self.encode(x28.reshape(-1).astype(np.float32), y)
        return self.decode(mu[None], y)[0], mu

    def latent_grid(self, y, n=10, span=2.5):
        """Decode an n x n grid over the 2-D latent space (the 'style map')."""
        lin = np.linspace(-span, span, n)
        zz = np.array([[a, b] for b in lin[::-1] for a in lin], np.float32)
        return self.decode(zz, y).reshape(n, n, 28, 28), lin


def load_metrics():
    with open(os.path.join(HERE, "metrics.json")) as f:
        return json.load(f)


def load_samples():
    d = np.load(os.path.join(HERE, "sample_digits.npz"))
    return d["images"], d["labels"]
