"""
Generative vs. Discriminative Model Lab  -  Flask backend
=========================================================

Run:   python app.py        (then open the address printed in the terminal)

Endpoints
  GET  /                      web UI
  POST /api/analyze           {image: <dataURL>}  -> classifier + generative model results
  POST /api/generate          {digit, n, temperature, seed} -> new samples from the CVAE
  GET  /api/latent_grid?digit=3                    -> decoded 2-D latent "style map"
  GET  /api/samples           real MNIST test digits for quick testing
  GET  /api/metrics           training/evaluation metrics of both models
  GET  /api/health            simple status check
"""
import base64
import io
import os
import socket
import sys
import threading
import time
import webbrowser

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

from models.inference import CVAE, Classifier, load_metrics, load_samples, preprocess, softmax

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE, "static"), static_url_path="/static")

print("Loading models ...")
CLF = Classifier()
GEN = CVAE()
METRICS = load_metrics()
SAMPLE_X, SAMPLE_Y = load_samples()
ELBO_PCT = {int(k): v for k, v in METRICS["cvae"]["test_elbo_percentiles"].items()}
print("Models ready.")


# ----------------------------------------------------------------------------- utils
def to_png(arr, scale=4):
    """2-D array in [0,1] -> base64 PNG data URL (nearest-neighbour upscaled)."""
    a = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(a, "L")
    if scale != 1:
        im = im.resize((a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def tile(imgs, cols, pad=1):
    """Stack a list of 28x28 images into one grid image."""
    n = len(imgs)
    rows = (n + cols - 1) // cols
    h, w = imgs[0].shape
    out = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad), 0.18, np.float32)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        out[pad + r * (h + pad): pad + r * (h + pad) + h, pad + c * (w + pad): pad + c * (w + pad) + w] = im
    return out


def decode_data_url(data_url):
    raw = data_url.split(",", 1)[1] if "," in data_url else data_url
    im = Image.open(io.BytesIO(base64.b64decode(raw)))
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))  # transparent -> white paper
        im = Image.alpha_composite(bg, im)
    return im.convert("L")


def typicality(elbo):
    """Where does this ELBO fall among real MNIST test digits?"""
    keys = sorted(ELBO_PCT)
    vals = [ELBO_PCT[k] for k in keys]
    pct = float(np.interp(elbo, vals, keys, left=0.0, right=100.0))
    if pct >= 10:
        label = "Typical - looks like real MNIST handwriting"
    elif pct >= 1:
        label = "Unusual - the generative model finds this digit atypical"
    else:
        label = "Very unlikely - probably not a clean digit (out-of-distribution)"
    return round(pct, 1), label


# ----------------------------------------------------------------------------- routes
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    return jsonify(status="ok", classifier=True, generator=True)


@app.route("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.route("/api/samples")
def samples():
    return jsonify(samples=[{"label": int(y), "image": to_png(x / 255.0, 2), "raw": to_png(x / 255.0, 10)}
                            for x, y in zip(SAMPLE_X, SAMPLE_Y)])


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True, silent=True) or {}
    if "image" not in data:
        return jsonify(error="No image received"), 400
    try:
        img = decode_data_url(data["image"])
    except Exception as e:  # noqa
        return jsonify(error=f"Could not read the image: {e}"), 400

    x28, info = preprocess(img)
    if info.get("empty"):
        return jsonify(error="The image looks empty - draw or upload a digit first."), 400

    # ---------------- discriminative: p(y|x) in ONE forward pass
    t0 = time.perf_counter()
    probs, logits, fmap = CLF.predict(x28)
    clf_ms = (time.perf_counter() - t0) * 1000
    pred = int(probs.argmax())

    # ---------------- generative: p(x|y) for EVERY class, then Bayes' rule
    t0 = time.perf_counter()
    elbos = []
    for c in range(10):
        e, rec, kl, _ = GEN.elbo(x28, c)
        elbos.append(e)
    elbos = np.array(elbos)
    gen_post = softmax(elbos + np.log(0.1))          # uniform prior p(y) = 1/10
    gen_ms = (time.perf_counter() - t0) * 1000
    gen_pred = int(gen_post.argmax())

    recon, mu = GEN.reconstruct(x28, pred)
    samples, _ = GEN.sample(pred, n=8, seed=int(data.get("seed", 0)) or None)
    pct, typ_label = typicality(float(elbos.max()))

    # normalise each feature map for display
    fm = []
    for m in fmap:
        rng = m.max() - m.min()
        fm.append(to_png((m - m.min()) / rng if rng > 0 else m * 0, 2))

    return jsonify(
        input28=to_png(x28, 8),
        inverted=info["inverted"],
        classifier=dict(
            prediction=pred,
            confidence=round(float(probs[pred]), 4),
            probabilities=[round(float(p), 4) for p in probs],
            logits=[round(float(v), 3) for v in logits],
            feature_maps=fm,
            inference_ms=round(clf_ms, 2),
        ),
        generative=dict(
            prediction=gen_pred,
            confidence=round(float(gen_post[gen_pred]), 4),
            probabilities=[round(float(p), 4) for p in gen_post],
            elbo=[round(float(v), 2) for v in elbos],
            reconstruction=to_png(recon, 8),
            recon_label=pred,
            latent=[round(float(v), 3) for v in mu],
            samples=[to_png(s, 3) for s in samples],
            typicality_percentile=pct,
            typicality_label=typ_label,
            inference_ms=round(gen_ms, 2),
        ),
        agree=pred == gen_pred,
    )


@app.route("/api/generate", methods=["POST"])
def generate():
    d = request.get_json(force=True, silent=True) or {}
    digit = int(d.get("digit", 0)) % 10
    n = int(min(max(int(d.get("n", 16)), 1), 64))
    temp = float(min(max(float(d.get("temperature", 1.0)), 0.1), 3.0))
    seed = d.get("seed")
    t0 = time.perf_counter()
    imgs, z = GEN.sample(digit, n=n, seed=int(seed) if seed not in (None, "") else None, temperature=temp)
    ms = (time.perf_counter() - t0) * 1000
    # the classifier "grades" every generated sample
    checks = []
    for im in imgs:
        p, _, _ = CLF.predict(im)
        checks.append(dict(pred=int(p.argmax()), conf=round(float(p.max()), 3)))
    ok = sum(c["pred"] == digit for c in checks)
    return jsonify(digit=digit, temperature=temp, images=[to_png(im, 3) for im in imgs],
                   latent=[[round(float(a), 3) for a in v] for v in z], checks=checks,
                   classifier_agreement=round(ok / n, 3), inference_ms=round(ms, 2))


@app.route("/api/latent_grid")
def latent_grid():
    digit = int(request.args.get("digit", 0)) % 10
    grid, lin = GEN.latent_grid(digit, n=10)
    return jsonify(digit=digit, image=to_png(tile(list(grid.reshape(-1, 28, 28)), 10), 2),
                   span=float(lin[-1]))


# ----------------------------------------------------------------------------- start
def free_port(preferred=5050):
    for port in [preferred] + list(range(5051, 5100)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 0)) or free_port()
    url = f"http://127.0.0.1:{port}"
    print("\n" + "=" * 60)
    print("  Generative vs. Discriminative Model Lab is running")
    print(f"  Open this in your browser:  {url}")
    print("  Press CTRL+C to stop.")
    print("=" * 60 + "\n")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False)
