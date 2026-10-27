"""
Automated tests for the lab.  Run from the project folder:

    python tests/test_lab.py          (no extra packages needed)
or  python -m pytest tests -v        (if pytest is installed)

Covers the three things the lab "should test":
  1. classification vs generation
  2. model inference (shapes, probabilities, speed, API)
  3. discriminative vs generative behaviour
"""
import base64
import io
import os
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from models.inference import CVAE, Classifier, load_samples, preprocess, softmax  # noqa: E402

CLF, GEN = Classifier(), CVAE()
X, Y = load_samples()
X = X.astype(np.float32) / 255.0


# ------------------------------------------------------------- 1. classification
def test_classifier_output_is_probability_distribution():
    probs, logits, fmap = CLF.predict(X[0])
    assert probs.shape == (10,) and abs(probs.sum() - 1) < 1e-5 and (probs >= 0).all()
    assert fmap.shape == (16, 28, 28)


def test_classifier_accuracy_on_bundled_test_digits():
    acc = np.mean([CLF.predict(x)[0].argmax() == y for x, y in zip(X, Y)])
    print(f"    classifier accuracy on 50 held-out digits: {acc:.0%}")
    assert acc >= 0.94


# ------------------------------------------------------------- 2. generation
def test_generator_produces_valid_images():
    imgs, z = GEN.sample(7, n=6, seed=0)
    assert imgs.shape == (6, 28, 28) and z.shape == (6, 2)
    assert imgs.min() >= 0 and imgs.max() <= 1


def test_generated_digits_are_recognised_by_classifier():
    ok = 0
    for d in range(10):
        imgs, _ = GEN.sample(d, n=10, seed=d)
        ok += sum(CLF.predict(im)[0].argmax() == d for im in imgs)
    print(f"    classifier recognises {ok}/100 generated digits as the requested class")
    assert ok >= 80


def test_same_seed_same_samples_different_seed_different_samples():
    a, _ = GEN.sample(3, n=4, seed=1)
    b, _ = GEN.sample(3, n=4, seed=1)
    c, _ = GEN.sample(3, n=4, seed=2)
    assert np.allclose(a, b) and not np.allclose(a, c)


# ------------------------------------------------------------- 3. discriminative vs generative
def test_generative_bayes_rule_classification():
    correct = 0
    for x, y in zip(X, Y):
        el = np.array([GEN.elbo(x, c)[0] for c in range(10)])
        correct += softmax(el).argmax() == y
    acc = correct / len(Y)
    print(f"    generative (Bayes-rule) accuracy on 50 digits: {acc:.0%}")
    assert acc >= 0.8


def test_noise_is_less_likely_than_real_digit_under_generative_model():
    rng = np.random.default_rng(0)
    noise = (rng.random((28, 28)) > 0.8).astype(np.float32)
    real = max(GEN.elbo(X[0], c)[0] for c in range(10))
    fake = max(GEN.elbo(noise, c)[0] for c in range(10))
    conf = CLF.predict(noise)[0].max()
    print(f"    ELBO real digit {real:.1f} vs noise {fake:.1f}; classifier confidence on noise {conf:.0%}")
    assert real > fake + 50


def test_inference_speed():
    t = time.perf_counter(); CLF.predict(X[0]); t_c = time.perf_counter() - t
    t = time.perf_counter(); [GEN.elbo(X[0], c) for c in range(10)]; t_g = time.perf_counter() - t
    print(f"    classifier {t_c * 1000:.1f} ms | generative scoring over 10 classes {t_g * 1000:.1f} ms")
    assert t_c < 1.0 and t_g < 2.0


# ------------------------------------------------------------- preprocessing + API
def test_preprocess_handles_dark_ink_on_white_paper():
    im = Image.new("L", (200, 300), 255)
    from PIL import ImageDraw
    ImageDraw.Draw(im).line([(100, 40), (100, 260)], fill=0, width=25)   # a "1"
    x, info = preprocess(im)
    assert info["inverted"] and x.shape == (28, 28) and x.max() > 0.5
    assert CLF.predict(x)[0].argmax() == 1


def test_flask_api():
    import app as server
    c = server.app.test_client()
    assert c.get("/api/health").json["status"] == "ok"
    buf = io.BytesIO(); Image.fromarray((X[0] * 255).astype(np.uint8)).resize((280, 280)).save(buf, "PNG")
    r = c.post("/api/analyze", json={"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()})
    assert r.status_code == 200
    j = r.json
    assert j["classifier"]["prediction"] == int(Y[0])
    assert len(j["generative"]["samples"]) == 8 and len(j["generative"]["probabilities"]) == 10
    g = c.post("/api/generate", json={"digit": 5, "n": 12}).json
    assert len(g["images"]) == 12
    assert c.get("/api/latent_grid?digit=2").status_code == 200
    assert c.post("/api/analyze", json={}).status_code == 400


if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    passed = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS  {name}"); passed += 1
        except Exception as e:  # noqa
            print(f"FAIL  {name}: {e!r}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
