# Generative vs. Discriminative Model Lab

A web application where a user **draws or uploads a handwritten digit** and sees how a
**classifier (discriminative model)** and a **generative model** each handle it.

| Requirement | How it is met |
|---|---|
| **Frontend:** digit drawing canvas or image upload | 280×280 drawing canvas (mouse + touch, undo, brush size), image upload (photos of paper are auto-inverted), and 50 real MNIST test digits to click |
| **Backend:** trained MNIST classifier + small generative model | Flask server loads a trained **CNN classifier** (99.1 % test accuracy) and a trained **Conditional VAE** (2-D latent space) |
| **Output:** predicted digit, confidence, generated samples | Predicted digit, confidence %, probability bar chart, 8 newly generated samples of the predicted digit, reconstruction, latent style map, plus a Generator Studio for any digit |
| **Should test:** classification vs generation, model inference, discriminative vs generative | Side-by-side panels, inference-time display, Bayes-rule classification by the generative model, typicality (out-of-distribution) meter, Model Report tab, and an automated test suite (`tests/test_lab.py`) |

---

## ▶ How to run (friend's laptop)

**Only requirement:** Python 3.9 or newer — <https://www.python.org/downloads/>
(on Windows, tick **"Add Python to PATH"** while installing). Internet is needed only the first
time, to install three small packages (Flask, NumPy, Pillow ≈ 30 MB). **PyTorch is NOT needed.**

1. Unzip the folder (right-click → *Extract All* — do not run it from inside the zip).
2. Start it:
   - **Windows:** double-click **`RUN_WINDOWS.bat`**
   - **macOS:** right-click **`RUN_MAC.command`** → *Open* → *Open*
     (or in Terminal: `bash RUN_MAC.command`)
   - **Linux:** `bash RUN_MAC.command`
3. The browser opens automatically at `http://127.0.0.1:5050`. Keep the black terminal window open while using the app.

**Manual method** (any OS):
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Run the tests: `python tests/test_lab.py` → should print **10/10 tests passed**.

---

## Aim
To build an application that compares a **discriminative model** and a **generative model**
on the same handwritten-digit input, and to study the difference between classification and generation.

## Theory
| | Discriminative | Generative |
|---|---|---|
| Models | `p(y \| x)` — the decision boundary | `p(x \| y)·p(y)` — how the data itself is distributed |
| Question | "Which digit is this?" | "What does a digit look like? How likely is this image?" |
| Can create new data | No | Yes |
| Can classify | Yes, directly | Yes, using Bayes' rule `p(y\|x) ∝ p(x\|y)·p(y)` |
| Examples | Logistic regression, SVM, CNN | Naive Bayes, VAE, GAN, Diffusion models, GPT |

**Conditional VAE.** An encoder maps (image, label) to a Gaussian over a latent vector `z`;
a decoder maps (`z`, label) back to an image. It is trained to maximise the **ELBO**
(reconstruction log-likelihood − KL divergence), a lower bound on `log p(x | y)`.
To generate, sample `z ~ N(0, I)`, choose a label and decode.

## Models

| | Classifier (discriminative) | CVAE (generative) |
|---|---|---|
| Architecture | Conv(16,3×3)–ReLU–MaxPool → Conv(32,3×3)–ReLU–MaxPool → Dense(128)–Dropout → Dense(10) | Encoder 794→512→256→(μ, log σ²) ∈ ℝ² · Decoder (2+10)→256→512→784 |
| Parameters | 206,922 | 1,076,500 |
| Training | 6 epochs, Adam, cross-entropy, light augmentation (±12° rotation, shift, scale) | 40 epochs, Adam, −ELBO (Bernoulli reconstruction + KL) |
| Result on 10,000 test images | **99.1 % accuracy** | **94.1 %** accuracy when used as a classifier via Bayes' rule · median ELBO −125 nats |

Both models were trained with PyTorch on the 60,000 MNIST training images
(`training/train_models.py`) and exported to NumPy `.npz` files. The app runs inference in
pure NumPy (verified to match PyTorch to within 6×10⁻⁶), which is why PyTorch is not needed to run it.

## Procedure / how the app works
1. **Input** — user draws on the canvas, uploads an image, or picks a real MNIST digit.
2. **Preprocessing** (`models/inference.py → preprocess`) — convert to greyscale, invert if the
   background is light, remove background, crop to the digit, scale to fit 20×20, place in 28×28
   and centre by centre of mass (the exact MNIST recipe).
3. **Discriminative inference** — one forward pass of the CNN → softmax → predicted digit + confidence.
   Layer-1 feature maps are shown.
4. **Generative inference** — the CVAE computes the ELBO of the image under each of the 10 labels,
   applies Bayes' rule with a uniform prior → its own prediction. The best ELBO is compared with the
   distribution on real test digits to give a **typicality percentile**.
5. **Generation** — the CVAE reconstructs the user's digit, samples 8 new digits of the predicted
   class, and places the digit on the 2-D latent **style map**.
6. **Generator Studio** — generate 4–48 samples of any digit with a temperature control; every sample is
   graded by the classifier.
7. **Model Report** — training curves, confusion matrix and head-to-head comparison.

## Output / observations
Screenshots are in `docs/screenshots/`.

| Experiment | Observation |
|---|---|
| Clean "7" drawn on canvas | Both models predict 7. Classifier ≈ 100 % confident (1 forward pass, ~1 ms); generative model agrees (10 passes, ~3 ms). Typicality ≈ 50th percentile. |
| Random scribble (not a digit) | Classifier still outputs a digit with ~60 % confidence. Generative model's typicality drops to the 0th percentile → flags it as out-of-distribution. |
| Photo of a dark "4" on light paper | Auto-inverted and centred; both models predict 4. |
| Generator Studio, temperature 1.0 | ~100 % of generated digits recognised by the classifier as the requested class. |
| Generator Studio, temperature 2.5 | Samples become distorted; classifier agreement falls (≈ 90 % or less). |
| Latent style map | Moving across the 2-D latent grid changes slant and stroke thickness smoothly while the label stays fixed. |
| Accuracy on 10,000 test images | Discriminative 99.1 % vs generative-via-Bayes 94.1 %. |

## Result
The discriminative CNN is the better **classifier** (99.1 % vs 94.1 %) and is faster at inference,
because it learns only the decision boundary `p(y | x)`. The generative CVAE learns `p(x | y)`,
which lets it **generate new digits, reconstruct inputs, and detect unusual / out-of-distribution
inputs** — things the classifier cannot do — while still being able to classify through Bayes' rule.

---

## Project structure
```
GenVsDisc_Lab/
├── app.py                  Flask backend (API + serves the UI)
├── models/
│   ├── inference.py        NumPy inference: preprocessing, CNN, CVAE
│   ├── classifier.npz      trained CNN weights
│   ├── cvae.npz            trained CVAE weights
│   ├── metrics.json        accuracy, loss curves, confusion matrix, ELBO stats
│   └── sample_digits.npz   50 real MNIST test digits for the UI
├── static/                 frontend (index.html, style.css, app.js) — no internet/CDN needed
├── training/train_models.py   PyTorch training + export script (optional)
├── tests/test_lab.py       automated tests (10 tests)
├── docs/screenshots/       output screenshots
├── RUN_WINDOWS.bat         one-click launcher (Windows)
├── RUN_MAC.command         one-click launcher (macOS / Linux)
└── requirements.txt        flask, numpy, pillow
```

## API
| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/analyze` | `{image: dataURL}` → both models' predictions, confidences, ELBOs, reconstruction, samples |
| POST | `/api/generate` | `{digit, n, temperature, seed}` → new samples + classifier check |
| GET | `/api/latent_grid?digit=d` | decoded 10×10 latent style map |
| GET | `/api/metrics` | training / evaluation metrics |
| GET | `/api/samples` | real MNIST test digits |
| GET | `/api/health` | status |

## Retraining (optional)
```bash
pip install torch numpy
python training/train_models.py                    # downloads MNIST via torchvision
python training/train_models.py --data path/to/mnist_files   # or use local idx files
```
Training took ≈ 1 min (classifier) + ≈ 4 min (CVAE) on a laptop CPU.

## Troubleshooting
| Problem | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python and tick **Add Python to PATH**. |
| macOS: "cannot be opened because it is from an unidentified developer" | Right-click `RUN_MAC.command` → *Open*, or run `bash RUN_MAC.command` in Terminal. |
| Browser didn't open | Open the address printed in the terminal (e.g. `http://127.0.0.1:5050`). |
| Port in use | The app automatically picks the next free port (5051, 5052 …) and prints it. |
| `pip install` fails | Check the internet connection (needed only on the first run). |
