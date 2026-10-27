# 🧠 GenVsDisc Lab — Generative vs. Discriminative Model Laboratory

### An Interactive Machine Learning Laboratory for Digit Classification and Generation

<p align="center">
  <strong>Explore the difference between discriminative and generative AI through an interactive web application.</strong>
</p>

---

## 🚀 Live Demo

### **[🌐 https://gen-vs-disc-lab.vercel.app/)**



---

## 📌 About the Project

**GenVsDisc Lab** is an interactive machine learning web application designed to demonstrate the fundamental differences between discriminative and generative models.

The application allows users to experiment with handwritten digit recognition, explore classification predictions, and generate digit-like images through a browser-based interface.

Rather than simply presenting theoretical concepts, GenVsDisc Lab provides a practical environment for exploring how different machine learning approaches process and generate data.

## ✨ Key Features

- **✍️ Interactive Digit Drawing:** Draw handwritten digits directly on the canvas.
- **🖼️ Image Upload:** Upload digit images for model inference.
- **🔍 Digit Classification:** Predict digit classes using a CNN-based classifier.
- **📊 Prediction Probabilities:** Explore model predictions and their confidence scores.
- **🎨 Digit Generation:** Generate digit-like images using a Conditional Variational Autoencoder (CVAE).
- **📈 Model Metrics:** Explore available evaluation metrics and model reports.
- **🧪 Interactive Experimentation:** Compare classification and generation within a unified interface.
- **💻 Browser-Based Interface:** Access the application without installing the project locally when using the deployed version.

## 🧠 Models Used

### 1. Discriminative Model — CNN Classifier

A Convolutional Neural Network (CNN) classifier predicts the class of an input image.

**Primary objective:** Learn the relationship between input images and their corresponding labels.

**Example:** Given an image of a handwritten digit, predict whether it represents `0`, `1`, `2`, and so on.

### 2. Generative Model — Conditional Variational Autoencoder

A Conditional Variational Autoencoder (CVAE) learns a latent representation of the input data and generates new samples conditioned on a selected class.

**Primary objective:** Learn useful representations of the data and generate new digit-like images.

**Example:** Generate a new image conditioned on the digit `7`.

### Generative vs. Discriminative Learning

| Aspect | Discriminative Model | Generative Model |
|---|---|---|
| Main objective | Predict the target class | Learn data representations and generate samples |
| Input | Image or digit sample | Selected class or conditioning input |
| Output | Predicted class and probabilities | Generated digit-like image |
| Model used | CNN classifier | Conditional VAE |
| Application | Image classification | Image generation |

*Actual capabilities depend on the trained models and the implementation included in this repository.*

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and machine learning logic |
| Flask | Web application and API endpoints |
| NumPy | Numerical computation and model inference |
| Pillow | Image processing |
| HTML5 | Application structure |
| CSS3 | User interface styling |
| JavaScript | Interactive features and API communication |
| Vercel | Cloud deployment |
| GitHub | Source code management |

## 📂 Project Structure

```text
GenVsDisc-Lab/
│
├── app.py                 # Flask application and API routes
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── RUN_WINDOWS.bat        # Windows launch script
├── RUN_MAC.command        # macOS launch script
│
├── models/                # Model inference and saved artifacts
├── static/                # Frontend HTML, CSS, and JavaScript
├── training/              # Model training scripts
├── tests/                 # Project tests
└── docs/                  # Additional documentation
```

*The structure above is illustrative. Refer to the repository for the actual files and directories.*

## ⚙️ Run Locally

### Prerequisites

- Python 3.9 or a compatible version supported by the project dependencies.
- pip package manager.
- Git (optional).

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/GenVsDisc-Lab.git
```

Navigate into the project directory:

```bash
cd GenVsDisc-Lab
```

Replace `YOUR_GITHUB_USERNAME` with your GitHub username.

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate the Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Start the Application

```bash
python app.py
```

### Step 6: Open in Your Browser

Visit:

```text
http://127.0.0.1:5050
```

Keep the terminal open while using the application.

## ☁️ Deployment

The project can be deployed using Vercel's Python/Flask support.

1. Push the project to a GitHub repository.
2. Import the repository into Vercel.
3. Configure the project according to the Vercel Flask deployment requirements.
4. Deploy the application.
5. Test the homepage, API endpoints, classification, and generation features.
6. Update the Live Demo link at the top of this README with the actual deployment URL.

**Deployment URL:** `YOUR_VERCEL_DEPLOYMENT_URL`

**Deployment documentation:** [Vercel Flask Documentation](https://vercel.com/docs/frameworks/backend/flask)

## 🎯 Learning Objectives

This project demonstrates:

- The difference between generative and discriminative learning.
- How CNNs can be used for image classification.
- How conditional VAEs can generate class-conditioned samples.
- How model inference can be integrated into a Flask application.
- How frontend components communicate with backend APIs.
- How machine learning applications can be prepared for cloud deployment.

## 🔮 Future Improvements

Potential enhancements include:

- Additional classification and generation models.
- More detailed model evaluation and comparison.
- Improved inference visualization.
- Expanded testing and error handling.
- Additional datasets and experimentation options.

## 👨‍💻 Author

**Your Name**

- GitHub: [YOUR_GITHUB_USERNAME]([https://github.com/YOUR_GITHUB_USERNAME](https://github.com/piyushT3003/GenVsDisc-Lab))
- Project Repository: `https://github.com/YOUR_GITHUB_USERNAME/GenVsDisc-Lab`
- Live Demo: [GenVsDisc Lab](YOUR_VERCEL_DEPLOYMENT_URL)

Replace the placeholders with your details before publishing.

## 📄 License

Add a license if you intend to distribute the project for reuse. Until a license is specified, the repository remains subject to applicable copyright law.

---

<p align="center">
  <strong>GenVsDisc Lab</strong><br>
  Making Generative and Discriminative Machine Learning Interactive.
</p>
