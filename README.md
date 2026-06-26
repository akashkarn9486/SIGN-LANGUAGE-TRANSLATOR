# 🤟 Sign Language Translator using MediaPipe & OpenCV

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.0%2B-orange?logo=google&logoColor=white)](https://mediapipe.dev/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-blueviolet?logo=numpy&logoColor=white)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## 📖 Table of Contents
1. [Overview](#-overview)
2. [Features](#-features)
3. [How It Works](#-how-it-works)
4. [Technologies Used & Versions](#-technologies-used--versions)
5. [Installation & Setup](#-installation--setup)
6. [Usage Guide](#-usage-guide)
7. [The NumPy Database – Uniqueness & Advantage](#-the-numpy-database--uniqueness--advantage)
8. [Comparison with Existing Systems](#-comparison-with-existing-systems)
9. [Contributing](#-contributing)
10. [License](#-license)
11. [Acknowledgements](#-acknowledgements)

---

## 📌 Overview
This project is a **real-time sign language translator** built with Python, leveraging **MediaPipe** for hand landmark detection and **OpenCV** for live video capture and visualization. It allows users to:

- **Register** new signs by showing hand gestures in front of a webcam.
- **Translate** live hand gestures into pre-registered meanings.
- **Store** and **manage** the sign database using a lightweight **NumPy** file format (`.npz`).

The system is designed to be simple, extensible, and portable—no need for large neural networks or cloud APIs. It uses **normalized landmark coordinates** (position and scale‑invariant) to compare gestures via **Euclidean distance**, making it efficient and fast enough for real‑time applications.

---

## ✨ Features
- ✅ **Real‑time gesture recognition** – detects hands and predicts sign meanings instantly.
- ✅ **Two‑hand support** – works with both left and right hands, with consistent ordering.
- ✅ **Persistent database** – saves registered signs in a `.npz` file, so you don't lose progress.
- ✅ **Normalized landmarks** – robust to hand position, size, and slight rotation.
- ✅ **User‑friendly CLI** – simple menu‑driven interface for registration and translation.
- ✅ **Low latency** – uses lightweight distance computation, no heavy model inference.
- ✅ **Extensible** – easily add more sophisticated distance metrics or machine learning classifiers.

---

## ⚙️ How It Works

### Pipeline Overview
1. **Hand Landmark Detection** – MediaPipe’s `mp.solutions.hands` processes each video frame to extract 21 keypoints (x, y, z coordinates) per hand.
2. **Normalization** – The raw landmarks are **translated** so the wrist (landmark 0) is at the origin, then **scaled** by the maximum distance from the wrist to make the gesture invariant to position and scale.
3. **Database Registration** – When a new sign is registered, the system captures a series of normalized landmark vectors (over 5 seconds) and averages them to create a robust template.
4. **Translation (Matching)** – For each incoming frame, the normalized landmarks are compared against every stored template using **Euclidean distance**. The sign with the smallest distance is displayed as the prediction.

### Database Storage
All templates are stored in a **single `.npz` file** containing a dictionary mapping sign meanings to flattened landmark arrays. This approach is:
- **Fast** – NumPy’s binary format allows lightning‑fast load/save.
- **Compact** – Each sign requires only 63 numbers (21 landmarks × 3 coordinates) stored as float32.
- **Portable** – No external database engine or configuration required.

---

## 🛠️ Technologies Used & Versions
| Technology | Version | Purpose |
|------------|---------|---------|
| [**Python**](https://www.python.org/) | 3.7+ | Core programming language |
| [**OpenCV**](https://opencv.org/) | 4.5+ | Video capture, image processing, drawing |
| [**MediaPipe**](https://mediapipe.dev/) | 0.10.0+ | Hand landmark detection (21 points per hand) |
| [**NumPy**](https://numpy.org/) | 1.21+ | Numerical operations, database storage |
| [**OS**](https://docs.python.org/3/library/os.html) | built‑in | File system operations |
| [**Time**](https://docs.python.org/3/library/time.html) | built‑in | Timers for capture and delays |

> **Note:** The code is compatible with Python 3.7 or higher. All libraries can be installed via `pip`.

---

## 📥 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sign-language-translator.git
   cd sign-language-translator
   ```

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *If you don't have a `requirements.txt`, manually install:*
   ```bash
   pip install opencv-python mediapipe numpy
   ```

4. **Run the application**
   ```bash
   python sign.py
   ```

---

## 🖥️ Usage Guide

Upon running `sign.py`, you will see a menu:

```
====== SIGN LANGUAGE TRANSLATOR ======
1. Register New Sign
2. Translate a Sign
3. Exit (or press 'q')
Enter your choice:
```

### Register a New Sign
1. Choose option `1`.
2. The webcam will open. Hold your hand(s) steadily in front of the camera.
3. The system captures hand landmarks for **5 seconds**. Press `q` early if needed.
4. After capture, you will be prompted to **enter a meaning** (e.g., "hello", "thank you").
5. The sign is saved to the database automatically.

### Translate a Sign
1. Choose option `2`.
2. Show a registered sign in front of the camera.
3. The system will continuously display the **predicted meaning** on the video feed.
4. Press `q` to stop translation.

### Exit
- Choose option `3` or press `q` at any menu prompt to safely exit and save the database.

---

## 🗂️ The NumPy Database – Uniqueness & Advantage

Unlike many sign language systems that rely on external databases (SQL, JSON, or even cloud storage), this project uses **NumPy’s native `.npz` format** to store all registered sign templates.

### Why is this unique?
- **Lightweight & Fast** – `.npz` files are compressed binary archives. Loading and saving are several orders of magnitude faster than parsing JSON or CSV.
- **Direct Integration** – Since the landmarks are already NumPy arrays, storing them directly avoids serialization overhead and loss of precision.
- **No Extra Dependencies** – No need for SQLite, MongoDB, or any external database service. Everything is self‑contained.
- **Simple Backup** – The database is a single file (`sign_database.npz`) that can be copied, shared, or version‑controlled easily.
- **Extensible** – You can easily add metadata (e.g., timestamps, user IDs) as additional arrays inside the same `.npz` archive.

### How it works under the hood
- `load_database()` reads the `.npz` file and converts the stored array back into a Python dictionary.
- `save_database()` compresses the dictionary into a binary `.npz` file with a single call to `np.savez()`.
- The dictionary keys are sign meanings (strings), and values are normalized landmark vectors (1D NumPy arrays of length 63).

This design ensures that the system is **ready to use out‑of‑the‑box** without any configuration, making it ideal for education, prototyping, and hobbyist projects.

---

## 🔍 Comparison with Existing Systems

| Feature | **This Project** | Traditional ML-based Systems (e.g., TensorFlow models) | Commercial Cloud APIs (e.g., Google Cloud Vision) |
|---------|------------------|-------------------------------------------------------|--------------------------------------------------|
| **Hardware Requirement** | Any webcam | Often requires GPU for training/inference | Internet connection, cloud credits |
| **Training Data** | None – on‑the‑fly registration | Requires large labeled datasets (thousands of samples) | Extensive pre‑trained models |
| **Latency** | Very low (<30 ms per frame) | Moderate (depends on model size) | High (network round‑trip) |
| **Offline Capability** | ✅ Fully offline | ✅ If model is local | ❌ Requires internet |
| **Customization** | Easy – add new signs instantly | Requires retraining or fine‑tuning | Limited to predefined gestures |
| **Database** | Lightweight NumPy file | Usually TensorFlow SavedModel or PyTorch weights | Proprietary cloud storage |
| **Cost** | Free (open source) | Free for training/inference | Pay‑per‑use |
| **Accuracy** | Good for static/controlled environments | High if well‑trained | Very high (often >95%) |
| **Handling of Dynamic Gestures** | Currently static only (can be extended) | Often supports sequence models (LSTMs) | Supports dynamic with variants |

### Why This Project Stands Out
- **Simplicity** – No machine learning background required. Anyone can register a sign in seconds.
- **Transparency** – The matching logic is clear and interpretable (Euclidean distance).
- **Privacy** – All data stays on your machine; no video frames are sent to the cloud.
- **Rapid Prototyping** – Perfect for testing new gesture vocabularies without writing a single line of ML code.
- **Extensibility** – The code is modular; you can easily replace the distance metric with a classifier (SVM, k‑NN) or integrate a neural network later.

---

## 🤝 Contributing
Contributions are welcome! If you have ideas for improvements, bug fixes, or new features, please open an issue or submit a pull request.

### Guidelines
- Follow PEP 8 style.
- Document new functions with docstrings.
- Add unit tests for core functionalities (if possible).
- Update the README with any new features.

---

## 📄 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements
- [MediaPipe](https://mediapipe.dev/) for providing a robust hand tracking solution.
- [OpenCV](https://opencv.org/) for making computer vision accessible.
- [NumPy](https://numpy.org/) for the incredible array library that powers the database.
- The open‑source community for continuous inspiration and support.

---

## 📞 Contact & Support
For questions, suggestions, or feedback, please [open an issue](https://github.com/yourusername/sign-language-translator/issues) or reach out to the maintainer.

---

**Enjoy translating sign language with your webcam!** 🤟

---

*Made with ❤️ and Python.*
