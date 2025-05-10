# Machine Learning Lab Exercises

> A set of hands‑on Python exercises for a University Machine Learning course, covering both classification and clustering on the Fashion‑MNIST dataset. Implements classical algorithms (k‑NN, SVM, Naïve Bayes, neural networks) and unsupervised methods (k‑Means with various distance metrics), with evaluation by accuracy, F1‑score, and purity.  


---

## 📋 Table of Contents

1. [Overview](#overview)  
2. [Lab 1: Classification](#lab--‑classification)  
3. [Lab 2: Clustering](#lab‑2‑clustering)  
4. [Results & Comparison](#results‑comparison)  
5. [Technologies](#technologies)  
6. [Installation](#installation)  
7. [Usage](#usage)  
8. [Contributing](#contributing)  
9. [License](#license)  
10. [Contact](#contact)  

---

## Overview

These exercises give practical experience with both supervised and unsupervised Machine Learning on the Fashion‑MNIST dataset (28×28 grayscale images of 10 clothing categories).  

- **Lab 1 (Classification):** compare k‑Nearest Neighbors (Euclidean, cosine), Neural Networks (1 & 2 hidden layers), SVM (linear, RBF, cosine kernels), and Naïve Bayes; evaluate by accuracy & F1‑score.  
- **Lab 2 (Clustering):** implement k‑Means with Euclidean (L2), Manhattan (L1) and cosine distance; evaluate by purity & F‑measure.  


---

## Lab 1: Classification

- **Algorithms implemented** from scratch or via scikit‑learn:  
  - k‑NN (k=1,5,10) with Euclidean & cosine distance  
  - Feed‑forward Neural Network (1 hidden layer K=500; 2 hidden layers K₁=500, K₂=200)  
  - Support Vector Machine (linear, Gaussian/RBF, cosine kernels)  
  - Naïve Bayes (Gaussian)  

- **Evaluation metrics:** accuracy and weighted F1‑score on test set.

---

## Lab 2: Clustering

- **k‑Means clustering** (k=10) on the same dataset, using three distance measures:  
  1. Euclidean (L2)  
  2. Manhattan (L1)  
  3. Cosine (normalized to distance)  

- **Evaluation metrics:**  
  - **Purity**: fraction of correctly assigned points per cluster  
  - **F‑measure** (cluster‑level F1).

---

## Results & Comparison

### Classification (Lab 1)

| Algorithm                                        | Accuracy | F1‑Score |
|--------------------------------------------------|---------:|---------:|
| k‑NN (Euclidean, k=5)                            | 85.54%   | 85.46%   |
| k‑NN (Cosine, k=1)                               | 85.78%   | 85.60%   |
| NN (1 hidden layer, 500 neurons)                 | 86.15%   | 86.12%   |
| NN (2 hidden layers, 500–200 neurons)            | 86.89%   | 86.95%   |
| SVM (Gaussian / RBF kernel)                      | 88.29%   | 88.24%   |
| SVM (Linear kernel)                              | 84.63%   | 84.56%   |
| Naïve Bayes                                      | 67.29%   | 66.00%   |

> Best classification: **SVM with Gaussian kernel** (88.3% accuracy, 88.2% F1).

### Clustering (Lab 2)

| Distance Measure | Purity   | F‑Measure |
|------------------|---------:|----------:|
| Manhattan (L1)   | 0.6426   | 0.0121    |
| Cosine           | 0.6094   | 0.0514    |
| Euclidean (L2)   | 0.6094   | 0.0168    |

> Best clustering by purity: **Manhattan distance**; by F‑measure: **Cosine distance**.

---

## Technologies

- **Language:** Python 3.7+  
- **Libraries:** TensorFlow/Keras, scikit‑learn, NumPy, SciPy, Matplotlib  
- **Environment:** PyCharm Professional, Jupyter notebooks (optional)  
- **Tools:** virtualenv, pip  

---

## Installation

1. **Clone the repository**  
```bash
git clone https://github.com/johnprif/Machine-Learning.git
cd Machine-Learning
```
2. **Create & activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate    # Linux/macOS  
venv\Scripts\activate       # Windows  
```
3. **Install dependencies**
```bash
pip install --upgrade pip
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn
```
4. **Usage**
- **Run a Lab 1 (classification)**:
```bash
python lab1_image_classification.py
```
- **Run a Lab 2 (clustering)**:
```bash
python lab2_kmeans_clustering.py
```

## Contributing
This open‑source tool was developed for university teaching. Contributions from future students and faculty are welcome:
1. Fork this repository.
2. Create a branch: `git checkout -b feature/YourFeature`.
3. Commit your changes: `git commit -m "Add YourFeature"`.
4. Push: `git push origin feature/YourFeature`.
5. Open a Pull Request.

## License
This project is released under the **MIT License**. See [LICENSE](https://github.com/johnprif/Machine-Learning/blob/main/LICENSE) for details.

## Contact
John Priftis - [giannispriftis37@gmail.com](mailto:giannispriftis37@gmail.com)

*Designed as an open‑source teaching resource so the department can extend and reuse these exercises in future courses.*

