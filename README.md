# Machine Learning Lab Exercises

> A collection of hands‑on Python laboratory exercises for a university Machine Learning course, implementing image classification with Google’s TensorFlow and classical algorithms (e.g. k‑Nearest Neighbor) to compare performance and accuracy.

---

## 📋 Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Labs & Reports](#labs--reports)  
4. [Technologies](#technologies)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [Contributing](#contributing)  
8. [License](#license)  
9. [Contact](#contact)  

---

## Overview

These **Machine Learning Lab Exercises** are part of an academic curriculum in Python, designed to give students practical experience with both deep learning and classical machine‑learning methods. Students build image‑classification models using **TensorFlow**, implement algorithms such as k‑Nearest Neighbor with Euclidean distance, and then compare accuracy and performance across approaches.

---

## 🔥 Features

- **Deep Learning**: Image classification pipelines built with TensorFlow’s high‑level Keras API.  
- **Classical ML**: Implementation of k‑NN, decision trees, and logistic regression from scratch and via scikit‑learn.  
- **Performance Comparison**: Automated evaluation scripts compute accuracy, precision, recall, and runtime for each model.  
- **Data Preprocessing**: Demonstrations of normalization, train/test splits, and cross‑validation.  
- **Reporting**: Each lab includes a PDF report detailing methodology, results, and insights.  

---

## Labs & Reports

| Lab         | Description                                               | Report (PDF)            |
|-------------|-----------------------------------------------------------|-------------------------|
| Lab 1       | Image loader & preprocessing; basic TensorFlow classifier | `Lab1_report.pdf`       |
| Lab 2       | k‑Nearest Neighbor vs. Decision Tree comparison           | `Lab2_report.pdf`       |
| Lab 3       | Convolutional Neural Network for multi‑class recognition  | `Lab3_report.pdf`       |
| Lab 4       | Transfer learning with pre‑trained models                 | `Lab4_report.pdf`       |

Details for each lab’s implementation and analysis are in the corresponding PDF files in the repository.  

---

## Technologies

| Category               | Tools & Libraries                                    |
|------------------------|------------------------------------------------------|
| Language               | Python 3.9+ :contentReference[oaicite:5]{index=5}                        |
| Deep Learning          | TensorFlow 2.x, Keras             |
| Classical ML           | scikit‑learn                     |
| Data Manipulation      | NumPy, pandas                                        |
| Visualization          | Matplotlib, Seaborn                                  |
| Reporting              | LaTeX / LibreOffice for PDF reports                  |
| Environment Management | virtualenv, pip                  |

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
- **Run a lab script** (e.g. Lab1):
```bash
python lab1_image_classification.py
```
- **View results & plots** in the generated `outputs/` folder.
- **Open the PDF report** for detailed methodology and performance charts.

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

