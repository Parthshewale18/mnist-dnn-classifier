# MNIST DNN Classifier

A Deep Neural Network (DNN) implementation for handwritten digit classification using the MNIST dataset.

## Project Overview

This project demonstrates the complete machine learning workflow:

* Data Loading
* Data Preprocessing
* Deep Neural Network Construction
* Model Training
* Model Evaluation
* Performance Visualization

The model is built using TensorFlow and Keras, while Scikit-Learn is used for evaluation metrics and validation splitting.

---

## Tech Stack

* Python 3.12
* TensorFlow
* Keras
* Scikit-Learn
* NumPy
* Matplotlib

---

## Project Structure

```text
mnist-dnn-classifier/
│
├── models/
│   ├── best_model.keras
│   └── final_model.keras
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── main.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Neural Network Architecture

```text
Input Layer      : 784 neurons
Hidden Layer 1   : 128 neurons (ReLU)
Dropout          : 0.3
Hidden Layer 2   : 64 neurons (ReLU)
Dropout          : 0.3
Output Layer     : 10 neurons (Softmax)
```

---

## Dataset

The project uses the MNIST handwritten digit dataset:

* Training Samples: 60,000
* Testing Samples: 10,000
* Image Size: 28 × 28 grayscale

Dataset source:

```python
from tensorflow.keras.datasets import mnist
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Parthshewale18/mnist-dnn-classifier.git
cd mnist-dnn-classifier
```

Create a virtual environment:

```bash
py -3.12 -m venv .venv
```

Activate it:

```bash
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute:

```bash
python src/main.py
```

The pipeline will:

1. Load the MNIST dataset
2. Preprocess the images
3. Train the DNN
4. Evaluate performance
5. Generate visualizations
6. Save the trained model

---

## Evaluation Metrics

The project reports:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## Future Improvements

* Convolutional Neural Networks (CNN)
* Hyperparameter Tuning
* TensorBoard Integration
* Model Deployment
* Experiment Tracking

---

## Author

Parth Shewale

Learning Deep Learning and building projects with TensorFlow, Keras, and Scikit-Learn.
