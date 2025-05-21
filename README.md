# Test-time Adaptation for Graph-based Molecular Solubility Prediction

This repository investigates applying graph neural networks (GNNs) to predict molecular solubility, using a self-supervised learning (SSL) approach with a test-time adaptation (TTA) step.

---

## Overview

In this project, we:
- Load and preprocess molecular data from CSV files.  
- Train a GNN model with both supervised and self-supervised tasks.  
- Optionally adapt the encoder part of the model with test-time adaptation for better performance on unseen data.  

The key scripts can be found in the Jupyter Notebook `Molecular-Test-TIme-Adaptation.ipynb`.

This project is inspired by the works of x et al. and y et al.

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/p0017/Molecular-Test-Time-Adaptation.git
cd Molecular-Test-Time-Adaptation
```

### 2. Install required packages
Set up a conda environment. This can take a few minutes.
```bash
conda env create -f environment.yml
conda activate molecular
```

### 4. Train the Model
Open the notebook and set:
- `train_hyperparam_opt` to `False` or `True` (to toggle train hyperparameter optimization via Optuna).
- `test_hyperparam_opt` to `False` or `True` (to toggle test hyperparameter optimization via Optuna).  
- `load_trained_model` to `False` if you want to train from scratch.  
- `save_model = True` if you wish to store the best model.

Then run all cells.

---

## Project Files
- **Molecular-Test-TIme-Adaptation.ipynb**: Is the main Jupyter Notebook.
- **data_utils.py**: Provides functions and classes for data loading and preprocessing.  
- **model_utils.py**: Contains the model architecture.  
- **train_test_utils.py**: Includes functions for training and predicting with and without TTA.

---

## Acknoledgements

This project was carried out as part of the seminar **165.164 Selected Topics in Theoretical Chemistry** at TU Wien, under the supervision of Prof. Esther Heid.