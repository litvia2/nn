# Sample Neural Network (PyTorch)

This is a minimal example of a feedforward neural network (MLP) trained on the scikit-learn Digits dataset using PyTorch.

## Project Structure

- `mlp_digits_pytorch.py` — the main training script
- `requirements.txt` — dependencies for running the script
- `README.md` — this file

## Setup

1. **Clone or unzip this repo**
   ```bash
   unzip sample_nn_project.zip
   cd sample_nn_project
   ```

2. **(Optional) Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # on mac/linux
   # .venv\Scripts\activate   # on windows (powershell)
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the training script**
   ```bash
   python mlp_digits_pytorch.py
   ```

## What it does

- Loads the classic `digits` dataset (8x8 grayscale images).
- Trains a small neural network (MLP with ReLU + Dropout).
- Tracks training/validation accuracy.
- Saves the best model to `best_mlp_digits.pt`.
- Prints a classification report at the end.

## Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- numpy

---

Enjoy experimenting 🚀
