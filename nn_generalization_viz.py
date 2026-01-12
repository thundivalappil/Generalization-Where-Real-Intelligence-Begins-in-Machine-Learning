"""
Neural Network Generalization 

This script generates 3 visuals that clearly explain generalization:

1) Training vs Validation learning curves (loss + accuracy)
2) Decision boundary (complex vs smooth)
3) Train vs Test confidence scatter (consistency check)

Outputs are saved as PNGs in ./outputs

Run:
    python nn_generalization_viz.py
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# 1) Synthetic dataset (2D moons)

def make_moons(n=1200, noise=0.25, seed=42):
    rng = np.random.default_rng(seed)
    n1 = n // 2
    t1 = rng.uniform(0, np.pi, size=n1)
    x1 = np.c_[np.cos(t1), np.sin(t1)]
    t2 = rng.uniform(0, np.pi, size=n - n1)
    x2 = np.c_[1 - np.cos(t2), 1 - np.sin(t2) - 0.5]
    X = np.vstack([x1, x2])
    y = np.hstack([np.zeros(n1, dtype=int), np.ones(n - n1, dtype=int)])
    X += rng.normal(scale=noise, size=X.shape)
    idx = rng.permutation(n)
    return X[idx], y[idx]

X, y = make_moons()

# Split: train/val/test
rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
X, y = X[idx], y[idx]
n = len(X)
n_train = int(0.65 * n)
n_val = int(0.15 * n)

X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

# Standardize (important for neural nets)
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0) + 1e-9

X_train_s = (X_train - mu) / sd
X_val_s   = (X_val   - mu) / sd
X_test_s  = (X_test  - mu) / sd

# Outputs folder
out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)


# 2) Train two networks: (overfit) vs (generalize)
#    Uses TensorFlow if available; otherwise falls back to scikit-learn.

use_tf = True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    tf.random.set_seed(42)
except Exception:
    use_tf = False

def predict_proba(model, Xs):
    if use_tf:
        return model.predict(Xs, verbose=0).reshape(-1)
    return model.predict_proba(Xs)[:, 1]

def train_tf(hidden_layers, l2=0.0, dropout=0.0, epochs=250, lr=1e-3, patience=18, name="model"):
    inputs = keras.Input(shape=(2,))
    x = inputs
    for units in hidden_layers:
        x = layers.Dense(
            units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2) if l2 > 0 else None
        )(x)
        if dropout > 0:
            x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name=name)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        )
    ]

    hist = model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=epochs,
        batch_size=64,
        verbose=0,
        callbacks=callbacks
    )
    return model, hist.history

def train_sklearn(hidden_layers, alpha=0.0, max_iter=600):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=alpha,              # L2 regularization
        learning_rate_init=0.001,
        max_iter=max_iter,
        random_state=42
    )
    clf.fit(X_train_s, y_train)
    return clf, None

if use_tf:
    overfit_model, overfit_hist = train_tf([128, 128, 128], l2=0.0,   dropout=0.0,  patience=30, name="overfit_nn")
    good_model,    good_hist    = train_tf([32, 32],       l2=1e-3,  dropout=0.25, patience=18, name="generalize_nn")
else:
    overfit_model, overfit_hist = train_sklearn((128, 128, 128), alpha=0.0)
    good_model,    good_hist    = train_sklearn((32, 32),         alpha=1e-3)

# 3) Visual 1 Learning Curves

def plot_learning_curves(hist, title, outpath):
    plt.figure(figsize=(8.5, 5.2))
    epochs = np.arange(1, len(hist["loss"]) + 1)
    plt.plot(epochs, hist["loss"],         label="Training loss")
    plt.plot(epochs, hist["val_loss"],     label="Validation loss")
    plt.plot(epochs, hist["accuracy"],     label="Training accuracy")
    plt.plot(epochs, hist["val_accuracy"], label="Validation accuracy")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

if use_tf:
    plot_learning_curves(overfit_hist, "Learning Curves (Overfitting Pattern)", out_dir / "learning_curves_overfit.png")
    plot_learning_curves(good_hist,    "Learning Curves (Better Generalization)", out_dir / "learning_curves_generalize.png")
else:
    print("TensorFlow not found. Creating illustrative learning-curve plots instead.")
    # Illustrative (fast) learning curves to still explain generalization visually
    import numpy as np
    def _make_curves(kind="overfit", n=35, seed=7):
        rng = np.random.default_rng(seed)
        e = np.arange(1, n+1)
        if kind=="overfit":
            train_loss = np.exp(-e/10) + 0.05*rng.normal(size=n) + 0.15
            val_loss   = np.exp(-e/12) + 0.05*rng.normal(size=n) + 0.22 + (e/n)*0.55
            train_acc  = 0.55 + 0.45*(1-np.exp(-e/10)) + 0.02*rng.normal(size=n)
            val_acc    = 0.55 + 0.40*(1-np.exp(-e/12)) - (e/n)*0.20 + 0.02*rng.normal(size=n)
        else:
            train_loss = np.exp(-e/11) + 0.04*rng.normal(size=n) + 0.18
            val_loss   = np.exp(-e/11) + 0.04*rng.normal(size=n) + 0.20
            train_acc  = 0.55 + 0.40*(1-np.exp(-e/11)) + 0.015*rng.normal(size=n)
            val_acc    = 0.55 + 0.38*(1-np.exp(-e/11)) + 0.015*rng.normal(size=n)
        return e, train_loss.clip(0, None), val_loss.clip(0, None), train_acc.clip(0,1), val_acc.clip(0,1)
    def _plot_curves(kind, title, outpath):
        e, tl, vl, ta, va = _make_curves(kind=kind)
        plt.figure(figsize=(8.5, 5.2))
        plt.plot(e, tl, label="Training loss")
        plt.plot(e, vl, label="Validation loss")
        plt.plot(e, ta, label="Training accuracy")
        plt.plot(e, va, label="Validation accuracy")
        plt.title(title + " (illustrative)")
        plt.xlabel("Epoch")
        plt.ylabel("Metric value")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(outpath, dpi=220)
        plt.close()
    _plot_curves("overfit", "Learning Curves (Overfitting Pattern)", out_dir / "learning_curves_overfit.png")
    _plot_curves("generalize", "Learning Curves (Better Generalization)", out_dir / "learning_curves_generalize.png")

# ----------------------------
# 4) Visual 2 Decision Boundary
# ----------------------------
def plot_decision_boundary(model, title, outpath):
    x_min, x_max = X_train_s[:, 0].min() - 1.2, X_train_s[:, 0].max() + 1.2
    y_min, y_max = X_train_s[:, 1].min() - 1.2, X_train_s[:, 1].max() + 1.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 360), np.linspace(y_min, y_max, 360))
    grid = np.c_[xx.ravel(), yy.ravel()]

    zz = predict_proba(model, grid).reshape(xx.shape)

    plt.figure(figsize=(7.3, 6.2))
    plt.contourf(xx, yy, zz, levels=30, alpha=0.9)
    plt.contour(xx, yy, zz, levels=[0.5], linewidths=2)

    plt.scatter(X_train_s[:, 0], X_train_s[:, 1], c=y_train, s=18, edgecolor="k", linewidth=0.2, alpha=0.9, label="Train")
    plt.scatter(X_test_s[:, 0],  X_test_s[:, 1],  c=y_test,  s=22, marker="x", alpha=0.9, label="Test")

    plt.title(title)
    plt.xlabel("Feature 1 (standardized)")
    plt.ylabel("Feature 2 (standardized)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

plot_decision_boundary(overfit_model, "Decision Boundary (Complex → Likely Overfitting)", out_dir / "decision_boundary_overfit.png")
plot_decision_boundary(good_model,    "Decision Boundary (Smooth → Better Generalization)", out_dir / "decision_boundary_generalize.png")

# ----------------------------
# 5) Visual 3 Train vs Test Confidence
# ----------------------------
def plot_confidence_scatter(model, outpath):
    p_train = predict_proba(model, X_train_s)
    p_test  = predict_proba(model, X_test_s)

    plt.figure(figsize=(8.5, 5.3))
    plt.scatter(y_train + np.random.normal(0, 0.02, len(y_train)), p_train, alpha=0.55, label="Train", s=18, edgecolor="k", linewidth=0.15)
    plt.scatter(y_test  + np.random.normal(0, 0.02, len(y_test)),  p_test,  alpha=0.75, label="Test",  s=26, marker="x")

    plt.title("Train vs Test Confidence (Consistency = Generalization)")
    plt.xlabel("Actual class (0/1)")
    plt.ylabel("Predicted probability (class=1)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()

plot_confidence_scatter(good_model, out_dir / "train_vs_test_confidence.png")

print("\nDone PNGs saved in:", out_dir.resolve())
