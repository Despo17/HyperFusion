import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

# ==============================
# MAIN FUNCTION
# ==============================

def evaluate_model(y_true, y_pred, y_prob=None, save_prefix="model"):
    """
    y_true  : actual labels (0/1)
    y_pred  : predicted labels (0/1)
    y_prob  : predicted probabilities (for ROC)
    """

    print("\n📊 Running Evaluation...")

    # ==============================
    # METRICS
    # ==============================

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n📈 PERFORMANCE METRICS")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Save metrics
    with open(f"{save_prefix}_metrics.txt", "w") as f:
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall   : {rec:.4f}\n")
        f.write(f"F1 Score : {f1:.4f}\n")

    # ==============================
    # CONFUSION MATRIX
    # ==============================

    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    classes = ["Low Vol", "High Vol"]
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    cm_path = f"{save_prefix}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    print(f"✅ Confusion Matrix saved: {cm_path}")

    # ==============================
    # ROC CURVE
    # ==============================

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        roc_path = f"{save_prefix}_roc_curve.png"
        plt.savefig(roc_path)
        plt.close()

        print(f"✅ ROC Curve saved: {roc_path}")

    print("\n🎯 Evaluation Completed Successfully!")


# ==============================
# HELPER (FOR YOUR PROJECT)
# ==============================

def evaluate_from_regression(y_true, y_pred):
    """
    Converts regression output (volatility) into classification
    """

    threshold = np.mean(y_true)

    y_true_cls = (y_true > threshold).astype(int)
    y_pred_cls = (y_pred > threshold).astype(int)

    # Normalize for ROC
    y_prob = y_pred / (np.max(y_pred) + 1e-8)

    evaluate_model(y_true_cls, y_pred_cls, y_prob)


# ==============================
# TEST RUN
# ==============================

if __name__ == "__main__":

    # Dummy data (replace with real)
    y_true = np.random.rand(100)
    y_pred = y_true + np.random.normal(0, 0.05, 100)

    evaluate_from_regression(y_true, y_pred)