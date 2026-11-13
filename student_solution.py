import os
import math
import subprocess
import pandas as pd
from pathlib import Path
from sklearn.metrics import auc

TRAIN_FILE = "cardio_train.csv"
TEST_FILE = "cardio_test.csv"
LABEL_FILE = "cardio.labels"
NGA_FILE = "negsel.jar"

def process_data(in_fp):
    df = pd.read_csv(in_fp)
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    feats = []
    for i in range(21):
        feats.append(f"x_{i}")

    for feat in feats:
        df[feat] = pd.cut(df[feat], bins=10, labels=labels)

    processed = []

    for index, row in df[feats].iterrows():
        row_str = ""

        # form complete string
        for feat in feats:
            letter = row[feat]
            row_str += str(letter)

        processed.append(row_str)

    print(processed)
    out_fp = in_fp.replace(".csv", "_processed.csv")

    processed = pd.Series(processed)
    processed.to_csv(out_fp, index=False, header=False)

    return out_fp

def anom_scores(fp):
    args = ["java", "-jar", NGA_FILE, "-n", "10", "-r", "4", "-self", TRAIN_FILE,]
    with open(fp, "r", encoding="utf-8") as f:
        res = subprocess.run(args=args, stdin=f, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,check=False)
    if res.returncode != 0:
        raise RuntimeError(res.stderr)
    score_tup = tuple(float(value) for value in res.stdout.split())

    return score_tup

def final_anom(fp):
    print(f"Yay final_anom({fp})!")
    a_sum = 0.0
    for val in anom_scores(fp):
        if val == 0:
            val = 1e-6
        a_sum += math.log(val)
    return a_sum

def get_labels():
    label_tup = tuple()
    with open(LABEL_FILE, "r", encoding="utf-8") as f:
        for l in f:
            label_tup += (int(l.strip()),)

    return label_tup

def fp_tp_calc(fp, theta):
    print(f"Yay fp_tp_calc({fp}, {theta})!")
    s_vals = anom_scores(fp)
    l_vals = get_labels()

    # How to determine normal vs anomaly? Get labels
    # FPR = normal instances with score > theta / total normal instances
    # TPR = anomalous instances with score > theta / total anomalous instances
    n_scores: List[float] = []
    a_scores: List[float] = []
    for s, l in zip(s_vals, l_vals):
        if l == 0:
            n_scores.append(s)
        else:
            a_scores.append(s)

    n_gt = sum(s > theta for s in n_scores)
    a_gt = sum(s > theta for s in a_scores)

    fpr = n_gt / len(n_scores)
    tpr = a_gt / len(a_scores)
    # RETURN: (fpr, tpr)
    return (fpr, tpr)

def auc_calc(fp):
    print(f"Yay auc_calc({fp})!")
    # Int threshold vals range(0, 11)
    # call fptpcalc to get roc point in (FPR, TPR) form
    # Use the trapezoidal rule to calculate the area under the ROC curve. You may implement the trapezoidal rule manually or use sklearn.metrics.auc
    points = [fp_tp_calc(fp, theta) for theta in range(0, 11)]

    # ascending order
    points.sort(key=lambda point: point[0])

    # separate x, y
    fpr = [point[0] for point in points]
    tpr = [point[1] for point in points]

    return auc(fpr, tpr)

def main():
    print("Hello world!")
    ptrain_fp = process_data(TRAIN_FILE)
    print(f"processed train file: {ptrain_fp}")

if __name__ == "__main__":
    main()
