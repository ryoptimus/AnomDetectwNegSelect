import os
import math
import subprocess
import pandas as pd
from pathlib import Path
from sklearn.metrics import auc

TRAIN_FILE = "cardio_train.csv"
PTRAIN_FILE = "cardio_train_processed.csv"
TEST_FILE = "cardio_test.csv"
PTEST_FILE = "cardio_test_processed.csv"
LABEL_FILE = "cardio.labels"
NGA_FILE = "negsel.jar"

def process_data(in_fp):
    df = pd.read_csv(in_fp)

    labels = list("ABCDEFGHIJ")
    feats = [f"x_{i}" for i in range(21)]

    for feat in feats:
        df[feat] = pd.cut(
            df[feat],
            bins=10,
            labels=labels,
            right=(feat != "x_20")
        )

    processed = df[feats].agg("".join, axis=1)

    out_fp = in_fp.replace(".csv", "_processed.csv")
    processed.to_csv(out_fp, index=False, header=False)

    return out_fp

def anom_scores(fp, r):
    #java_path = "/mnt/c/Program Files/Common Files/Oracle/Java/javapath/java.exe"
    java_path = "java"
    args = [java_path, "-jar", NGA_FILE, "-n", "21", "-r", str(r), "-self", PTRAIN_FILE,]
    with open(fp, "r", encoding="utf-8") as f:
        res = subprocess.run(args=args, stdin=f, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,check=False)
    if res.returncode != 0:
        raise RuntimeError(res.stderr)
    score_tup = tuple(float(value) for value in res.stdout.split())

    return score_tup

def final_anom(fp, r):
    a_sum = 0.0
    for val in anom_scores(fp, r):
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

def fp_tp_calc(fp, theta, r):
    s_vals = anom_scores(fp, r)
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

def auc_calc(fp, r):
    points = [fp_tp_calc(fp, theta, r) for theta in range(11)]

    # ascending order
    points.sort(key=lambda point: point[0])

    # separate x, y
    fpr = [point[0] for point in points]
    tpr = [point[1] for point in points]

    return auc(fpr, tpr)

def r_auc_tuple(fp):
    ptrain_fp = process_data(TRAIN_FILE)
    processed_test = process_data(fp)

    tuples = []

    for r in range(2, 11):
        auc_val = auc_calc(processed_test, r)
        tuples.append((r, auc_val))

    return tuples

def main():
    ra_tups = r_auc_tuple(TEST_FILE)

if __name__ == "__main__":
    main()


