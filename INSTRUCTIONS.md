# CSE 568 Project 04: Anomaly Detection with Negative Selection

Submit a single file named `student_solution.py`.

Your file must define these functions:

```python
def process_data(file):
    ...

def r_auc_tuple(file):
    ...
```

You may also include helper functions such as `fp_tp_calc(file, theta, r)` or `auc_calc(file, r)`.

## Assignment summary

This project applies the same Textor negative selection algorithm from Project 03 to the cardiotocography data set.

Input files available during grading:

```text
cardio_train.csv
cardio_test.csv
cardio.labels
negsel.jar
```

For compatibility with zyBooks-style paths, the autograder also creates:

```text
/usercode/cardio_train.csv
/usercode/cardio_test.csv
/usercode/cardio.labels
/usercode/negsel.jar
/usercode/negsel_src/negsel.jar
```

## Question 1: `process_data(file)`

Convert either `cardio_train.csv` or `cardio_test.csv` into a processed CSV containing one 21-character string per row.

Requirements:

```text
- Use only the 21 feature columns x_0 through x_20.
- Do not include the sample column.
- Do not include the label column from cardio_test.csv.
- Bin each feature column into 10 bins.
- Use labels A, B, C, D, E, F, G, H, I, J.
- Do not include a header or index in the processed output file.
- Return the path to the processed output file.
```

The first processed test row should be:

```text
EDADDABADABFBEEAGFFCF
```

## Question 3: `r_auc_tuple(file)`

Return a list of tuples for `r` values from 2 through 10 inclusive:

```python
[(2, auc_for_r_2), (3, auc_for_r_3), ..., (10, auc_for_r_10)]
```

For each value of `r`, run the negative selection algorithm with string length `n = 21`, using the processed training file as `self` and the processed test file as input.

Use this command pattern for anomaly scores:

```bash
java -jar negsel.jar -n 21 -r R_VALUE -self PROCESSED_TRAIN_FILE < PROCESSED_TEST_FILE
```

Do not use the `-c` flag when computing AUC values.

Use the labels in `cardio.labels`, where `0` means normal and `1` means anomalous. Calculate ROC points for integer theta values from 0 through 10 inclusive, then compute AUC using the trapezoidal rule.

The sample checks from the notebook are:

```python
tuples = r_auc_tuple('cardio_test.csv')
assert len(tuples) == 9
assert round(tuples[0][1], 2) == 0.75
assert round(tuples[7][1], 2) == 0.0
```
