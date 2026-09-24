# Data

**Do not commit datasets to Git.** They are large, may be license-restricted, and bloat history.

Instead:

1. Document exactly where the data comes from (name, version, URL, date accessed).
2. Provide a script or one-liner to download it into this folder.
3. Reference the file(s) via `config.yaml` (`data.source: csv`, `data.csv_path: data/your_file.csv`).

## This project's dataset

- **Name / source:** _e.g., CIC-IoT2023 — https://www.unb.ca/cic/datasets/iotdataset-2023.html_
- **Download:** _command or steps_
- **Target column:** _e.g., `label`_
- **Notes:** _class balance, preprocessing, known label issues_

The template ships with a **synthetic** dataset (`data.source: synthetic`) so `make reproduce`
runs before you wire up real data.

## CSV label requirements

The starter model and metrics support **binary classification**. Numeric `0/1`
labels work without extra settings. For other two-class labels, set
`data.positive_label` to the label representing the positive class (for example,
`positive_label: attack` for `benign/attack`, or `positive_label: 2` for `1/2`).
That label becomes `1`; the other becomes `0`. Precision, recall, F1 and ROC-AUC
refer to this positive class. Match the CSV label's type in YAML: quote string
labels and leave numeric labels unquoted. Missing targets and datasets with other
than two classes are rejected with a clear error. For multiclass work, adapt both
the model and evaluation, or explicitly map the task to two classes first.

The grading harness reads metrics from `output.dir` in `config.yaml`. If you
change this directory, also update `.gitignore` to exclude the generated files.
