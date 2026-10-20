# Salary Regression Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a reproducible notebook pipeline that prepares the salary dataset for manual neural-network regression and writes train/test CSVs plus a preprocessing report.

**Architecture:** `DataProcessing.ipynb` will contain small, testable functions for schema validation, train-only fitting of categorical/numeric transformations, controlled train-only oversampling, artifact writing, and invariant checks. The original CSV remains unchanged; the notebook writes processed artifacts under `archive/`.

**Tech Stack:** Python 3.12, pandas, NumPy, scikit-learn `train_test_split`, Jupyter notebook JSON, JSON report output.

## Global Constraints

- Target: `salary_in_usd` as continuous regression target.
- Currency: use existing `salary_in_usd`; do not recalculate exchange rates.
- Oversampling: sampling with replacement, only after the train/test split.
- Balance `work_mode`, `experience_level`, and `role_family` with a controlled cap.
- Keep the test distribution unchanged and never fit transformations from test data.
- Keep input columns `work_year`, `experience_level`, `job_title`, `remote_ratio`, `work_mode`, and `role_family`.
- Remove `salary`, `salary_currency`, `employee_residence`, `company_location`, `company_size`, `employment_type`, `employment_type_label`, `experience_level_label`, `salary_outlier_flag`, and `isco_group_hint`.
- Final artifacts must contain no `NaN` or infinite values.
- Do not modify the manual MLP architecture or training algorithm.

---

## File Map

- Modify: `DataProcessing.ipynb` — executable preprocessing pipeline and validation output.
- Read: `archive/ai_jobs_salaries_clean.csv` — source dataset.
- Create: `archive/train_processed.csv` — normalized, one-hot encoded, oversampled training data.
- Create: `archive/test_processed.csv` — normalized, one-hot encoded test data with original distribution.
- Create: `archive/preprocessing_report.json` — schema, split, scaling, and distribution metadata.
- Modify: `docs/superpowers/specs/2026-10-01-salary-regression-preprocessing-design.md` only if implementation uncovers a confirmed design correction.

## Interfaces

The notebook must define these functions:

```python
validate_source_dataframe(df: pd.DataFrame) -> None
split_target_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]
fit_transformers(X_train: pd.DataFrame) -> dict
transform_features(X: pd.DataFrame, transformers: dict) -> pd.DataFrame
controlled_oversample(
    train_df: pd.DataFrame,
    balance_columns: list[str],
    target_quantile: float = 0.50,
    max_multiplier: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]
build_preprocessing_report(...) -> dict
run_preprocessing(input_path: str, output_dir: str, seed: int = 42) -> dict
```

`run_preprocessing` must return a report dictionary and write all three output
artifacts. The transformed CSVs must use one shared feature-column order, with
`salary_in_usd` as the final target column.

### Task 1: Add notebook setup and source-schema validation

**Files:**
- Modify: `DataProcessing.ipynb`

- [ ] **Step 1: Add imports and constants**

Add a code cell defining:

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("archive/ai_jobs_salaries_clean.csv")
OUTPUT_DIR = Path("archive")
SEED = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "salary_in_usd"
FEATURE_COLUMNS = [
    "work_year", "experience_level", "job_title",
    "remote_ratio", "work_mode", "role_family",
]
DROP_COLUMNS = [
    "salary", "salary_currency", "employee_residence",
    "company_location", "company_size", "employment_type",
    "employment_type_label", "experience_level_label",
    "salary_outlier_flag", "isco_group_hint",
]
BALANCE_COLUMNS = ["work_mode", "experience_level", "role_family"]
```

- [ ] **Step 2: Write `validate_source_dataframe`**

Implement explicit checks:

```python
def validate_source_dataframe(df):
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN] + DROP_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        raise TypeError(f"{TARGET_COLUMN} must be numeric")
    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"{TARGET_COLUMN} contains missing values")
    if not np.isfinite(df[TARGET_COLUMN].to_numpy(dtype=float)).all():
        raise ValueError(f"{TARGET_COLUMN} contains non-finite values")
```

- [ ] **Step 3: Run the validation cell**

Run the notebook cell with the source CSV. Expected result: validation passes
for the current 71,913-row dataset; a missing required column must raise a
named `ValueError`.

- [ ] **Step 4: Commit the setup**

```bash
git add DataProcessing.ipynb
git commit -m "feat: validate salary preprocessing input"
```

### Task 2: Implement leakage-safe split and feature transformers

**Files:**
- Modify: `DataProcessing.ipynb`

**Interfaces:**
- Consumes: `validate_source_dataframe`, `FEATURE_COLUMNS`, and `TARGET_COLUMN`.
- Produces: `split_target_features`, `fit_transformers`, and `transform_features`.

- [ ] **Step 1: Add `split_target_features`**

Implement duplicate removal before splitting and preserve the continuous target:

```python
def split_target_features(df):
    clean = df.drop_duplicates().reset_index(drop=True)
    X = clean[FEATURE_COLUMNS].copy()
    y = clean[TARGET_COLUMN].copy()
    return X, y
```

- [ ] **Step 2: Add train-only transformer fitting**

`fit_transformers` must:

1. Fill numeric missing values from train medians.
2. Fill categorical missing values with `"Unknown"`.
3. Record the complete categorical vocabulary from train.
4. One-hot encode with stable train-derived columns.
5. Record train min/max for every resulting numeric feature.
6. Replace zero denominators with `1.0`.

Use a dictionary with keys `numeric_columns`, `categorical_columns`,
`medians`, `categories`, `encoded_columns`, `minimums`, and `denominators`.
Do not use test rows while fitting any key.

- [ ] **Step 3: Add `transform_features`**

Transform an arbitrary split using only the saved dictionary. Reindex one-hot
columns to `encoded_columns` with zeros for unseen categories, apply saved
medians, and compute:

```python
scaled = (encoded - minimums) / denominators
return scaled.replace([np.inf, -np.inf], np.nan).fillna(0.0)
```

The function must return a `DataFrame` with the exact same column order for
train and test.

- [ ] **Step 4: Add focused notebook assertions**

Run a split and assert:

```python
assert set(FEATURE_COLUMNS).isdisjoint({TARGET_COLUMN})
assert list(X_train_transformed.columns) == list(X_test_transformed.columns)
assert np.isfinite(X_train_transformed.to_numpy()).all()
assert np.isfinite(X_test_transformed.to_numpy()).all()
assert ((X_train_transformed >= 0) & (X_train_transformed <= 1)).all().all()
```

- [ ] **Step 5: Commit the transformation layer**

```bash
git add DataProcessing.ipynb
git commit -m "feat: add leakage-safe feature transformations"
```

### Task 3: Implement controlled train-only oversampling

**Files:**
- Modify: `DataProcessing.ipynb`

**Interfaces:**
- Consumes: raw `X_train`, `y_train`, and `BALANCE_COLUMNS`.
- Produces: oversampled raw training frame and a per-column count report.

- [ ] **Step 1: Define the controlled target size**

For each balance column, calculate the median category count as the default
target and cap the target at `max_multiplier * category_count` for each rare
category. Do not target the largest class count. This prevents
`Other / Unclassified` from causing unbounded expansion.

- [ ] **Step 2: Implement `controlled_oversample`**

Use deterministic `np.random.default_rng(seed)`. For each category below its
target, sample existing rows from that category with replacement and append
them. Keep every appended row’s original `salary_in_usd`. Record counts before
and after for each balance column.

The implementation must ensure:

```python
assert len(oversampled) >= len(train_df)
assert set(oversampled.columns) == set(train_df.columns)
assert oversampled[TARGET_COLUMN].notna().all()
```

- [ ] **Step 3: Prevent test contamination**

Call `controlled_oversample` only on the training frame after
`train_test_split`. Add an assertion that the test row count and per-column
category counts equal the pre-oversampling test snapshot.

- [ ] **Step 4: Verify distribution behavior**

Run the current dataset and print the before/after counts for
`work_mode`, `experience_level`, and `role_family`. Verify that the result is
larger than or equal to the original train set, finite, deterministic across
two runs with seed `42`, and does not expand by more than the configured
controlled cap.

- [ ] **Step 5: Commit the balancing layer**

```bash
git add DataProcessing.ipynb
git commit -m "feat: control train category oversampling"
```

### Task 4: Build the end-to-end artifact writer

**Files:**
- Modify: `DataProcessing.ipynb`
- Create: `archive/train_processed.csv`
- Create: `archive/test_processed.csv`
- Create: `archive/preprocessing_report.json`

**Interfaces:**
- Consumes: all validation, split, transformation, and oversampling functions.
- Produces: three persistent artifacts and the report returned by `run_preprocessing`.

- [ ] **Step 1: Implement `build_preprocessing_report`**

Include:

```python
{
    "input_path": str(INPUT_PATH),
    "seed": 42,
    "test_size": 0.20,
    "input_rows": ...,
    "deduplicated_rows": ...,
    "train_rows_before_oversampling": ...,
    "train_rows_after_oversampling": ...,
    "test_rows": ...,
    "target_column": "salary_in_usd",
    "feature_columns": [...],
    "removed_columns": [...],
    "balance_columns": [...],
    "balance_counts_before": {...},
    "balance_counts_after": {...},
    "scaling": "train_min_max",
}
```

- [ ] **Step 2: Implement `run_preprocessing`**

The function must:

1. Read and validate the source.
2. Drop exact duplicates.
3. Split raw `X` and `y` with `train_test_split(..., test_size=0.20, random_state=42, shuffle=True)`.
4. Snapshot test distributions.
5. Oversample only raw train rows.
6. Fit transformers on the oversampled train data only.
7. Transform train and test with one shared feature schema.
8. Append `salary_in_usd` as the final column.
9. Write both CSVs and UTF-8 JSON report.
10. Assert output invariants before returning.

- [ ] **Step 3: Add final invariant checks**

Assert:

```python
train = pd.read_csv(OUTPUT_DIR / "train_processed.csv")
test = pd.read_csv(OUTPUT_DIR / "test_processed.csv")
assert train.columns.tolist() == test.columns.tolist()
assert train.columns[-1] == TARGET_COLUMN
assert TARGET_COLUMN not in train.drop(columns=[TARGET_COLUMN]).columns
assert np.isfinite(train.to_numpy(dtype=float)).all()
assert np.isfinite(test.to_numpy(dtype=float)).all()
assert len(test) == report["test_rows"]
```

Also verify the test category snapshots against the raw held-out test frame
before the categorical columns are one-hot encoded.

- [ ] **Step 4: Execute the full notebook**

Run all notebook cells from a clean kernel. Expected result: the three
artifacts exist, both CSVs have matching schemas, the target is numeric, and
the report contains all required metadata.

- [ ] **Step 5: Commit the complete pipeline**

```bash
git add DataProcessing.ipynb archive/train_processed.csv archive/test_processed.csv archive/preprocessing_report.json
git commit -m "feat: generate processed salary regression datasets"
```

### Task 5: Verify compatibility with the manual network

**Files:**
- Read: `Train.py`
- Modify: `DataProcessing.ipynb` only if an incompatibility is found

- [ ] **Step 1: Load the generated CSVs**

Confirm that the input matrix is numeric, the target is the final column, and
the train/test feature columns match exactly.

- [ ] **Step 2: Check regression target handling**

Confirm that the notebook does not bin, threshold, or normalize
`salary_in_usd`; the manual network receives a continuous target. If the
network expects a separate `X`/`y` split, use:

```python
X_train = train.drop(columns=[TARGET_COLUMN]).to_numpy(dtype=float)
y_train = train[TARGET_COLUMN].to_numpy(dtype=float).reshape(-1, 1)
X_test = test.drop(columns=[TARGET_COLUMN]).to_numpy(dtype=float)
y_test = test[TARGET_COLUMN].to_numpy(dtype=float).reshape(-1, 1)
```

- [ ] **Step 3: Run a smoke test**

Run the smallest available training invocation or a forward pass using the
generated `X_train` shape. Expected result: matrix multiplication accepts the
feature dimensions and no `NaN`/`inf` reaches the network.

- [ ] **Step 4: Commit compatibility corrections**

```bash
git add DataProcessing.ipynb
git commit -m "test: verify processed data for manual network"
```

## Plan self-review

- **Spec coverage:** target and removed columns are covered in Tasks 1 and 4;
  train-only split and transformations in Task 2; controlled balancing and
  unchanged test distribution in Task 3; artifacts and validation in Task 4;
  network compatibility in Task 5.
- **Placeholder scan:** no placeholder marker or unspecified implementation
  step is required; all named functions, paths, parameters, and output checks
  are explicit.
- **Type consistency:** all later tasks consume the function names and
  dictionary keys defined in the interface section.
- **Repository limitation:** this workspace currently is not a Git
  repository, so the commit commands are execution-ready but cannot run until
  the project is placed in a repository.
