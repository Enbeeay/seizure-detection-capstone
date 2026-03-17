# Seizure Forecasting Demo

This project is a machine learning pipeline for seizure prediction using intracranial EEG (iEEG) data from the Melbourne-University AES-MathWorks-NIH Seizure Prediction Challenge dataset. The system processes raw .mat EEG recordings, removes invalid signals, normalizes and downsamples the data, splits each recording into time windows, and extracts time-domain and spectral features. These features are then used in deep learning and statistical models to distinguish preictal brain activity (the period leading up to a seizure) from interictal activity (normal periods between seizures). The goal of the project is to improve seizure prediction performance on highly imbalanced EEG data while reducing false positives and building a pipeline that can be extended with patient-specific modeling and more advanced architectures.

## What it does

- Loads `.mat` EEG clips stored either as `dataStruct.data` or a top-level `data`
  matrix.
- Removes all-zero dropout rows.
- Splits each clip into sliding windows.
- Extracts time-domain and FFT bandpower features per channel.
- Trains a simple ridge forecaster on interictal windows.
- Scores a second clip by next-window forecast error.
- Saves plots and summary metrics for a quick demo.

## Run it

Use the local Unreal Engine Python plus the vendored dependencies:

```powershell
& '.\run_demo.py --train .\Pat1Test_1_0.mat --eval .\Pat1Train_1_1.mat
```

`--train` should point to an interictal clip and `--eval` should point to a
preictal clip for the cleanest first demo.

You can also point the script at your dataset root and let it pick files based
on the Kaggle naming pattern:

```powershell
& .\run_demo.py --dataset-root D:\seizure-detection-data --patient 1
```

This expects filenames in one of these formats:

- `I_J_K.mat` for training clips where `K=0` is interictal and `K=1` is preictal
- `I_J.mat` for unlabeled test clips

If your files are nested into patient folders or train/test folders, that is
fine; the indexer searches recursively under the dataset root.

## Outputs

The script writes files under `outputs/`:

- `forecast_error_timeseries.png`
- `forecast_error_distribution.png`
- `metrics.json`

The main metrics to show in a demo are `roc_auc`, `pr_auc`, and the difference
between `train_score_mean` and `eval_score_mean`.

## Deep Train/Test Pipeline

For the full file-level project, use the new deep sequence forecaster:

```powershell
.\.venv\Scripts\python.exe .\train_deep_forecaster.py --train-root D:\seizure-detection-data-train --test-root D:\seizure-detection-data-test
```

You can choose the window representation with `--feature-mode`:

```powershell
.\.venv\Scripts\python.exe .\train_deep_forecaster.py --train-root D:\seizure-detection-data-train --test-root D:\seizure-detection-data-test --test-label-csv D:\contest_test_data_labels_public.csv --feature-mode time
.\.venv\Scripts\python.exe .\train_deep_forecaster.py --train-root D:\seizure-detection-data-train --test-root D:\seizure-detection-data-test --test-label-csv D:\contest_test_data_labels_public.csv --feature-mode spectral
.\.venv\Scripts\python.exe .\train_deep_forecaster.py --train-root D:\seizure-detection-data-train --test-root D:\seizure-detection-data-test --test-label-csv D:\contest_test_data_labels_public.csv --feature-mode both
```

What it does:

- loads all labeled train and test files matching names like `Pat1Train_J_K.mat`
  and `Pat1Test_J_K.mat`
- creates a deep sequence forecaster on time-domain, spectral, or combined EEG windows
- trains the forecaster on one subset of the training files
- fits a probability calibrator on a second subset using file labels
- reports validation metrics on a held-out subset
- writes file-level probabilities for the test set to CSV

Main outputs are written to `outputs/deep_forecaster/`:

- `validation_predictions.csv`
- `test_predictions.csv`
- `metrics.json`
- `sequence_forecaster.pt`
- `probability_calibrator.joblib`
- `training_loss.png`
