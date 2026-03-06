# Posture Detection Project

Real-time posture detection using a webcam. Classifies sitting posture into three categories: **upright**, **slouched**, and **leaning**.

## Prerequisites

- Python 3.x
- Webcam

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the model

Collect labeled training data and train the classifier:

```bash
python train_model.py
```

- Press **u** for upright
- Press **s** for slouched
- Press **l** for leaning
- Press **q** to stop collecting and train the model

For best accuracy, collect at least 50–100+ samples per class with varied sitting positions, distances, and lighting.

### 2. Run live prediction

After training, run real-time posture detection:

```bash
python live_prediction_model.py
```

- Press **q** to exit

**Options:**
- `--reminder SEC` – Remind you when in bad posture for SEC seconds (e.g. `--reminder 30`)
- `--log` – Log posture history to `posture_history.csv` for analytics

**Example with reminders and logging:**
```bash
python live_prediction_model.py --reminder 30 --log
```

### 3. View posture analytics

If you used `--log`, view stats:

```bash
python view_analytics.py
python view_analytics.py --today   # Today only
```

## Project Structure

- `train_model.py` – Data collection and model training
- `live_prediction_model.py` – Real-time posture prediction (reminders, session summary, streak)
- `view_analytics.py` – View posture history analytics
- `pose_utils.py` – Shared pose detection utilities
- `posture_data.csv` – Saved training data
- `posture_model.pkl` – Trained classifier
- `feature_columns.pkl` – Feature column order for inference

## Retraining

The model uses computed posture features (angles, ratios) rather than raw coordinates. If you have old `posture_model.pkl` or `feature_columns.pkl` from a previous raw-coordinate pipeline, delete them and retrain. Collect at least 50–100 samples per class for best accuracy.
