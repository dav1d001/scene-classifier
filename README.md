# Scene Classifier — End-to-End ML Pipeline & MLOps

Image classification of natural scenes (buildings, forest, glacier, mountain, sea, street)
using the Intel Image Classification dataset, deployed as an API + dashboard with a
retraining pipeline, trigger, and load-testing results.

**🌐 Live URL:** API: https://scene-classifier-y85d.onrender.com
UI: https://scene-classifier-ui.onrender.com

---

## Project Description

This project demonstrates the full ML lifecycle for an image classifier:

- **Data acquisition & preprocessing** — Intel Image Classification dataset (6 scene classes)
- **Model creation** — transfer learning (MobileNetV2) + dropout regularization + early stopping,
  with a fine-tuning stage
- **Evaluation** — accuracy, loss, precision, recall, F1 (macro), confusion matrix
  (see `notebook/scene_classifier.ipynb`)
- **API** — FastAPI service for prediction, bulk upload, and retraining (`api/main.py`)
- **UI** — Streamlit dashboard for model uptime, data visualizations, prediction, and
  triggering retraining (`ui/app.py`)
- **Retraining pipeline** — fine-tunes the deployed model on newly uploaded labeled images,
  only promoting the new model if it doesn't regress accuracy (`src/retrain.py`)
- **Load testing** — Locust script simulating request floods against the API, compared
  across different numbers of Docker containers (`locust/locustfile.py`)

## Directory Structure

```
scene-classifier/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
├── notebook/
│   └── scene_classifier.ipynb      # full training + evaluation pipeline
├── src/
│   ├── preprocessing.py            # data acquisition + preprocessing (shared by train & retrain)
│   ├── model.py                    # model architecture, training, evaluation
│   ├── prediction.py               # single-image inference
│   └── retrain.py                  # retraining pipeline + auto-trigger logic
├── api/
│   ├── main.py                     # FastAPI app (predict / upload / retrain / uptime)
│   └── database.py                 # SQLite logging for monitoring
├── ui/
│   └── app.py                      # Streamlit dashboard
├── locust/
│   └── locustfile.py               # flood-request load test
├── data/
│   ├── train/  └── test/           # place the extracted Kaggle dataset here (seg_train/seg_test)
└── models/
    └── scene_classifier.keras      # saved model artifact (produced by the notebook)
```

## Setup

### 1. Clone & install

```bash
git clone <(https://github.com/dav1d001/scene-classifier.git)>
cd scene-classifier
python -m venv venv && source venv/bin/activate     # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Get the dataset (Kaggle)

1. Create a free Kaggle account and generate an API token: Account → Create New API Token,
   which downloads `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<you>\.kaggle\kaggle.json` (Windows).
3. Run the first two cells of `notebook/scene_classifier.ipynb` — this downloads and extracts
   the **Intel Image Classification** dataset via `kagglehub`.

### 3. Train the model

Run `notebook/scene_classifier.ipynb` top to bottom. It will:
- preprocess the data
- train + fine-tune the model
- print all evaluation metrics and the confusion matrix
- save `models/scene_classifier.keras`
- demonstrate the retraining function on simulated new uploads

### 4. Run the API locally

```bash
uvicorn api.main:app --reload --port 8000
```

Docs / manual testing: `http://localhost:8000/docs`

### 5. Run the UI locally

```bash
API_URL=http://localhost:8000 streamlit run ui/app.py
```

Open `http://localhost:8501`.

### 6. Run everything with Docker

```bash
docker compose up --build
```

- API → `http://localhost:8000`
- UI  → `http://localhost:8501`

### 7. Load-test with Locust

```bash
# put a few sample images in locust/sample_images/ first
locust -f locust/locustfile.py --host http://localhost:8000
```

Open `http://localhost:8089`, set number of users + spawn rate, start the test.
To compare latency at different scale-out levels, re-run against different container counts:

```bash
docker compose up --build --scale api=1
docker compose up --build --scale api=2
docker compose up --build --scale api=4
```

## Flood Request Simulation — Results

| Containers | Users | Spawn rate | Median latency | 95th %ile latency | RPS | Failures |
|---|---|---|---|---|---|---|
| 1 |50 |5|~43,000ms|~183,000|0.7 |1% | |

| Environment | Users | Spawn rate | Median latency | 95th %ile latency | RPS | Failures |
|---|---|---|---|---|---|---|
|Render|free tier|(1 instance)|50|5|17,000 ms|37,000 ms|1.07|0%




NB:
Render's free tier runs on a single shared CPU instance, so this reflects single-instance, resource-constrained behavior
The local Docker container-scaling comparison (1 vs 2 vs 4) was attempted but blocked by a persistent Windows/WSL2 networking issue that intermittently prevented local requests from completing, despite containers themselves running and passing health checks
Zero failures under load — the system degrades gracefully (slower) rather than crashing, which is a meaningful result on its own

