"""
locust/locustfile.py
---------------------
Simulates a flood of prediction requests against the API to measure latency
and response time under load, at different container counts.

Run (from repo root, with the API running e.g. on http://localhost:8000):
    locust -f locust/locustfile.py --host http://localhost:8000

Then open http://localhost:8089, set number of users + spawn rate, and
start the test. Repeat with `docker compose up --scale api=1/2/4` to compare
latency/RPS at different scale-out levels for the README's results section.

Place a handful of sample .jpg images in locust/sample_images/ before running.
"""

import os
import random
from locust import HttpUser, task, between

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_images")


class ScenePredictUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        if os.path.isdir(SAMPLE_DIR):
            self.images = [
                os.path.join(SAMPLE_DIR, f) for f in os.listdir(SAMPLE_DIR)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        else:
            self.images = []

    @task(5)
    def predict(self):
        if not self.images:
            return
        img_path = random.choice(self.images)
        with open(img_path, "rb") as f:
            self.client.post("/predict", files={"file": (os.path.basename(img_path), f, "image/jpeg")})

    @task(1)
    def check_uptime(self):
        self.client.get("/uptime")

    @task(1)
    def health(self):
        self.client.get("/health")
