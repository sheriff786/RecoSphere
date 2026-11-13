"""
Load test for the RecoSphere API.

    python -m serve.make_loadtest_users --n 5000          # once, creates users.txt / items.txt
    locust -f loadtest/locustfile.py --host http://127.0.0.1:8000 --headless -u 20 -r 5 -t 2m --csv results/loadtest

Use 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and every new connection waits ~2 s
before falling back to IPv4 (shows up as a ~2 s max latency, one per simulated user).

Report in REPORT.md: RPS, p50, p95, failures, number of simulated users (-u), and the hardware.
Events are off by default (they change user histories); set LOCUST_EVENT_WEIGHT=1 to include them.
"""
import os
import random
from pathlib import Path

from locust import HttpUser, between, task

HERE = Path(__file__).parent
USERS = [x for x in (HERE / "users.txt").read_text().split("\n") if x]
ITEMS = [x for x in (HERE / "items.txt").read_text().split("\n") if x]
EVENT_WEIGHT = int(os.environ.get("LOCUST_EVENT_WEIGHT", 0))


class Shopper(HttpUser):
    wait_time = between(0.0, 0.05)

    @task(20)
    def recommend(self):
        u = random.choice(USERS)
        # separate rows in the report: personalized path vs fallback path (made-up users)
        name = "/recommend [fallback]" if u.startswith("new_user_") else "/recommend [personalized]"
        self.client.get(f"/recommend/{u}", name=name)

    @task(1)
    def event(self):
        if EVENT_WEIGHT:
            self.client.post("/event", json={"user_id": random.choice(USERS), "item_id": random.choice(ITEMS)},
                             name="/event")
