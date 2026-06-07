#!/usr/bin/env python3
"""
Locust Load Generator for IntentContinuum Experiments

Runs on the SDN controller and sends requests via SSH to the master node,
where they travel through the SDN network to the application microservices.

Supports two modes:
1. Flat load: --users N (constant N users for the whole run)
2. Staged load: Uses LoadTestShape to vary users over time, matching
   the IntentContinuum paper's pattern [10, 20, 15, 10, 5, 20, 10]

Usage:
    # Flat load:
    locust -f locustfile.py --headless --users 10 --spawn-rate 1 \
        --run-time 900s --host ssh://master --csv=results

    # Staged load (automatic, just set run-time):
    LOCUST_STAGED=1 locust -f locustfile.py --headless \
        --run-time 900s --host ssh://master --csv=results
"""

import os
import subprocess
import time
import uuid
import logging

from locust import User, task, between, events, LoadTestShape

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
MASTER_HOST = os.environ.get("MASTER_HOST", "10.56.1.209")
MASTER_USER = os.environ.get("MASTER_USER", "cc")
REMOTE_IMAGE = os.environ.get("REMOTE_IMAGE", "/home/cc/test_converted.jpg")
SDN_ENTRY_POINT = os.environ.get("SDN_ENTRY_POINT", "http://192.168.100.100:5001/resize")
WEBHOOKS = os.environ.get("WEBHOOKS", "http://microservice2-service:5002/bw,http://microservice3-service:8081/,http://microservice4-service:5004/notify")
DB_URL = os.environ.get("DB_URL", "http://db-service:5006/track_time")
LOGS_URL = os.environ.get("LOGS_URL", "http://db-service:5006/log")
REQUEST_TIMEOUT = 60  # seconds

# When True, run curl directly (no SSH wrapper). Used when Locust runs on the
# same machine that can reach the application endpoint directly.
USE_DIRECT_CURL = os.environ.get("DIRECT_CURL", "0") == "1"

# ── Staged Load Configuration ─────────────────────────────────────────────
# Each tuple: (duration_seconds, num_users, spawn_rate)
# Hybrid of the IntentContinuum paper pattern [10, 20, 15, 10, 5, 20, 10]: same oscillating
# shape for comparability, but stages stretched 120s -> 180s and the two peaks raised 20 -> 25
# users so they sustain pressure long enough to separate the controllers (TiB% / MTTR).
LOAD_STAGES = [
    (180, 10, 1),   # 0-180s:     10 users (ramp up)
    (180, 25, 2),   # 180-360s:   25 users (peak)
    (180, 15, 1),   # 360-540s:   15 users (decrease)
    (180, 10, 1),   # 540-720s:   10 users (normal)
    (180, 5, 1),    # 720-900s:   5 users  (low)
    (180, 25, 2),   # 900-1080s:  25 users (peak again)
    (180, 10, 1),   # 1080-1260s: 10 users (settle)
]

USE_STAGED = os.environ.get("LOCUST_STAGED", "0") == "1"


class ImageProcessingUser(User):
    """
    Simulates a sensor/camera sending images to the processing pipeline.
    
    Each user sends synchronous requests one at a time via SSH to the
    master node, matching the IntentContinuum paper's experimental setup.
    """
    
    # Wait 0.5-1.5 seconds between requests per user
    wait_time = between(0.5, 1.5)
    
    @task
    def send_image_request(self):
        """Send an image processing request via SSH to the master node."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        if USE_DIRECT_CURL:
            command = [
                "curl", "-X", "POST",
                "-F", f"image=@{REMOTE_IMAGE}",
                "-H", f"X-Request-ID: {request_id}",
                "-H", f"X-Webhooks: {WEBHOOKS}",
                "-H", "X-Special-Object: person",
                "-H", f"X-Central-DB-URL: {DB_URL}",
                "-H", f"X-Logs-URL: {LOGS_URL}",
                SDN_ENTRY_POINT,
                "--max-time", str(REQUEST_TIMEOUT),
                "-w", "%{time_total}",
                "-o", "/dev/null", "-s",
            ]
        else:
            curl_command = (
                f'curl -X POST '
                f'-F "image=@{REMOTE_IMAGE}" '
                f'-H "X-Request-ID: {request_id}" '
                f'-H "X-Webhooks: {WEBHOOKS}" '
                f'-H "X-Special-Object: person" '
                f'-H "X-Central-DB-URL: {DB_URL}" '
                f'-H "X-Logs-URL: {LOGS_URL}" '
                f'{SDN_ENTRY_POINT} '
                f'--max-time {REQUEST_TIMEOUT} '
                f'-w "%{{time_total}}" '
                f'-o /dev/null -s'
            )
            command = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                f"{MASTER_USER}@{MASTER_HOST}",
                curl_command,
            ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=REQUEST_TIMEOUT + 30
            )
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            if result.returncode == 0:
                try:
                    response_time_s = float(result.stdout.strip())
                    response_time_ms = response_time_s * 1000
                    
                    events.request.fire(
                        request_type="POST",
                        name="/resize",
                        response_time=response_time_ms,
                        response_length=0,
                        exception=None,
                        context={}
                    )
                except ValueError:
                    events.request.fire(
                        request_type="POST",
                        name="/resize",
                        response_time=elapsed_ms,
                        response_length=0,
                        exception=Exception(f"Could not parse response: {result.stdout}"),
                        context={}
                    )
            else:
                events.request.fire(
                    request_type="POST",
                    name="/resize",
                    response_time=elapsed_ms,
                    response_length=0,
                    exception=Exception(f"curl failed (rc={result.returncode}): {result.stderr[:200]}"),
                    context={}
                )
                
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.time() - start_time) * 1000
            events.request.fire(
                request_type="POST",
                name="/resize",
                response_time=elapsed_ms,
                response_length=0,
                exception=Exception("SSH timeout"),
                context={}
            )
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            events.request.fire(
                request_type="POST",
                name="/resize",
                response_time=elapsed_ms,
                response_length=0,
                exception=e,
                context={}
            )


if USE_STAGED:
    class StagesShape(LoadTestShape):
        """
        Staged load pattern matching the IntentContinuum paper.

        Varies user count over time: [10, 20, 15, 10, 5, 20, 10]
        with 120-second intervals between changes.

        Only active when LOCUST_STAGED=1 environment variable is set.
        """

        stages = LOAD_STAGES

        def tick(self):
            run_time = self.get_run_time()

            elapsed = 0
            for duration, users, spawn_rate in self.stages:
                if run_time < elapsed + duration:
                    return (users, spawn_rate)
                elapsed += duration

            # Past all stages — stop
            return None