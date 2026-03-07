#!/usr/bin/env python3
"""
Locust Load Generator for IntentContinuum Experiments

Runs on the SDN controller and sends requests via SSH to the master node,
where they travel through the SDN network to the application microservices.

This matches the same request mechanism used by the Intent Watch Loop.

Usage:
    # Headless mode with staged load (recommended for experiments):
    python3 -m locust -f locustfile.py --headless --host ssh://master \
        --run-time 900s --csv results/experiment1

    # Or use the run_experiment.py wrapper which handles everything.
"""

import os
import subprocess
import time
import uuid
import logging

from locust import User, task, between, events

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────
# These match config.yaml - adjust if your setup differs
MASTER_HOST = os.environ.get("MASTER_HOST", "10.132.0.7")
MASTER_USER = os.environ.get("MASTER_USER", "antonios-icontinuum")
REMOTE_IMAGE = os.environ.get("REMOTE_IMAGE", "/home/antonios-icontinuum/test_converted.jpg")
SDN_ENTRY_POINT = os.environ.get("SDN_ENTRY_POINT", "http://192.168.100.100:5001/resize")
WEBHOOKS = os.environ.get("WEBHOOKS", "http://microservice2-service:5002/bw,http://microservice3-service:8081/,http://microservice4-service:5004/notify")
DB_URL = os.environ.get("DB_URL", "http://db-service:5006/track_time")
LOGS_URL = os.environ.get("LOGS_URL", "http://db-service:5006/log")
REQUEST_TIMEOUT = 60  # seconds


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
        
        ssh_command = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            f"{MASTER_USER}@{MASTER_HOST}",
            curl_command
        ]
        
        try:
            result = subprocess.run(
                ssh_command,
                capture_output=True,
                text=True,
                timeout=REQUEST_TIMEOUT + 30
            )
            
            elapsed_ms = (time.time() - start_time) * 1000  # Locust expects ms
            
            if result.returncode == 0:
                try:
                    response_time_s = float(result.stdout.strip())
                    response_time_ms = response_time_s * 1000
                    
                    # Report success to Locust
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
