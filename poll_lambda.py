#!/usr/bin/env python3
"""
Lambda Cloud GH200 (96GB) availability poller + auto-launcher.

Polls GET /instance-types until `gpu_1x_gh200` shows capacity in any
allowed region, then POSTs /instance-operations/launch. Stops on success,
auth failure, quota errors, or Ctrl-C.

Usage:
    export LAMBDA_API_KEY=secret_...
    python lambda_gh200_sniper.py                  # poll + launch
    python lambda_gh200_sniper.py --list-images    # print available image families and exit

Defaults: 1x GH200, image family `lambda-stack-24-04` (closest ARM-compatible
match to "GPU Base 24.04" — GPU Base is x86-64 only and won't run on the
GH200's ARM superchip), SSH key "laptop-bash", no filesystem.

Override via env vars:
    LAMBDA_INSTANCE_TYPE    Instance type (default: gpu_1x_gh200)
    LAMBDA_SSH_KEY_NAME     SSH key name (default: laptop-bash)
    LAMBDA_IMAGE_FAMILY     Image family (default: lambda-stack-24-04)
    LAMBDA_REGIONS          Comma-separated allowlist (default: any region)
    LAMBDA_INSTANCE_NAME    Name tag (default: gh200-auto)
    LAMBDA_USER_DATA_FILE   Optional cloud-init/user-data file path
    POLL_INTERVAL           Seconds between polls (default: 10)
    POLL_JITTER             Random extra seconds per interval (default: 3)
"""

import os
import sys
import time
import random
import json
import signal
from datetime import datetime
from urllib.parse import urljoin

import requests
from requests.auth import HTTPBasicAuth

API_BASE = "https://cloud.lambdalabs.com/api/v1/"
INSTANCE_TYPE = os.environ.get("LAMBDA_INSTANCE_TYPE", "gpu_1x_gh200")
#INSTANCE_TYPE = "gpu_2x_a100"

API_KEY = os.environ.get("LAMBDA_API_KEY")
SSH_KEY_NAME = os.environ.get("LAMBDA_SSH_KEY_NAME", "laptop-bash-root")
IMAGE_FAMILY = os.environ.get("LAMBDA_IMAGE_FAMILY", "gpu-base-24-04")
ALLOWED_REGIONS = [
    r.strip() for r in os.environ.get("LAMBDA_REGIONS", "").split(",") if r.strip()
]
INSTANCE_NAME = os.environ.get("LAMBDA_INSTANCE_NAME", "gh200-auto")
USER_DATA_FILE = os.environ.get("LAMBDA_USER_DATA_FILE")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "3"))
POLL_JITTER = float(os.environ.get("POLL_JITTER", "2"))

if not API_KEY:
    sys.exit("Set LAMBDA_API_KEY env var.")

AUTH = HTTPBasicAuth(API_KEY, "")
HEADERS = {"User-Agent": "gh200-sniper/1.0 (personal automation)"}

_stop = False
def _handle_sigint(signum, frame):
    global _stop
    _stop = True
    print("\n[!] Caught signal, stopping after this poll...", flush=True)
signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)


def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def list_images():
    """Print available images and exit."""
    url = urljoin(API_BASE, "images")
    r = requests.get(url, auth=AUTH, headers=HEADERS, timeout=15)
    if r.status_code == 401:
        sys.exit("[FATAL] 401 Unauthorized — check LAMBDA_API_KEY.")
    r.raise_for_status()
    images = r.json().get("data", [])
    print(f"{'family':<28} {'arch':<10} {'version':<10} {'id'}")
    print("-" * 90)
    for img in images:
        fam = img.get("family", "")
        arch = img.get("architecture", "")
        ver = img.get("version", "")
        iid = img.get("id", "")
        print(f"{fam:<28} {arch:<10} {ver:<10} {iid}")


def get_available_regions():
    """Return list of region names where the requested instance type has capacity."""
    url = urljoin(API_BASE, "instance-types")
    r = requests.get(url, auth=AUTH, headers=HEADERS, timeout=15)
    if r.status_code == 401:
        sys.exit("[FATAL] 401 Unauthorized — check LAMBDA_API_KEY.")
    if r.status_code == 429:
        log("[WARN] 429 rate-limited; backing off.")
        return None
    r.raise_for_status()
    data = r.json().get("data", {})
    entry = data.get(INSTANCE_TYPE)
    if not entry:
        log(f"[WARN] Instance type {INSTANCE_TYPE} not in catalog response.")
        return []
    regions = [r["name"] for r in entry.get("regions_with_capacity_available", [])]
    if ALLOWED_REGIONS:
        regions = [r for r in regions if r in ALLOWED_REGIONS]
    return regions


def launch(region):
    """Attempt to launch a GH200 in the given region. Returns instance ID list or None."""
    url = urljoin(API_BASE, "instance-operations/launch")
    payload = {
        "region_name": region,
        "instance_type_name": INSTANCE_TYPE,
        "ssh_key_names": [SSH_KEY_NAME],
        "file_system_names": [],            # explicitly none
        "quantity": 1,
        "name": INSTANCE_NAME,
        "image": {"family": IMAGE_FAMILY},  # alternative: {"id": "<image_id>"}
    }
    if USER_DATA_FILE:
        try:
            with open(USER_DATA_FILE, "r", encoding="utf-8") as f:
                payload["user_data"] = f.read()
        except OSError as e:
            sys.exit(f"[FATAL] Could not read LAMBDA_USER_DATA_FILE={USER_DATA_FILE!r}: {e}")
    log(f"[LAUNCH] region={region} payload={json.dumps(payload)}")
    r = requests.post(url, json=payload, auth=AUTH, headers=HEADERS, timeout=30)

    if r.status_code == 200:
        ids = r.json().get("data", {}).get("instance_ids", [])
        log(f"[SUCCESS] Launched: {ids}")
        return ids

    try:
        err = r.json().get("error", {})
    except ValueError:
        err = {}
    code = err.get("code", "")
    msg = err.get("message", r.text[:300])
    log(f"[FAIL] HTTP {r.status_code} code={code} msg={msg}")

    # Hard stops — no point retrying
    if r.status_code in (401, 403):
        sys.exit("[FATAL] Auth/permission error.")
    if "quota" in code.lower() or "quota" in msg.lower():
        sys.exit("[FATAL] Account quota reached — won't keep retrying.")
    # Bad image family / unknown SSH key won't fix itself either
    if r.status_code == 400 and any(k in msg.lower() for k in ("image", "ssh", "key")):
        sys.exit(f"[FATAL] Configuration error (image/ssh): {msg}")

    # Soft fails (capacity vanished, transient 5xx) → keep polling
    return None


def main():
    if "--list-images" in sys.argv:
        list_images()
        return

    log(f"Polling for {INSTANCE_TYPE} every ~{POLL_INTERVAL}s "
        f"(image={IMAGE_FAMILY}, ssh_key={SSH_KEY_NAME}, "
        f"regions={ALLOWED_REGIONS or 'any'})")
    consecutive_errors = 0
    while not _stop:
        try:
            regions = get_available_regions()
            consecutive_errors = 0
            if regions:
                log(f"[HIT] Capacity in: {regions}")
                for region in regions:
                    if launch(region):
                        return
                    # if launch failed (race lost), fall through and keep polling
            else:
                log(f"[miss] no capacity")
        except requests.RequestException as e:
            consecutive_errors += 1
            log(f"[ERROR] {e!r} (consecutive={consecutive_errors})")
            if consecutive_errors >= 5:
                sleep = min(300, POLL_INTERVAL * 2 ** min(consecutive_errors - 5, 5))
                log(f"[BACKOFF] sleeping {sleep:.0f}s")
                time.sleep(sleep)
                continue

        time.sleep(POLL_INTERVAL + random.uniform(0, POLL_JITTER))

    log("Stopped.")


if __name__ == "__main__":
    main()
