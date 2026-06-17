#!/usr/bin/env python3
"""
Launch a Lambda Cloud GPU instance, run a local worker shell script over SSH,
copy results back, and terminate the instance through the Lambda API.

The Lambda API key stays local. The API is used only for lifecycle operations
(poll/launch via poll_lambda.py, inspect status/IP, terminate); remote execution
and file transfer use SSH/SCP/rsync.

Lifecycle:
    1. Run poll_lambda.py as a subprocess to poll for capacity and launch.
    2. Wait until the instance is active and has a public IP.
    3. Wait for SSH to come up, then copy the env file (optional), setup
       (optional), and worker scripts to the remote workdir.
    4. Run the setup script (if given), then the worker script REMOTELY on the
       GPU instance, with the env file auto-sourced into both.
    5. On a clean worker run, copy results back: run the copy script LOCALLY on
       this machine if given, otherwise use the built-in rsync/scp.
    6. Terminate the instance via the API in a finally block (even on failure).

    Where each script runs:
      --setup-script   remote (on the GPU instance)
      worker_script    remote (on the GPU instance)
      --copy-script    local  (on this machine; pulls data back over SSH)

    The run is fail-fast: any failure in setup, the worker, or the result copy
    aborts immediately (results are NOT copied after a failed worker, since
    partial output isn't trusted). All orchestrator and remote output is teed to
    a local log file as it streams, so the log survives even when the run fails
    and the instance is destroyed.

Usage:
    export LAMBDA_API_KEY=secret_...
    python run_lambda_worker.py WORKER.sh --ssh-key-path ~/.ssh/lambda_ed25519 \\
        [--setup-script setup.sh] [--env-file .env] \\
        [--remote-results-dir /home/ubuntu/lambda-worker/out]

Examples:
    # Minimal: launch default instance, run worker.sh, terminate when done.
    python run_lambda_worker.py worker.sh --ssh-key-path ~/.ssh/lambda

    # Pin region/type, bootstrap with a setup script + env file, copy results back.
    python run_lambda_worker.py worker.sh \\
        --ssh-key-path ~/.ssh/lambda \\
        --setup-script setup.sh \\
        --env-file .env \\
        --region us-east-1 \\
        --instance-type gpu_1x_gh200 \\
        --lambda-ssh-key-name laptop-bash-root \\
        --user-data-file cloud-init.yaml \\
        --remote-workdir /home/ubuntu/lambda-worker \\
        --remote-results-dir /home/ubuntu/lambda-worker/results \\
        --local-results-dir ./lambda_results \\
        --log-file ./lambda_results/run.log

    # Inspect/debug without tearing down the instance afterward.
    python run_lambda_worker.py worker.sh --ssh-key-path ~/.ssh/lambda --keep-instance

Required:
    WORKER.sh           Local shell script copied to the instance and run there.
    --ssh-key-path      Private SSH key matching the Lambda SSH key name below.
    LAMBDA_API_KEY      API key in the environment (or set --api-key-env).

Key options (see --help for the full list and defaults):
    --setup-script          Local script run before the worker (clone repo, pip install);
                            if it fails, the worker is skipped.
    --setup-command         Override the remote setup command (default: bash ./<setup_script>).
    --env-file              Local KEY=VALUE file (e.g. HF_TOKEN=hf_...) copied to the remote
                            workdir (mode 600) and exported into setup + worker shells.
    --region                Lambda region; omit to accept any region.
    --instance-type         Instance type to launch (default: gpu_1x_gh200).
    --image-family          Image family for launch (default: gpu-base-24-04).
    --lambda-ssh-key-name   Name of the SSH key registered in Lambda.
    --user-data-file        cloud-init/user-data file for bootstrapping.
    --remote-workdir        Remote dir for the scripts (default: /home/ubuntu, i.e. ~).
    --remote-command        Override the remote command (default: bash ./WORKER.sh).
    --copy-script           Local script run on this machine to copy data back; replaces
                            the built-in copy. Gets LAMBDA_* env vars (see below).
    --remote-results-dir    Remote path for the built-in copy back after a clean worker run.
    --local-results-dir     Local destination for results (default: ./lambda_results).
    --log-file              Local log file (default: lambda-worker-<timestamp>.log).
    --copy-method           Built-in copy transport: rsync (default) or scp.
    --no-ssh-multiplex      Disable SSH connection multiplexing (see notes).
    --keep-instance         Skip termination at the end (for debugging).

Copy-script environment (set when --copy-script runs locally):
    LAMBDA_INSTANCE_IP        Public IP of the instance.
    LAMBDA_SSH_USER           SSH user (ubuntu).
    LAMBDA_SSH_KEY_PATH       Path to the private SSH key.
    LAMBDA_SSH_OPTS           Ready-to-use ssh invocation, e.g. rsync -e "$LAMBDA_SSH_OPTS".
    LAMBDA_REMOTE_WORKDIR     Remote working directory.
    LAMBDA_REMOTE_RESULTS_DIR Remote results path (only if --remote-results-dir was set).
    LAMBDA_LOCAL_RESULTS_DIR  Local results destination (created before the script runs).

Notes:
    - poll_lambda.py polls indefinitely until capacity appears (no launch timeout).
    - SSH host-key checking is disabled because Lambda recycles public IPs.
    - The instance is always terminated via the API on exit unless --keep-instance.
    - SSH connection multiplexing is on by default: only the first connection
      authenticates, so a passphrase-protected key prompts just once (when SSH
      first comes up) and every later ssh/scp/rsync reuses that connection. Load
      the key into ssh-agent (ssh-add) to avoid the prompt entirely.
    - LAMBDA_SSH_OPTS includes the multiplexing options, so a --copy-script using
      it reuses the same authenticated connection.
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import random
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from requests.auth import HTTPBasicAuth


API_BASE = "https://cloud.lambdalabs.com/api/v1/"
SSH_USER = "ubuntu"
ACTIVE_STATUSES = {"active", "running"}
FAILURE_STATUSES = {"unhealthy", "alert", "failed", "error"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local worker shell script on a temporary Lambda GPU instance."
    )
    parser.add_argument("worker_script", type=Path, help="Local shell script to copy and run remotely.")
    parser.add_argument(
        "--setup-script",
        type=Path,
        help="Optional local shell script copied and run before the worker (e.g. clone repo, pip install). "
        "If it fails, the worker is skipped.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional local KEY=VALUE file (e.g. with HF_TOKEN) copied to the remote workdir with mode 600 "
        "and auto-sourced (exported) into the setup and worker shells. Not echoed to logs.",
    )
    parser.add_argument("--api-key-env", default="LAMBDA_API_KEY", help="Env var holding the Lambda API key.")
    parser.add_argument("--poll-script", type=Path, default=Path("poll_lambda.py"), help="Existing poll/launch script.")
    parser.add_argument("--region", help="Lambda region name. If omitted, poll_lambda.py may use any region.")
    parser.add_argument("--instance-type", default="gpu_1x_gh200", help="Lambda instance type name.")
    parser.add_argument("--image-family", default=os.environ.get("LAMBDA_IMAGE_FAMILY", "gpu-base-24-04"))
    parser.add_argument("--lambda-ssh-key-name", default=os.environ.get("LAMBDA_SSH_KEY_NAME", "laptop-bash-root"))
    parser.add_argument("--instance-name", default=os.environ.get("LAMBDA_INSTANCE_NAME", "lambda-worker-auto"))
    parser.add_argument("--ssh-key-path", type=Path, required=True, help="Private SSH key for ubuntu@instance-ip.")
    parser.add_argument("--user-data-file", type=Path, help="Optional cloud-init/user-data file passed to launch.")
    parser.add_argument(
        "--remote-workdir",
        default="/home/ubuntu",
        help="Remote working directory (absolute path; this is ubuntu's home, i.e. ~).",
    )
    parser.add_argument(
        "--setup-command",
        help="Remote command to run for setup. Defaults to: bash ./<setup_script_name>",
    )
    parser.add_argument(
        "--remote-command",
        help="Remote command to run after copying the worker. Defaults to: bash ./<worker_script_name>",
    )
    parser.add_argument(
        "--copy-script",
        type=Path,
        help="Optional LOCAL shell script run on THIS machine after a clean worker run to copy data back. "
        "Receives instance details via LAMBDA_* env vars (LAMBDA_INSTANCE_IP, LAMBDA_SSH_USER, "
        "LAMBDA_SSH_KEY_PATH, LAMBDA_SSH_OPTS, LAMBDA_REMOTE_WORKDIR, LAMBDA_REMOTE_RESULTS_DIR, "
        "LAMBDA_LOCAL_RESULTS_DIR). If given, it replaces the built-in rsync/scp copy.",
    )
    parser.add_argument("--remote-results-dir", help="Remote path for the built-in copy back on success.")
    parser.add_argument("--local-results-dir", type=Path, default=Path("lambda_results"), help="Local results destination.")
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Local file to tee all orchestrator + remote output to. "
        "Default: lambda-worker-<timestamp>.log in the current directory.",
    )
    parser.add_argument("--copy-method", choices=("rsync", "scp"), default="rsync")
    parser.add_argument(
        "--no-ssh-multiplex",
        dest="ssh_multiplex",
        action="store_false",
        help="Disable SSH connection multiplexing (one auth shared across all connections).",
    )
    parser.add_argument("--poll-interval", type=float, default=float(os.environ.get("POLL_INTERVAL", "10")))
    parser.add_argument("--poll-jitter", type=float, default=float(os.environ.get("POLL_JITTER", "3")))
    parser.add_argument("--api-timeout", type=float, default=30)
    parser.add_argument("--instance-timeout", type=float, default=1800)
    parser.add_argument("--ssh-timeout", type=float, default=900)
    parser.add_argument("--ssh-connect-timeout", type=int, default=10)
    parser.add_argument("--keep-instance", action="store_true", help="Do not terminate the instance at the end.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def configure_logging(verbose: bool, log_file: Path) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    logging.info("Logging to %s", log_file)


def lambda_api_request(
    method: str,
    path: str,
    api_key: str,
    *,
    json_body: dict[str, Any] | None = None,
    timeout: float = 30,
    max_attempts: int = 6,
) -> dict[str, Any]:
    url = urljoin(API_BASE, path)
    auth = HTTPBasicAuth(api_key, "")
    headers = {"User-Agent": "lambda-worker-orchestrator/1.0"}

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.request(
                method,
                url,
                auth=auth,
                headers=headers,
                json=json_body,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            if attempt == max_attempts:
                raise RuntimeError(f"{method} {path} failed after {attempt} attempts: {exc}") from exc
            sleep_with_backoff(attempt, "Lambda API transport error")
            continue

        if response.status_code == 429 or response.status_code >= 500:
            if attempt == max_attempts:
                break
            sleep_with_backoff(attempt, f"Lambda API HTTP {response.status_code}")
            continue

        if response.status_code in (401, 403):
            raise RuntimeError(f"Lambda API auth/permission error HTTP {response.status_code}")

        if not response.ok:
            raise RuntimeError(f"Lambda API {method} {path} HTTP {response.status_code}: {response.text[:500]}")

        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(f"Lambda API {method} {path} returned non-JSON response") from exc

    raise RuntimeError(f"Lambda API {method} {path} HTTP {response.status_code}: {response.text[:500]}")


def sleep_with_backoff(attempt: int, reason: str) -> None:
    delay = min(120.0, 2.0 * (2 ** (attempt - 1))) + random.uniform(0.0, 2.0)
    logging.warning("%s; sleeping %.1fs before retry", reason, delay)
    time.sleep(delay)


def poll_capacity_or_launch_when_available(
    args: argparse.Namespace, api_key: str, id_sink: dict[str, str | None]
) -> str:
    # Polling for capacity is delegated to poll_lambda.py (run as a subprocess);
    # this orchestrator only consumes the launched instance id it reports.
    return launch_instance(args, api_key, id_sink)


def launch_instance(args: argparse.Namespace, api_key: str, id_sink: dict[str, str | None]) -> str:
    if not args.poll_script.exists():
        raise FileNotFoundError(f"Poll script not found: {args.poll_script}")

    env = os.environ.copy()
    env[args.api_key_env] = api_key
    env["LAMBDA_API_KEY"] = api_key
    env["LAMBDA_INSTANCE_TYPE"] = args.instance_type
    env["LAMBDA_IMAGE_FAMILY"] = args.image_family
    env["LAMBDA_SSH_KEY_NAME"] = args.lambda_ssh_key_name
    env["LAMBDA_INSTANCE_NAME"] = args.instance_name
    env["POLL_INTERVAL"] = str(args.poll_interval)
    env["POLL_JITTER"] = str(args.poll_jitter)
    if args.region:
        env["LAMBDA_REGIONS"] = args.region
    if args.user_data_file:
        env["LAMBDA_USER_DATA_FILE"] = str(args.user_data_file)

    logging.info("Launching %s with %s", args.instance_type, args.poll_script)
    proc = subprocess.Popen(
        [sys.executable, str(args.poll_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )

    launched_id: str | None = None
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        logging.info("poll_lambda: %s", line)
        ids = parse_launch_ids(line)
        if ids:
            launched_id = ids[0]
            id_sink["id"] = launched_id

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{args.poll_script} exited with status {rc}")
    if not launched_id:
        raise RuntimeError(f"{args.poll_script} exited without reporting a launched instance id")
    return launched_id


def parse_launch_ids(line: str) -> list[str]:
    marker = "[SUCCESS] Launched:"
    if marker not in line:
        return []
    raw = line.split(marker, 1)[1].strip()
    try:
        ids = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(ids, list):
        return []
    return [str(item) for item in ids if item]


def wait_for_active_instance(args: argparse.Namespace, api_key: str, instance_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + args.instance_timeout
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        instance = get_instance(api_key, instance_id, args.api_timeout)
        status = str(instance.get("status", "")).lower()
        ip = instance.get("ip") or instance.get("public_ip") or instance.get("hostname")
        logging.info("instance %s status=%s ip=%s", instance_id, status or "unknown", ip or "pending")
        if status in ACTIVE_STATUSES and ip:
            return instance
        if status in FAILURE_STATUSES:
            raise RuntimeError(f"Instance {instance_id} entered failure status {status!r}")
        sleep = min(60.0, args.poll_interval * (1.2 ** min(attempt, 15))) + random.uniform(0, args.poll_jitter)
        time.sleep(sleep)
    raise TimeoutError(f"Timed out waiting for instance {instance_id} to become active with a public IP")


def get_instance(api_key: str, instance_id: str, timeout: float) -> dict[str, Any]:
    payload = lambda_api_request("GET", "instances", api_key, timeout=timeout)
    data = payload.get("data", [])
    for instance in data:
        if str(instance.get("id")) == instance_id:
            return instance
    raise RuntimeError(f"Instance {instance_id} was not found in Lambda instances list")


def wait_for_ssh(args: argparse.Namespace, ip: str) -> None:
    deadline = time.monotonic() + args.ssh_timeout
    test = "true"
    while time.monotonic() < deadline:
        rc = run_remote_command(args, ip, test, check=False, log_command=False, quiet=True)
        if rc == 0:
            logging.info("SSH is ready on %s", ip)
            return
        logging.info("SSH not ready yet on %s: rc=%s", ip, rc)
        time.sleep(min(30.0, args.poll_interval) + random.uniform(0, args.poll_jitter))
    raise TimeoutError(f"Timed out waiting for SSH on {ip}")


def ssh_options(args: argparse.Namespace) -> list[str]:
    # Lambda recycles public IPs across ephemeral instances, so a stale
    # known_hosts entry for a reused IP would otherwise abort every connection
    # with REMOTE HOST IDENTIFICATION HAS CHANGED. For throwaway instances we
    # skip host-key persistence entirely.
    opts = [
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        f"ConnectTimeout={args.ssh_connect_timeout}",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=6",
    ]
    # Connection multiplexing: the first connection opens a master socket and
    # authenticates (one passphrase prompt); subsequent ssh/scp/rsync calls reuse
    # it. ControlPersist keeps the master alive briefly between connections.
    control_path = getattr(args, "ssh_control_path", None)
    if control_path:
        opts += [
            "-o",
            "ControlMaster=auto",
            "-o",
            f"ControlPath={control_path}",
            "-o",
            "ControlPersist=600",
        ]
    return opts


def ssh_base(args: argparse.Namespace) -> list[str]:
    return ["ssh", "-i", str(args.ssh_key_path), *ssh_options(args)]


def close_ssh_master(args: argparse.Namespace, ip: str | None) -> None:
    # Tear down the multiplexing master connection so no ssh process lingers
    # after the run (otherwise it persists for ControlPersist seconds).
    control_path = getattr(args, "ssh_control_path", None)
    if not control_path or not ip:
        return
    try:
        subprocess.run(
            ["ssh", "-i", str(args.ssh_key_path), *ssh_options(args), "-O", "exit", f"{SSH_USER}@{ip}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        logging.debug("Failed to close SSH master connection", exc_info=True)


def run_remote_command(
    args: argparse.Namespace,
    ip: str,
    command: str,
    *,
    check: bool = True,
    log_command: bool = True,
    quiet: bool = False,
) -> int:
    target = f"{SSH_USER}@{ip}"
    full_cmd = [*ssh_base(args), target, command]
    if log_command:
        logging.info("remote: %s", command)
    if quiet:
        # Used for the SSH readiness probe: discard the connection-refused noise.
        rc = subprocess.run(full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    else:
        # Stream stdout+stderr line by line through logging so remote output is
        # teed live to both the console and the local log file. This means the
        # worker's output is preserved locally even when the run fails and we
        # never copy results back.
        proc = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logging.info("remote| %s", line.rstrip())
        rc = proc.wait()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, full_cmd)
    return rc


def run_local_command(cmd: list[str], *, env: dict[str, str] | None = None, label: str = "local") -> None:
    # Local counterpart to run_remote_command: stream stdout+stderr through
    # logging so a locally-run script (e.g. the copy script) is teed to the log.
    logging.info("%s: %s", label, " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        logging.info("local| %s", line.rstrip())
    rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def copy_script_to_remote(args: argparse.Namespace, ip: str, local_script: Path) -> str:
    remote_script = f"{args.remote_workdir.rstrip('/')}/{local_script.name}"
    run_remote_command(args, ip, f"mkdir -p {shlex.quote(args.remote_workdir)}")
    if args.copy_method == "rsync":
        ssh = " ".join(shlex.quote(part) for part in ssh_base(args))
        cmd = [
            "rsync",
            "-azP",
            "-e",
            ssh,
            str(local_script),
            f"{SSH_USER}@{ip}:{shlex.quote(remote_script)}",
        ]
    else:
        cmd = [
            "scp",
            "-i",
            str(args.ssh_key_path),
            *ssh_options(args),
            str(local_script),
            f"{SSH_USER}@{ip}:{shlex.quote(remote_script)}",
        ]
    logging.info("copying %s to %s", local_script.name, remote_script)
    subprocess.run(cmd, check=True)
    run_remote_command(args, ip, f"chmod +x {shlex.quote(remote_script)}")
    return remote_script


def copy_results_back(args: argparse.Namespace, ip: str) -> None:
    if not args.remote_results_dir:
        logging.info("No --remote-results-dir configured; skipping result copy")
        return
    args.local_results_dir.mkdir(parents=True, exist_ok=True)
    remote = f"{SSH_USER}@{ip}:{args.remote_results_dir.rstrip('/')}/"
    if args.copy_method == "rsync":
        ssh = " ".join(shlex.quote(part) for part in ssh_base(args))
        cmd = ["rsync", "-azP", "-e", ssh, remote, str(args.local_results_dir)]
    else:
        cmd = [
            "scp",
            "-r",
            "-i",
            str(args.ssh_key_path),
            *ssh_options(args),
            remote,
            str(args.local_results_dir),
        ]
    logging.info("copying results from %s to %s", args.remote_results_dir, args.local_results_dir)
    subprocess.run(cmd, check=True)


def run_copy_script(args: argparse.Namespace, ip: str) -> None:
    # Runs locally. Exposes the instance details so the script can pull data back
    # however it likes (selective rsync, tar-over-ssh, upload to S3, ...).
    # LAMBDA_SSH_OPTS is a ready-to-use ssh invocation, e.g. for `rsync -e`.
    args.local_results_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LAMBDA_INSTANCE_IP"] = ip
    env["LAMBDA_SSH_USER"] = SSH_USER
    env["LAMBDA_SSH_KEY_PATH"] = str(args.ssh_key_path)
    env["LAMBDA_SSH_OPTS"] = " ".join(shlex.quote(part) for part in ssh_base(args))
    env["LAMBDA_REMOTE_WORKDIR"] = args.remote_workdir
    env["LAMBDA_LOCAL_RESULTS_DIR"] = str(args.local_results_dir)
    if args.remote_results_dir:
        env["LAMBDA_REMOTE_RESULTS_DIR"] = args.remote_results_dir
    logging.info("running local copy script %s for instance %s", args.copy_script, ip)
    run_local_command(["bash", str(args.copy_script)], env=env, label="copy")


def copy_env_to_remote(args: argparse.Namespace, ip: str, local_env: Path) -> str:
    # Copied via scp/rsync (never passed on the command line, so the secret stays
    # out of `ps` and this script's command logging) and locked to mode 600.
    remote_env = f"{args.remote_workdir.rstrip('/')}/{local_env.name}"
    run_remote_command(args, ip, f"mkdir -p {shlex.quote(args.remote_workdir)}")
    if args.copy_method == "rsync":
        ssh = " ".join(shlex.quote(part) for part in ssh_base(args))
        cmd = ["rsync", "-azP", "-e", ssh, str(local_env), f"{SSH_USER}@{ip}:{shlex.quote(remote_env)}"]
    else:
        cmd = [
            "scp",
            "-i",
            str(args.ssh_key_path),
            *ssh_options(args),
            str(local_env),
            f"{SSH_USER}@{ip}:{shlex.quote(remote_env)}",
        ]
    logging.info("copying env file to %s (mode 600)", remote_env)
    subprocess.run(cmd, check=True)
    run_remote_command(args, ip, f"chmod 600 {shlex.quote(remote_env)}")
    return remote_env


def remote_run_prefix(args: argparse.Namespace, remote_env: str | None) -> str:
    # cd into the workdir, and (if an env file was copied) export its KEY=VALUE
    # entries into the shell before running the command.
    prefix = f"cd {shlex.quote(args.remote_workdir)} && "
    if remote_env:
        prefix += f"set -a && . {shlex.quote(remote_env)} && set +a && "
    return prefix


def run_setup(args: argparse.Namespace, ip: str, remote_script: str, remote_env: str | None) -> None:
    remote_command = args.setup_command or f"bash {shlex.quote(remote_script)}"
    run_remote_command(args, ip, f"{remote_run_prefix(args, remote_env)}{remote_command}")


def run_worker(args: argparse.Namespace, ip: str, remote_script: str, remote_env: str | None) -> None:
    remote_command = args.remote_command or f"bash {shlex.quote(remote_script)}"
    run_remote_command(args, ip, f"{remote_run_prefix(args, remote_env)}{remote_command}")


def terminate_instance(args: argparse.Namespace, api_key: str, instance_id: str) -> None:
    logging.info("Terminating Lambda instance %s through API", instance_id)
    payload = {"instance_ids": [instance_id]}
    lambda_api_request(
        "POST",
        "instance-operations/terminate",
        api_key,
        json_body=payload,
        timeout=args.api_timeout,
        max_attempts=8,
    )


def main() -> int:
    args = parse_args()
    log_file = args.log_file or Path(f"lambda-worker-{time.strftime('%Y%m%d-%H%M%S')}.log")
    configure_logging(args.verbose, log_file)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        logging.error("Set %s in the local environment", args.api_key_env)
        return 2
    if not args.worker_script.exists():
        logging.error("worker script not found: %s", args.worker_script)
        return 2
    if args.setup_script and not args.setup_script.exists():
        logging.error("setup script not found: %s", args.setup_script)
        return 2
    if args.env_file and not args.env_file.exists():
        logging.error("env file not found: %s", args.env_file)
        return 2
    if args.copy_script and not args.copy_script.exists():
        logging.error("copy script not found: %s", args.copy_script)
        return 2
    if not args.ssh_key_path.exists():
        logging.error("SSH key path not found: %s", args.ssh_key_path)
        return 2

    # SSH connection multiplexing: a per-run control socket lives in a temp dir
    # (unique path avoids colliding with a stale socket from a recycled IP). The
    # first connection authenticates; the rest reuse it.
    control_dir: str | None = None
    args.ssh_control_path = None
    if args.ssh_multiplex:
        control_dir = tempfile.mkdtemp(prefix="lw-cm-")
        args.ssh_control_path = os.path.join(control_dir, "cm-%C")

    # id_sink is populated the moment poll_lambda.py reports a launched id, so a
    # Ctrl-C between launch and the subprocess exiting still leaves something for
    # the finally block to terminate (otherwise we'd leak a billed instance).
    id_sink: dict[str, str | None] = {"id": None}
    ip: str | None = None
    try:
        instance_id = poll_capacity_or_launch_when_available(args, api_key, id_sink)
        instance = wait_for_active_instance(args, api_key, instance_id)
        ip = str(instance.get("ip") or instance.get("public_ip") or instance.get("hostname"))
        wait_for_ssh(args, ip)

        remote_env = copy_env_to_remote(args, ip, args.env_file) if args.env_file else None
        remote_setup = copy_script_to_remote(args, ip, args.setup_script) if args.setup_script else None
        remote_worker = copy_script_to_remote(args, ip, args.worker_script)

        # Fail-fast: setup -> worker -> copy results. Any failure raises and
        # aborts the run (the finally block still terminates the instance). We
        # only copy results back after a clean worker run, since partial output
        # isn't trusted; remote logs are already teed into the local log file.
        if remote_setup:
            run_setup(args, ip, remote_setup, remote_env)
        run_worker(args, ip, remote_worker, remote_env)
        if args.copy_script:
            run_copy_script(args, ip)
        else:
            copy_results_back(args, ip)
        return 0
    except KeyboardInterrupt:
        logging.warning("Interrupted")
        return 130
    except Exception:
        logging.exception("Lambda worker run failed")
        return 1
    finally:
        instance_id = id_sink["id"]
        if instance_id and not args.keep_instance:
            try:
                terminate_instance(args, api_key, instance_id)
            except Exception:
                logging.exception("Failed to terminate instance %s", instance_id)
        elif instance_id:
            logging.warning("Keeping instance %s because --keep-instance was set", instance_id)
        if control_dir:
            close_ssh_master(args, ip)
            shutil.rmtree(control_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
