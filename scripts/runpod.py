"""Provision / inspect / terminate a RunPod GPU box for the steering sweep.

Why RunPod at all: the DGX Spark manages ~65 tok/s on this workload (it is
decode-bound on unified memory), which puts the full grid at ~100 hours. Worse, its
GPU allocations come out of system RAM, so an oversized batch does not merely OOM the
job -- it starves the OS until sshd stops answering. A discrete GPU cannot do that.

We deploy with PUBLIC_KEY so RunPod's image installs our key into authorized_keys at
boot. That avoids mutating the account's SSH settings.

  python scripts/runpod.py up          # deploy
  python scripts/runpod.py status      # ip/port/cost
  python scripts/runpod.py down        # TERMINATE (billing stops)
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import urllib.request

KEY_FILE = pathlib.Path.home() / ".runpod_key"
PUBKEY = (pathlib.Path.home() / ".ssh/id_ed25519.pub").read_text().strip()
# Multiple pods: the steering sweep and the RL replication are independent jobs and
# want different GPUs, so each gets its own state file and can be torn down alone.
POD_NAME = os.environ.get("POD", "sweep")
STATE = pathlib.Path(__file__).parent.parent / "results" / f"runpod_{POD_NAME}.json"

PROFILES = {
    # 8B inference, decode-bound: buy memory bandwidth.
    "sweep": {"gpu": "H100 PCIe", "disk": 80},
    # 3B PPO (actor + critic + reference + optimizer states): buy VRAM, not bandwidth.
    "rl": {"gpu": "A100 PCIe", "disk": 120},
    # Jacobian-lens fitting on an 8B model: modest, cheapest GPU that fits it.
    "jlens": {"gpu": "H100 PCIe", "disk": 80},
}

# H100 PCIe: 80GB, ~2TB/s memory bandwidth (~7x the Spark). The 8B model plus SAE is
# ~18GB, so VRAM is not the constraint -- decode throughput is.
IMAGE = os.environ.get(
    "IMAGE", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
# DRIVER CEILING, learned the expensive way 2026-07-19. RunPod A100 PCIe hosts
# were on driver 570.133.20 = CUDA 12.8. `pip install vllm` pulls the current
# build, which wants torch 2.11+cu130, and that fails at import with
#
#   RuntimeError: The NVIDIA driver on your system is too old (found version 12080)
#
# The image SHIPS a working torch (2.4.1+cu124); our own install is what breaks
# it, so the failure looks like a pod problem rather than something we did. The
# working combination on that driver is `vllm==0.11.0` -> torch 2.8.0+cu128.
# Check `nvidia-smi --query-gpu=driver_version` on a new pod before installing,
# and pin vLLM to match rather than letting the resolver choose.
# GPU availability on RunPod changes hour to hour -- a hardcoded type fails hard with
# "no longer any instances available". GPU= overrides the profile without editing it.
#   GPU="A100 PCIe" POD=jlens python scripts/runpod.py up
GPU_TYPE = os.environ.get("GPU") or PROFILES[POD_NAME]["gpu"]
DISK_GB = PROFILES[POD_NAME]["disk"]


def _key() -> str:
    raw = KEY_FILE.read_text().strip()
    return raw.split("=")[-1].strip().strip("\"'")


def gql(query: str, variables: dict | None = None) -> dict:
    body: dict = {"query": query}
    if variables:
        body["variables"] = variables
    req = urllib.request.Request(
        f"https://api.runpod.io/graphql?api_key={_key()}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "sot/1.0"},
    )
    out = json.load(urllib.request.urlopen(req))
    if out.get("errors"):
        raise SystemExit(f"RunPod API error: {out['errors'][0].get('message')}")
    return out["data"]


def up() -> None:
    if STATE.exists():
        print(f"pod already recorded in {STATE}; run 'status' or 'down' first")
        return

    gpus = gql("query { gpuTypes { id displayName memoryInGb } }")["gpuTypes"]
    match = [g for g in gpus if g["displayName"] == GPU_TYPE]
    if not match:
        raise SystemExit(f"{GPU_TYPE} not offered right now")
    gpu_id = match[0]["id"]

    data = gql(
        """
        mutation Deploy($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) { id name machineId }
        }
        """,
        {
            "input": {
                "cloudType": "ALL",
                "gpuCount": 1,
                "gpuTypeId": gpu_id,
                "name": f"sot-{POD_NAME}",
                "imageName": IMAGE,
                "containerDiskInGb": DISK_GB,
                "volumeInGb": 0,
                "ports": "22/tcp",
                # RunPod's images append PUBLIC_KEY to authorized_keys at boot, so we
                # get SSH without touching the account's global SSH settings.
                "env": [{"key": "PUBLIC_KEY", "value": PUBKEY}],
                "startSsh": True,
            }
        },
    )["podFindAndDeployOnDemand"]

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(data, indent=2))
    print(f"deployed pod {data['id']}  ({GPU_TYPE})")
    print("run 'python scripts/runpod.py status' in ~30s for the ssh command")


def status() -> None:
    if not STATE.exists():
        raise SystemExit("no pod recorded; run 'up' first")
    pid = json.loads(STATE.read_text())["id"]
    p = gql(
        """
        query Pod($id: String!) {
          pod(input: {podId: $id}) {
            id desiredStatus costPerHr
            runtime { uptimeInSeconds ports { ip publicPort privatePort isIpPublic } }
          }
        }
        """,
        {"id": pid},
    )["pod"]

    print(f"pod {p['id']}  status={p['desiredStatus']}  ${p['costPerHr']}/hr")
    rt = p.get("runtime")
    if not rt:
        print("still provisioning -- no runtime yet, try again shortly")
        return
    print(f"uptime {rt['uptimeInSeconds']}s")
    for port in rt.get("ports") or []:
        if port["privatePort"] == 22 and port["isIpPublic"]:
            print(f"\n  ssh root@{port['ip']} -p {port['publicPort']} -i ~/.ssh/id_ed25519\n")


def down() -> None:
    if not STATE.exists():
        raise SystemExit("no pod recorded")
    pid = json.loads(STATE.read_text())["id"]
    gql("mutation Kill($id: String!) { podTerminate(input: {podId: $id}) }", {"id": pid})
    STATE.unlink()
    print(f"terminated {pid} -- billing stopped")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    {"up": up, "status": status, "down": down}.get(cmd, status)()
