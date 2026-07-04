#!/usr/bin/env python3
"""System-monitor sidecar for the PPO trainer.

Polls every `--interval` seconds and appends a line to
`logs/system_monitor.log`:

  - GPU: util, mem-used, mem-free, temp, power
  - CPU: load1/5/15, total mem used, swap
  - Trainer: PID, heartbeat age (seconds since last `_heartbeat` write),
    last heartbeat stage, episode, update, phase
  - Worker child count and aggregate RSS
  - eps/s derived from the curriculum CSV (delta over the polling window)

Designed to be left running alongside `python main.py resume`. If the
trainer's heartbeat goes stale for more than `--stale-secs` (default 90s)
the line is tagged `STALE` so a tail can see the hang the moment it
starts — not 30 minutes later when nvidia-smi happens to be checked.

Usage:
  python system_monitor.py [--interval 5] [--stale-secs 90]
  python system_monitor.py --once     # one-shot status print
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

SRC_DIR = Path(__file__).resolve().parent
LOG_DIR = SRC_DIR / "logs"
HEARTBEAT_PATH = LOG_DIR / "trainer_heartbeat.json"
MONITOR_LOG = LOG_DIR / "system_monitor.log"


def _nvidia_query() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.free,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        util, mu, mf, mt, temp, pwr = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return {
            "gpu_util_pct": int(util),
            "gpu_mem_used_mib": int(mu),
            "gpu_mem_free_mib": int(mf),
            "gpu_mem_total_mib": int(mt),
            "gpu_temp_c": int(temp),
            "gpu_power_w": float(pwr),
        }
    except Exception as e:
        return {"gpu_error": str(e)}


def _nvidia_processes() -> list:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        rows = []
        for line in out.strip().splitlines():
            if not line.strip():
                continue
            pid_s, mem_s = [x.strip() for x in line.split(",")]
            rows.append({"pid": int(pid_s), "mem_mib": int(mem_s)})
        return rows
    except Exception:
        return []


def _pid_alive(pid: int) -> bool:
    return Path(f"/proc/{pid}").exists()


def _read_heartbeat() -> Optional[dict]:
    if not HEARTBEAT_PATH.exists():
        return None
    try:
        with open(HEARTBEAT_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def _trainer_pid() -> Optional[int]:
    hb = _read_heartbeat()
    if hb and _pid_alive(hb["pid"]):
        return int(hb["pid"])
    try:
        out = subprocess.check_output(
            ["pgrep", "-fa", "main.py resume"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        for line in out.strip().splitlines():
            pid_s = line.split()[0]
            if pid_s.isdigit():
                return int(pid_s)
    except Exception:
        pass
    return None


def _rss_mib(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 0


def _thread_count(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def _child_pids(parent_pid: int) -> list:
    try:
        out = subprocess.check_output(
            ["pgrep", "-P", str(parent_pid)],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        return [int(p) for p in out.split() if p.isdigit()]
    except Exception:
        return []


def _mem_info() -> dict:
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("MemTotal", "MemFree", "MemAvailable", "SwapTotal", "SwapFree"):
                    info[k] = int(v.strip().split()[0]) // 1024  # MiB
    except Exception:
        pass
    return info


def _loadavg() -> tuple:
    try:
        with open("/proc/loadavg") as f:
            parts = f.read().split()
            return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        return (0.0, 0.0, 0.0)


def _latest_curriculum_csv() -> Optional[Path]:
    files = sorted(LOG_DIR.glob("*_curriculum.csv"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


_eps_state = {"path": None, "episode": None, "ts": None}

# Host PIDs we've seen own GPU memory while the trainer was healthy — kept so
# we don't falsely flag the trainer's own CUDA process as "orphaned" the
# moment its heartbeat goes above the stale threshold (PID-namespace mismatch
# between nvidia-smi and /proc inside the container).
_gpu_pid_baseline: set = set()


def _eps_per_sec() -> Optional[float]:
    csv_path = _latest_curriculum_csv()
    if csv_path is None:
        return None
    try:
        with open(csv_path) as f:
            last_row = None
            for row in csv.DictReader(f):
                last_row = row
            if last_row is None:
                return None
            ep = int(float(last_row["episode"]))
            now = time.time()
            prev_path = _eps_state["path"]
            prev_ep = _eps_state["episode"]
            prev_ts = _eps_state["ts"]
            _eps_state["path"] = str(csv_path)
            _eps_state["episode"] = ep
            _eps_state["ts"] = now
            if prev_path != str(csv_path) or prev_ep is None:
                return None
            dt = now - prev_ts
            if dt < 1.0:
                return None
            return max(0.0, (ep - prev_ep) / dt)
    except Exception:
        return None


def collect(stale_secs: float) -> dict:
    snap = {"ts": time.time()}
    snap.update(_nvidia_query())
    gpu_procs = _nvidia_processes()
    snap["gpu_procs"] = gpu_procs

    mem = _mem_info()
    snap["sys_mem_used_gib"] = round((mem.get("MemTotal", 0) - mem.get("MemAvailable", 0)) / 1024, 1)
    snap["sys_mem_total_gib"] = round(mem.get("MemTotal", 0) / 1024, 1)
    snap["swap_used_gib"] = round((mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)) / 1024, 1)
    snap["loadavg"] = list(_loadavg())

    pid = _trainer_pid()
    snap["trainer_pid"] = pid
    if pid is not None:
        snap["trainer_rss_gib"] = round(_rss_mib(pid) / 1024, 2)
        snap["trainer_threads"] = _thread_count(pid)
        kids = _child_pids(pid)
        snap["worker_count"] = len(kids)
        snap["worker_rss_gib"] = round(sum(_rss_mib(p) for p in kids) / 1024, 2)

    hb = _read_heartbeat()
    if hb is not None:
        snap["heartbeat_age_s"] = round(time.time() - hb["ts"], 1)
        snap["heartbeat_stage"] = hb["stage"]
        snap["heartbeat_episode"] = hb["episode"]
        snap["heartbeat_update"] = hb["update"]
        snap["heartbeat_phase"] = hb["phase"]
        snap["stale"] = snap["heartbeat_age_s"] > stale_secs
    else:
        snap["heartbeat_age_s"] = None
        snap["stale"] = False

    # GPU memory held by a PID that's no longer alive — the smoking gun for
    # the previous crash pattern (process died, CUDA context orphaned).
    #
    # Inside a container, nvidia-smi reports HOST PIDs while /proc uses the
    # container's PID namespace, so an alive trainer can look "orphaned"
    # here even though it's healthy. The trainer's host PID is stable across
    # the run, so we remember it on the first poll with a fresh heartbeat
    # (the only GPU process we've ever spawned) and never flag it again.
    raw_orphans = [p for p in gpu_procs if not _pid_alive(p["pid"])]
    if pid is not None and hb is not None and (
        snap["heartbeat_age_s"] is not None and snap["heartbeat_age_s"] < 30
    ):
        _gpu_pid_baseline.update(p["pid"] for p in gpu_procs)
    snap["orphan_gpu_pids"] = [
        p for p in raw_orphans if p["pid"] not in _gpu_pid_baseline
    ]

    snap["eps_per_sec"] = _eps_per_sec()
    return snap


def fmt(snap: dict) -> str:
    parts = []
    if snap.get("stale"):
        parts.append("STALE")
    if snap.get("orphan_gpu_pids"):
        parts.append(f"ORPHAN_GPU={snap['orphan_gpu_pids']}")
    gpu_util = snap.get("gpu_util_pct")
    if gpu_util is not None:
        parts.append(
            f"GPU={gpu_util}% mem={snap['gpu_mem_used_mib']}/{snap['gpu_mem_total_mib']}MiB "
            f"T={snap['gpu_temp_c']}C P={snap['gpu_power_w']:.0f}W"
        )
    pid = snap.get("trainer_pid")
    if pid is not None:
        parts.append(
            f"trainer pid={pid} rss={snap.get('trainer_rss_gib')}G "
            f"thr={snap.get('trainer_threads')} workers={snap.get('worker_count')} "
            f"worker_rss={snap.get('worker_rss_gib')}G"
        )
    else:
        parts.append("trainer=DOWN")
    if snap.get("heartbeat_age_s") is not None:
        parts.append(
            f"hb_age={snap['heartbeat_age_s']:.0f}s stage={snap.get('heartbeat_stage')} "
            f"ep={snap.get('heartbeat_episode')} upd={snap.get('heartbeat_update')} "
            f"phase={snap.get('heartbeat_phase')}"
        )
    eps = snap.get("eps_per_sec")
    if eps is not None:
        parts.append(f"eps/s={eps:.0f}")
    parts.append(
        f"sysmem={snap['sys_mem_used_gib']}/{snap['sys_mem_total_gib']}G "
        f"swap={snap['swap_used_gib']}G load={snap['loadavg'][0]:.1f}/{snap['loadavg'][1]:.1f}"
    )
    return time.strftime("%H:%M:%S") + " " + " | ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=5.0,
                    help="seconds between samples (default 5)")
    ap.add_argument("--stale-secs", type=float, default=90.0,
                    help="tag a line STALE when heartbeat is older than this (default 90)")
    ap.add_argument("--once", action="store_true",
                    help="print one snapshot to stdout and exit")
    ap.add_argument("--quiet", action="store_true",
                    help="don't print to stdout, only write to logfile")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if args.once:
        snap = collect(args.stale_secs)
        line = fmt(snap)
        print(line)
        return 0

    print(f"# system_monitor writing to {MONITOR_LOG} every {args.interval}s "
          f"(stale_secs={args.stale_secs})")
    with open(MONITOR_LOG, "a") as logf:
        logf.write(f"# === monitor started @ {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        logf.flush()
        while True:
            try:
                snap = collect(args.stale_secs)
                line = fmt(snap)
                logf.write(line + "\n")
                # Also dump the raw JSON every snapshot so we can post-mortem.
                logf.write("  raw=" + json.dumps(snap, default=str) + "\n")
                logf.flush()
                if not args.quiet:
                    print(line)
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\n# monitor stopped")
                return 0
            except Exception as e:
                logf.write(f"# monitor error: {e!r}\n")
                logf.flush()
                time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
