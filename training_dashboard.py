"""
Nine Men's Morris - Training Dashboard
Reads CSV training logs and displays all important metrics visually.
"""

import os
import csv
import json
import logging
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request

logging.basicConfig(
    level=os.environ.get("NMM_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("training_dashboard")

app = Flask(__name__)

LOGS_DIR = Path(__file__).parent / "src" / "logs"

# ─────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────

def get_log_files():
    """Return sorted list of log files (newest first)."""
    if not LOGS_DIR.exists():
        return []
    files = sorted(LOGS_DIR.glob("*_curriculum.csv"), reverse=True)
    return [{"name": f.name, "path": str(f), "size_kb": round(f.stat().st_size / 1024, 1)} for f in files]


def read_log(path: str):
    """Parse a curriculum CSV log into a list of dicts."""
    rows = []
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                cleaned = {}
                for k, v in row.items():
                    try:
                        cleaned[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
                    except (ValueError, TypeError):
                        cleaned[k] = v
                rows.append(cleaned)
    except Exception as e:
        logger.warning("Error reading log: %s", e)
    return rows


def summarise(rows):
    """Build a summary dict from parsed rows."""
    if not rows:
        return {}

    last = rows[-1]
    first = rows[0]

    # Find phase transitions
    phase_transitions = []
    prev_phase = None
    for r in rows:
        p = r.get("phase")
        if p != prev_phase:
            phase_transitions.append({"phase": p, "episode": r.get("episode"), "starting_stones": r.get("starting_stones")})
            prev_phase = p

    # Find milestones where minimax depth was first beaten
    depth_milestones = {}
    for r in rows:
        d = r.get("minimax_depth_beaten", 0)
        if d not in depth_milestones:
            depth_milestones[d] = r.get("episode")

    # Find clone generation transitions
    clone_gen_transitions = []
    prev_clone_gen = None
    for r in rows:
        cg = r.get("clone_gen", 0)
        if cg != prev_clone_gen:
            clone_gen_transitions.append({"clone_gen": cg, "episode": r.get("episode"), "index": len(clone_gen_transitions)})
            prev_clone_gen = cg

    # Best win rates
    best_wr_mm3 = max((r.get("wr_vs_mm_d3", 0) or 0 for r in rows), default=0)
    best_wr_mm4 = max((r.get("wr_vs_mm_d4", 0) or 0 for r in rows), default=0)
    best_wr_mm5 = max((r.get("wr_vs_mm_d5", 0) or 0 for r in rows), default=0)
    best_wr_mm6 = max((r.get("wr_vs_mm_d6", 0) or 0 for r in rows), default=0)
    best_wr_mm7 = max((r.get("wr_vs_mm_d7", 0) or 0 for r in rows), default=0)

    # Downsample rows for chart (max 400 points to keep response small)
    step = max(1, len(rows) // 400)
    sampled = rows[::step]
    if rows[-1] not in sampled:
        sampled.append(rows[-1])

    return {
        "first_episode": first.get("episode"),
        "last_episode": last.get("episode"),
        "total_steps": last.get("steps"),
        "current_phase": last.get("phase"),
        "current_starting_stones": last.get("starting_stones"),
        "current_wr": last.get("win_rate"),
        "current_dr": last.get("draw_rate"),
        "current_ema": last.get("ema_return"),
        "current_avg_return": last.get("avg_return"),
        "current_lr": last.get("lr"),
        "current_eps_per_sec": last.get("eps_per_sec"),
        "minimax_depth_beaten": last.get("minimax_depth_beaten"),
        "clone_gen": last.get("clone_gen"),
        "active_mm_depth": last.get("active_mm_max_depth"),
        "shaping_mult": last.get("shaping_mult"),
        "wr_vs_mm_d1": last.get("wr_vs_mm_d1"),
        "wr_vs_mm_d2": last.get("wr_vs_mm_d2"),
        "wr_vs_mm_d3": last.get("wr_vs_mm_d3"),
        "wr_vs_mm_d4": last.get("wr_vs_mm_d4"),
        "wr_vs_mm_d5": last.get("wr_vs_mm_d5"),
        "wr_vs_mm_d6": last.get("wr_vs_mm_d6"),
        "wr_vs_mm_d7": last.get("wr_vs_mm_d7"),
        "wr_vs_random": last.get("wr_vs_random"),
        "wr_vs_self": last.get("wr_vs_self"),
        "best_wr_mm3": best_wr_mm3,
        "best_wr_mm4": best_wr_mm4,
        "best_wr_mm5": best_wr_mm5,
        "best_wr_mm6": best_wr_mm6,
        "best_wr_mm7": best_wr_mm7,
        "total_rows": len(rows),
        "phase_transitions": phase_transitions,
        "depth_milestones": depth_milestones,
        "clone_gen_transitions": clone_gen_transitions,
        "sampled_rows": sampled,
    }


# ─────────────────────────────────────────────
# HTML Dashboard
# ─────────────────────────────────────────────

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "dashboard.html"
DASHBOARD_HTML = _TEMPLATE_PATH.read_text(encoding="utf-8")


# ─────────────────────────────────────────────
# API routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return DASHBOARD_HTML


@app.route('/api/logs')
def api_logs():
    return jsonify(get_log_files())


@app.route('/api/log_data')
def api_log_data():
    path = request.args.get('path', '')
    if not path or not Path(path).exists():
        return jsonify({'error': f'File not found: {path}'})
    rows = read_log(path)
    if not rows:
        return jsonify({'error': 'Empty or unreadable log file'})
    summary = summarise(rows)
    # Don't send raw rows to avoid huge payload; sampled_rows is already in summary
    return jsonify({'summary': summary})


if __name__ == '__main__':
    host = os.environ.get('NMM_DASHBOARD_HOST', '0.0.0.0')
    port = int(os.environ.get('NMM_DASHBOARD_PORT', '7861'))
    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = '0.0.0.0'

    print("=" * 60)
    print("Nine Men's Morris — Training Dashboard")
    print("=" * 60)
    log_files = get_log_files()
    print(f"Found {len(log_files)} log file(s) in {LOGS_DIR}")
    for lf in log_files[:5]:
        print(f"  • {lf['name']} ({lf['size_kb']} KB)")
    print()
    print(f"Open http://{local_ip}:{port} in your browser")
    print("=" * 60)

    app.run(host=host, port=port, debug=False)