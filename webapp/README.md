# Nine Men's Morris — Public Web App

A small, mobile-friendly web app for playing against (and comparing) trained
Nine Men's Morris neural-network models.

This folder is **self-contained for distribution**: it has its own
[`models/`](models/) directory, its own Flask server, and its own UI. To
share a model with someone, drop the `.pt` file into `webapp/models/` and
send them this folder.

## Quick start

```bash
# 1. From the repo root, install the fastnmm engine (one-time):
pip install -e ./fastnmm

# 2. Install webapp deps:
pip install -r webapp/requirements.txt

# 3. Run:
cd webapp
python app.py
```

Then open <http://localhost:7861> on the host machine, or
<http://YOUR-LAN-IP:7861> from a phone on the same network.

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `NMM_APP_HOST` | `0.0.0.0` | Bind address. Use `127.0.0.1` for local-only. |
| `NMM_APP_PORT` | `7861` | HTTP port. |

## Adding models

Drop any compatible `.pt` checkpoint into [`models/`](models/). The server
re-scans the directory each time `/api/models` is hit (on every
**New game**), so you don't need to restart.

Model files must:
- be a state-dict (or a checkpoint dict with `model_state_dict` inside) for
  the relational `ActorCritic` defined in `src/model.py`, and
- be < 200 MB (oversized files are skipped to avoid bad downloads).

## What's in here

```
webapp/
├── app.py              Flask backend, talks to fastnmm + ActorCritic
├── models/             Drop your .pt models here (own folder, no checkpoints)
├── requirements.txt    Runtime Python deps
├── static/
│   ├── app.js          Single-page client
│   ├── favicon.svg
│   └── style.css       Mobile-first responsive theme
└── templates/
    └── index.html      The page
```

## Game modes

For each side (white / black) you can pick:

- **Human** — you control it with taps/clicks.
- **AI** — neural-network policy from the selected `.pt` model. The
  `Temperature` slider controls sharpness: `0.00` is argmax (strongest but
  predictable), `0.40` is balanced, `1.0`+ is more varied.
- **Minimax** — the fastnmm built-in minimax bot at the chosen depth.
- **Random** — uniformly random legal moves (useful as a sanity check).

The phase badge shows whether the game is in **Placement**, **Movement**, or
**Capture**. During capture, tap the opponent's piece you want to remove.

## Production deployment

The Flask dev server is fine for friends-and-family use. For something more
public, put it behind `gunicorn`:

```bash
pip install gunicorn
cd webapp
gunicorn -w 1 -b 0.0.0.0:7861 app:app
```

Keep `-w 1` — the server keeps in-process game state per request, so a
second worker would race with the first.
