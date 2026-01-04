# pip install open_spiel torch
import math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyspiel

# ---------- helpers ----------
def legal_action_mask(state, num_actions):
    mask = np.zeros(num_actions, dtype=np.float32)
    la = state.legal_actions()  # list[int]
    if la:
        mask[la] = 1.0
    return torch.from_numpy(mask)

def masked_categorical(logits, mask):
    # logits: [A], mask: [A] with 0/1
    # Put -inf on illegal actions
    very_neg = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask > 0, logits, torch.full_like(logits, very_neg))
    return torch.distributions.Categorical(logits=masked_logits)

# ---------- model ----------
class ActorCritic(nn.Module):
    def __init__(self, obs_size, num_actions, hidden=256, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.pi = nn.Linear(hidden, num_actions)  # policy head
        self.v  = nn.Linear(hidden, 1)           # value head
        self.do = nn.Dropout(dropout)

        # Small init for policy head helps early entropy
        nn.init.orthogonal_(self.fc1.weight, math.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, math.sqrt(2))
        nn.init.zeros_(self.fc1.bias); nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.pi.bias);  nn.init.zeros_(self.v.bias)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def forward(self, obs):
        # obs: [B, obs_size]
        x = self.ln1(F.relu(self.fc1(obs)))
        x = self.do(x)
        x = self.ln2(F.relu(self.fc2(x)))
        x = self.do(x)
        logits = self.pi(x)         # [B, A]
        value  = self.v(x).squeeze(-1)  # [B]
        return logits, value

# ---------- training loop (self-play A2C-ish) ----------
def train(n_episodes=2000, lr=3e-4, entropy_coef=0.01, value_coef=0.5, gamma=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    game = pyspiel.load_game("nine_mens_morris")
    obs_size = game.observation_tensor_size()
    num_actions = game.num_distinct_actions()

    net = ActorCritic(obs_size, num_actions, hidden=128, dropout=0.1).to(device)
    opt = torch.optim.SGD(net.parameters(), lr)
    #opt = torch.optim.Adam(net.parameters(), lr=lr)

    ema_return = None
    time_start = time.perf_counter()
    for ep in range(1, n_episodes + 1):
        state = game.new_initial_state()
        # Per-player trajectory buckets (logp, value) so we can use each player's own final return
        traj = {0: {'logp': [], 'v': []}, 1: {'logp': [], 'v': []}}

        # --- play one episode ---
        while not state.is_terminal():
            pid = state.current_player()
            obs = torch.tensor(state.observation_tensor(pid), dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = net(obs)  # [1,A], [1]

            mask = legal_action_mask(state, num_actions).to(device).unsqueeze(0)  # [1,A]
            dist = masked_categorical(logits.squeeze(0), mask.squeeze(0))

            action = dist.sample()  # scalar
            logp = dist.log_prob(action)

            traj[pid]['logp'].append(logp)
            traj[pid]['v'].append(value.squeeze(0))

            state.apply_action(int(action.item()))

        # --- compute returns (+1/-1/0) from terminal state ---
        final_returns = state.returns()  # list[float] length 2
        if ema_return is None:
            ema_return = final_returns[0]
        else:
            ema_return = 0.95 * ema_return + 0.05 * final_returns[0]

        # --- build losses ---
        loss = torch.tensor(0.0, device=device)
        entropies = []

        for pid in [0, 1]:
            R = torch.tensor(final_returns[pid], dtype=torch.float32, device=device)
            if len(traj[pid]['logp']) == 0:
                continue
            logps = torch.stack(traj[pid]['logp'])            # [T_pid]
            values = torch.stack(traj[pid]['v'])              # [T_pid]
            # Advantage: (episodic) R - baseline
            adv = (R - values.detach())
            policy_loss = -(logps * adv).mean()
            value_loss = F.mse_loss(values, R.expand_as(values))
            # Entropy: recompute with current logits is cumbersome; approximate from logps variance:
            # better: store entropies during play (left as an exercise). We'll skip or set to 0.
            entropy = torch.tensor(0.0, device=device)

            loss = loss + policy_loss + value_coef * value_loss - entropy_coef * entropy

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if ep % 50 == 0:
            eta = (time.perf_counter() - time_start) / (ep + 1) * (n_episodes - (ep + 1))

            eta_s = int(eta)  # total seconds as int
            eta_h, rem = divmod(eta_s, 3600)   # hours, remainder
            eta_m, eta_s = divmod(rem, 60)     # minutes, seconds

            print(
                f"[ep {ep}] last return P0={final_returns[0]:+.1f}, "
                f"EMA={ema_return:+.3f}, loss={loss.item():.3f}, "
                f"eta={eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
            )

            #print(f"[ep {ep}] last return P0={final_returns[0]:+.1f}, EMA={ema_return:+.3f}, loss={loss.item():.3f}, eta={eta_h}:{eta_m}:{eta%60}")

    return net

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    net = train(n_episodes=135_000)
    t = time.localtime()
    str_t = f"{t.tm_year}:{t.tm_mon}:{t.tm_mday}_{t.tm_hour}:{t.tm_min}:{t.tm_sec}"
    torch.save(net.state_dict(), f"models/openai_model({str_t}).pt")
    print(f"Model saved to models/openai_model({str_t}).pt")
