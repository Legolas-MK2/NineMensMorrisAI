import math, time, csv, os
from multiprocessing import Process, Queue, Event
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyspiel

# ---------- helpers ----------
def legal_action_mask(state, num_actions):
    mask = np.zeros(num_actions, dtype=np.float32)
    la = state.legal_actions()
    if la:
        mask[la] = 1.0
    return mask

def masked_categorical(logits, mask):
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
        self.fc3 = nn.Linear(hidden, hidden)
        self.ln3 = nn.LayerNorm(hidden)
        self.pi = nn.Linear(hidden, num_actions)
        self.v = nn.Linear(hidden, 1)
        self.do = nn.Dropout(dropout)

        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, math.sqrt(2))
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)
        nn.init.zeros_(self.pi.bias)
        nn.init.zeros_(self.v.bias)

    def forward(self, obs):
        x = self.do(self.ln1(F.relu(self.fc1(obs))))
        x = self.do(self.ln2(F.relu(self.fc2(x))))
        x = self.do(self.ln3(F.relu(self.fc3(x))))
        return self.pi(x), self.v(x).squeeze(-1)


# ---------- Worker Process ----------
def env_worker(worker_id, obs_queue, action_queue, result_queue, 
               stop_event, num_envs_per_worker, game_name, draw_penalty):
    """
    Worker process that manages its own set of environments.
    Now stores observations, actions, masks for later recomputation.
    """
    game = pyspiel.load_game(game_name)
    num_actions = game.num_distinct_actions()
    
    states = [game.new_initial_state() for _ in range(num_envs_per_worker)]
    
    # Trajectory storage: store raw data for recomputation
    # Each step: {'obs': np.array, 'action': int, 'mask': np.array}
    trajectories = [{0: [], 1: []} for _ in range(num_envs_per_worker)]
    
    while not stop_event.is_set():
        # Collect observations from all non-terminal states
        obs_list = []
        mask_list = []
        env_indices = []
        pids = []
        
        for i, state in enumerate(states):
            if not state.is_terminal():
                pid = state.current_player()
                obs = state.observation_tensor(pid)
                mask = legal_action_mask(state, num_actions)
                obs_list.append(obs)
                mask_list.append(mask)
                env_indices.append(i)
                pids.append(pid)
        
        if not env_indices:
            # All games finished, send results and reset
            for i in range(num_envs_per_worker):
                returns = states[i].returns()
                if returns[0] == 0.0 and returns[1] == 0.0:
                    returns = [draw_penalty, draw_penalty]
                
                result_queue.put({
                    'worker_id': worker_id,
                    'env_id': i,
                    'trajectories': trajectories[i],
                    'returns': returns
                })
                
                states[i] = game.new_initial_state()
                trajectories[i] = {0: [], 1: []}
            continue
        
        # Send observations to main process
        obs_queue.put({
            'worker_id': worker_id,
            'obs': np.array(obs_list, dtype=np.float32),
            'mask': np.array(mask_list, dtype=np.float32),
            'env_indices': env_indices,
            'pids': pids
        })
        
        # Wait for actions
        try:
            action_data = action_queue.get(timeout=30)
        except:
            break
        
        actions = action_data['actions']
        obs_array = action_data['obs']      # Get back the obs we sent
        mask_array = action_data['mask']    # Get back the masks
        
        # Apply actions and store trajectory data for recomputation
        for i, (env_idx, pid) in enumerate(zip(env_indices, pids)):
            # Store data needed for recomputation (numpy arrays)
            trajectories[env_idx][pid].append({
                'obs': obs_array[i],
                'action': int(actions[i]),
                'mask': mask_array[i]
            })
            
            # Step environment
            states[env_idx].apply_action(int(actions[i]))
            
            # Check if game finished
            if states[env_idx].is_terminal():
                returns = states[env_idx].returns()
                if returns[0] == 0.0 and returns[1] == 0.0:
                    returns = [draw_penalty, draw_penalty]
                
                result_queue.put({
                    'worker_id': worker_id,
                    'env_id': env_idx,
                    'trajectories': trajectories[env_idx],
                    'returns': returns
                })
                
                states[env_idx] = game.new_initial_state()
                trajectories[env_idx] = {0: [], 1: []}


def compute_losses_recompute(net, batch_trajectories, batch_returns, device, num_actions):
    """
    Recompute forward pass to get gradients, then compute losses.
    """
    policy_loss = torch.tensor(0.0, device=device)
    value_loss = torch.tensor(0.0, device=device)
    entropy_sum = torch.tensor(0.0, device=device)
    total_steps = 0
    
    for traj, returns in zip(batch_trajectories, batch_returns):
        for pid in [0, 1]:
            steps = traj[pid]
            if not steps:
                continue
            
            R = returns[pid]
            
            # Batch all steps for this player in this episode
            obs_batch = torch.tensor(
                np.array([s['obs'] for s in steps]), 
                dtype=torch.float32, device=device
            )
            mask_batch = torch.tensor(
                np.array([s['mask'] for s in steps]), 
                dtype=torch.float32, device=device
            )
            actions = torch.tensor(
                [s['action'] for s in steps], 
                dtype=torch.long, device=device
            )
            
            # Recompute forward pass WITH gradients
            logits, values = net(obs_batch)
            
            dist = masked_categorical(logits, mask_batch)
            logps = dist.log_prob(actions)
            entropies = dist.entropy()
            
            # Compute losses
            adv = R - values.detach()
            policy_loss += -(logps * adv).mean()
            value_loss += F.mse_loss(values, torch.full_like(values, R))
            entropy_sum += entropies.mean()
            total_steps += len(steps)
    
    return policy_loss, value_loss, entropy_sum, len(batch_trajectories)


# ---------- Main Training Loop ----------
def train(n_episodes=2000,
          lr=3e-4,
          entropy_coef=0.01,
          value_coef=0.5,
          num_workers=8,
          envs_per_worker=64,
          hidden=512,
          dropout=0.1,
          draw_penalty=0.0,
          update_every=256):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_envs = num_workers * envs_per_worker
    print(f"Using device: {device}")
    print(f"Workers: {num_workers}, Envs per worker: {envs_per_worker}, Total: {total_envs}")
    
    game = pyspiel.load_game("nine_mens_morris")
    obs_size = game.observation_tensor_size()
    num_actions = game.num_distinct_actions()
    
    net = ActorCritic(obs_size, num_actions, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    
    # Queues
    obs_queues = {i: Queue(maxsize=2) for i in range(num_workers)}
    action_queues = {i: Queue(maxsize=2) for i in range(num_workers)}
    result_queue = Queue()
    stop_event = Event()
    
    # Start workers
    workers = []
    for i in range(num_workers):
        p = Process(target=env_worker, args=(
            i, obs_queues[i], action_queues[i], result_queue,
            stop_event, envs_per_worker, "nine_mens_morris", draw_penalty
        ))
        p.start()
        workers.append(p)
    
    # Logging
    t = time.localtime()
    str_t = f"{t.tm_year}-{t.tm_mon:02d}-{t.tm_mday:02d}_{t.tm_hour:02d}-{t.tm_min:02d}-{t.tm_sec:02d}"
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/{str_t}.csv"
    log_f = open(log_path, "w", newline="")
    writer = csv.DictWriter(log_f, fieldnames=["ep", "ema_return", "loss", "eps_per_sec", "eta"])
    writer.writeheader()
    
    total_episodes = 0
    ema_return = None
    time_start = time.perf_counter()
    
    batch_trajectories = []
    batch_returns = []
    
    try:
        while total_episodes < n_episodes:
            # Collect observations from all workers
            pending_workers = set(range(num_workers))
            worker_data = {}
            
            while pending_workers:
                for wid in list(pending_workers):
                    try:
                        data = obs_queues[wid].get(timeout=0.01)
                        worker_data[wid] = data
                        pending_workers.remove(wid)
                    except:
                        pass
                
                # Drain results
                while not result_queue.empty():
                    try:
                        result = result_queue.get_nowait()
                        batch_trajectories.append(result['trajectories'])
                        batch_returns.append(result['returns'])
                    except:
                        break
            
            # Batch all observations for GPU inference
            all_obs = []
            all_masks = []
            worker_splits = []
            
            for wid in range(num_workers):
                data = worker_data[wid]
                all_obs.append(data['obs'])
                all_masks.append(data['mask'])
                worker_splits.append({
                    'wid': wid,
                    'size': len(data['obs']),
                    'env_indices': data['env_indices'],
                    'pids': data['pids'],
                    'obs': data['obs'],
                    'mask': data['mask']
                })
            
            # Forward pass for action selection (no grad needed here)
            obs_batch = torch.from_numpy(np.concatenate(all_obs)).to(device)
            mask_batch = torch.from_numpy(np.concatenate(all_masks)).to(device)
            
            with torch.no_grad():
                logits, values = net(obs_batch)
                dist = masked_categorical(logits, mask_batch)
                actions = dist.sample()
            
            # Send actions back to workers (with obs/mask for storage)
            offset = 0
            for split in worker_splits:
                size = split['size']
                wid = split['wid']
                
                action_queues[wid].put({
                    'actions': actions[offset:offset+size].cpu().numpy(),
                    'obs': split['obs'],      # Send back for storage
                    'mask': split['mask']     # Send back for storage
                })
                offset += size
            
            # Drain results
            while not result_queue.empty():
                try:
                    result = result_queue.get_nowait()
                    batch_trajectories.append(result['trajectories'])
                    batch_returns.append(result['returns'])
                except:
                    break
            
            # Update when batch is ready
            if len(batch_trajectories) >= update_every:
                # Recompute forward pass WITH gradients and compute losses
                policy_loss, value_loss, entropy_sum, num_eps = compute_losses_recompute(
                    net, batch_trajectories, batch_returns, device, num_actions
                )
                
                total_episodes += num_eps
                
                total_loss = (policy_loss / num_eps + 
                             value_coef * value_loss / num_eps - 
                             entropy_coef * entropy_sum / num_eps)
                
                opt.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                
                # EMA update
                last_return = batch_returns[-1][0]
                if ema_return is None:
                    ema_return = last_return
                else:
                    ema_return = 0.95 * ema_return + 0.05 * last_return
                
                # Logging
                elapsed = time.perf_counter() - time_start
                eps_per_sec = total_episodes / elapsed
                eta_sec = int((n_episodes - total_episodes) / eps_per_sec) if eps_per_sec > 0 else 0
                eta_h, rem = divmod(eta_sec, 3600)
                eta_m, eta_s = divmod(rem, 60)
                
                print(f"[ep {total_episodes:,}] EMA={ema_return:+.3f}, "
                      f"loss={total_loss.item():.3f}, eps/s={eps_per_sec:.1f}, "
                      f"eta={eta_h:02d}:{eta_m:02d}:{eta_s:02d}")
                
                writer.writerow({
                    "ep": total_episodes,
                    "ema_return": float(ema_return),
                    "loss": float(total_loss.item()),
                    "eps_per_sec": float(eps_per_sec),
                    "eta": f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
                })
                log_f.flush()
                
                batch_trajectories = []
                batch_returns = []
    
    finally:
        stop_event.set()
        for p in workers:
            p.terminate()
            p.join(timeout=1)
        log_f.close()
    
    return net, str_t


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    net, str_t = train(
        n_episodes=10_000_000,
        lr=3e-4,
        entropy_coef=0.01,
        value_coef=0.5,
        num_workers=32,
        envs_per_worker=32,
        hidden=512,
        dropout=0.1,
        draw_penalty=-0.3,
        update_every=512,
    )
    
    save_path = f"models/nmm_a2c_{str_t}.pt"
    torch.save(net.state_dict(), save_path)
    print(f"Model saved to {save_path}")