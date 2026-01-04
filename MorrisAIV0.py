from open_spiel.python.examples.play_via_console_example import play_game
from torch import nn
import torch.nn.functional as F
import pyspiel
from typing import Tuple
import numpy as np
import torch

GAME = pyspiel.load_game("nine_mens_morris")
NUM_ACTIONS = GAME.num_distinct_actions()
OBS_SIZE = GAME.observation_tensor_size()
device = torch.device("cuda")


class MorrisNNV0(nn.Module):
    def __init__(self, obs_size=OBS_SIZE, num_actions=NUM_ACTIONS, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.pi  = nn.Linear(hidden, num_actions)  # policy logits
        self.v   = nn.Linear(hidden, 1)            # state value

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        logits = self.pi(x)           # [B, NUM_ACTIONS]
        value  = self.v(x).squeeze(-1)  # [B]
        return logits, value

def get_state(s: pyspiel.State) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Turn an OpenSpiel State into (obs, legal_mask, current_player).

    obs:         float32 array of shape (OBS_SIZE,)
                 == GAME.observation_tensor_size()
    legal_mask:  float32 array of shape (NUM_ACTIONS,)
                 1.0 for legal actions, 0.0 for illegal
    current_player: int in {0, 1}
    """
    # Observation is always from the current player's POV
    current_player = s.current_player()
    print(current_player)
    if current_player < 0:
        print(s.to_string())
    obs = np.asarray(
        s.observation_tensor(current_player),
        dtype=np.float32,
    )

    # Build a global action mask over [0, NUM_ACTIONS)
    legal_mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    legal_actions = s.legal_actions()  # list[int]
    if legal_actions:
        legal_mask[legal_actions] = 1.0

    return obs, legal_mask, current_player

def train(epochs, batch_size, model, target_model, max_moves):
    for epoch in range(epochs):
        s = GAME.new_initial_state()
        num_moves = 0
        while not s.is_terminal() or num_moves < max_moves:
            num_moves += 1
            obs, mask, pid = get_state(s)
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)  # [1, OBS_SIZE]
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)  # [1, NUM_ACTIONS]

            logits, value = model(obs_t)  # now this matches the forward signature

            very_neg = torch.finfo(logits.dtype).min
            masked_logits = torch.where(mask_t > 0, logits, torch.full_like(logits, very_neg))
            dist = torch.distributions.Categorical(logits=masked_logits.squeeze(0))
            action = dist.sample()
            s.apply_action(int(action.item()))
            #print(int(action.item()))

def main():
    net = MorrisNNV0().to(device)
    target_net = MorrisNNV0().to(device)
    train(1, 1, net, target_net, 400)


if __name__ == "__main__":
    main()
