"""
Training logic shared across HW3-1, 3-2, 3-3.
Each trainer returns a generator that yields (epoch, loss) so Streamlit
can update the UI incrementally.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import random
from collections import deque

import numpy as np
import torch

from Gridworld import Gridworld

ACTION_SET = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
MOVE_POS   = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_state(game, device="cpu"):
    return torch.from_numpy(
        game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 100.0
    ).float().to(device)



# ──────────────────────────────────────────────
# HW3-1  Naive DQN  (no replay, static mode)
# ──────────────────────────────────────────────
def train_naive_dqn(model, epochs=1000, gamma=0.9, lr=1e-3, mode='static', device="cpu"):
    loss_fn   = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epsilon   = 1.0
    losses    = []
    rewards_list = []

    for i in range(epochs):
        game   = Gridworld(size=4, mode=mode)
        state1 = _get_state(game, device)
        status = 1
        episode_reward = 0.0
        loss   = torch.tensor(0.0)

        while status == 1:
            qval  = model(state1)
            qval_ = qval.data.numpy()
            action_ = np.random.randint(0, 4) if random.random() < epsilon \
                      else int(np.argmax(qval_))
            game.makeMove(ACTION_SET[action_])
            state2 = _get_state(game, device)
            reward = game.reward()
            episode_reward += reward

            with torch.no_grad():
                newQ = model(state2)
            maxQ = torch.max(newQ)
            Y = torch.tensor([reward + gamma * maxQ]).to(device) if reward == -1 \
                else torch.tensor([float(reward)]).to(device)
            Y = Y.detach()
            X = qval.squeeze()[action_]

            loss = loss_fn(X, Y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            state1 = state2

            if abs(reward) == 10:
                status = 0

        losses.append(loss.item() if hasattr(loss, "item") else loss)
        rewards_list.append(episode_reward)
        if epsilon > 0.1:
            epsilon -= 1.0 / epochs

        yield i, (loss.item() if hasattr(loss, "item") else loss), losses[:], rewards_list[:]


# ──────────────────────────────────────────────
# HW3-1  Experience Replay DQN  (random mode)
# ──────────────────────────────────────────────
def train_replay_dqn(model, epochs=5000, gamma=0.9, lr=1e-3,
                     mem_size=1000, batch_size=200, max_moves=50, mode='random', device="cpu"):
    loss_fn   = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epsilon   = 1.0
    replay    = deque(maxlen=mem_size)
    losses    = []
    rewards_list = []
    loss      = torch.tensor(0.0).to(device)

    for i in range(epochs):
        game   = Gridworld(size=4, mode=mode)
        state1 = _get_state(game, device)
        status = 1
        episode_reward = 0.0
        mov    = 0

        while status == 1:
            mov += 1
            qval    = model(state1)
            qval_   = qval.data.numpy()
            action_ = np.random.randint(0, 4) if random.random() < epsilon \
                      else int(np.argmax(qval_))
            game.makeMove(ACTION_SET[action_])
            state2 = _get_state(game, device)
            reward = game.reward()
            episode_reward += reward
            done   = reward != -1
            replay.append((state1, action_, reward, state2, done))
            state1 = state2

            if len(replay) > batch_size:
                mb    = random.sample(replay, batch_size)
                s1_b  = torch.cat([s for s, *_ in mb])
                a_b   = torch.tensor([a for _, a, *_ in mb], dtype=torch.long).to(device)
                r_b   = torch.tensor([r for _, _, r, *_ in mb], dtype=torch.float).to(device)
                s2_b  = torch.cat([s2 for _, _, _, s2, _ in mb])
                d_b   = torch.tensor([d for *_, d in mb], dtype=torch.float).to(device)

                Q1 = model(s1_b)
                with torch.no_grad():
                    Q2 = model(s2_b)
                Y  = r_b + gamma * (1 - d_b) * torch.max(Q2, dim=1)[0]
                X  = Q1.gather(1, a_b.unsqueeze(1)).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if abs(reward) == 10 or mov > max_moves:
                status = 0

        losses.append(loss.item() if hasattr(loss, "item") else loss)
        rewards_list.append(episode_reward)
        if epsilon > 0.1:
            epsilon -= 1.0 / epochs

        yield i, (loss.item() if hasattr(loss, "item") else loss), losses[:], rewards_list[:]


# ──────────────────────────────────────────────
# HW3-2  Double DQN  (player mode)
# Key change: online net selects action, target net evaluates Q
# ──────────────────────────────────────────────
def train_double_dqn(online, target, epochs=5000, gamma=0.9, lr=1e-3,
                     mem_size=1000, batch_size=200, max_moves=50,
                     sync_freq=500, mode='player', device="cpu"):
    loss_fn   = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(online.parameters(), lr=lr)
    epsilon   = 1.0
    replay    = deque(maxlen=mem_size)
    losses    = []
    rewards_list = []
    loss      = torch.tensor(0.0).to(device)
    j         = 0

    for i in range(epochs):
        game   = Gridworld(size=4, mode=mode)
        state1 = _get_state(game, device)
        status = 1
        episode_reward = 0.0
        mov    = 0

        while status == 1:
            j   += 1
            mov += 1
            qval    = online(state1)
            qval_   = qval.data.numpy()
            action_ = np.random.randint(0, 4) if random.random() < epsilon \
                      else int(np.argmax(qval_))
            game.makeMove(ACTION_SET[action_])
            state2 = _get_state(game, device)
            reward = game.reward()
            episode_reward += reward
            done   = reward != -1
            replay.append((state1, action_, reward, state2, done))
            state1 = state2

            if len(replay) > batch_size:
                mb   = random.sample(replay, batch_size)
                s1_b = torch.cat([s for s, *_ in mb])
                a_b  = torch.tensor([a for _, a, *_ in mb], dtype=torch.long).to(device)
                r_b  = torch.tensor([r for _, _, r, *_ in mb], dtype=torch.float).to(device)
                s2_b = torch.cat([s2 for _, _, _, s2, _ in mb])
                d_b  = torch.tensor([d for *_, d in mb], dtype=torch.float).to(device)

                Q1 = online(s1_b)
                with torch.no_grad():
                    # Double DQN: online picks best action, target evaluates it
                    best_actions = online(s2_b).argmax(dim=1, keepdim=True)
                    Q2_target    = target(s2_b).gather(1, best_actions).squeeze()
                Y  = r_b + gamma * (1 - d_b) * Q2_target
                X  = Q1.gather(1, a_b.unsqueeze(1)).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if j % sync_freq == 0:
                    target.load_state_dict(online.state_dict())

            if abs(reward) == 10 or mov > max_moves:
                status = 0

        losses.append(loss.item() if hasattr(loss, "item") else loss)
        rewards_list.append(episode_reward)
        if epsilon > 0.1:
            epsilon -= 1.0 / epochs

        yield i, (loss.item() if hasattr(loss, "item") else loss), losses[:], rewards_list[:]


# ──────────────────────────────────────────────
# HW3-2  Dueling DQN  (player mode)
# Architecture change only; training loop same as replay DQN + target net
# ──────────────────────────────────────────────
def train_dueling_dqn(online, target, epochs=5000, gamma=0.9, lr=1e-3,
                      mem_size=1000, batch_size=200, max_moves=50,
                      sync_freq=500, mode='player', device="cpu"):
    loss_fn   = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(online.parameters(), lr=lr)
    epsilon   = 1.0
    replay    = deque(maxlen=mem_size)
    losses    = []
    rewards_list = []
    loss      = torch.tensor(0.0).to(device)
    j         = 0

    for i in range(epochs):
        game   = Gridworld(size=4, mode=mode)
        state1 = _get_state(game, device)
        status = 1
        episode_reward = 0.0
        mov    = 0

        while status == 1:
            j   += 1
            mov += 1
            qval    = online(state1)
            qval_   = qval.data.numpy()
            action_ = np.random.randint(0, 4) if random.random() < epsilon \
                      else int(np.argmax(qval_))
            game.makeMove(ACTION_SET[action_])
            state2 = _get_state(game, device)
            reward = game.reward()
            episode_reward += reward
            done   = reward != -1
            replay.append((state1, action_, reward, state2, done))
            state1 = state2

            if len(replay) > batch_size:
                mb   = random.sample(replay, batch_size)
                s1_b = torch.cat([s for s, *_ in mb])
                a_b  = torch.tensor([a for _, a, *_ in mb], dtype=torch.long).to(device)
                r_b  = torch.tensor([r for _, _, r, *_ in mb], dtype=torch.float).to(device)
                s2_b = torch.cat([s2 for _, _, _, s2, _ in mb])
                d_b  = torch.tensor([d for *_, d in mb], dtype=torch.float).to(device)

                Q1 = online(s1_b)
                with torch.no_grad():
                    Q2 = target(s2_b)
                Y  = r_b + gamma * (1 - d_b) * torch.max(Q2, dim=1)[0]
                X  = Q1.gather(1, a_b.unsqueeze(1)).squeeze()
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if j % sync_freq == 0:
                    target.load_state_dict(online.state_dict())

            if abs(reward) == 10 or mov > max_moves:
                status = 0

        losses.append(loss.item() if hasattr(loss, "item") else loss)
        rewards_list.append(episode_reward)
        if epsilon > 0.1:
            epsilon -= 1.0 / epochs

        yield i, (loss.item() if hasattr(loss, "item") else loss), losses[:], rewards_list[:]


# ──────────────────────────────────────────────
# Test helper
# ──────────────────────────────────────────────
def test_model(model, mode='static', max_steps=15, device="cpu"):
    """Returns (win: bool, steps: list of board strings, boards: list of raw 4x4 char boards, q_vals: list of lists)"""
    game   = Gridworld(size=4, mode=mode)
    state  = _get_state(game, device)
    
    # Store raw board state for visualization
    board_state = game.display()
    boards = [board_state]
    steps  = [_board_str(game)]
    q_vals = []
    
    status = 1
    i      = 0

    while status == 1:
        qval    = model(state)
        # Handle gpu tensors for numpy conversion
        qval_np = qval.detach().cpu().numpy()
        q_vals.append(qval_np.squeeze().tolist())
        action_ = int(qval_np.argmax())
        action  = ACTION_SET[action_]
        game.makeMove(action)
        state  = _get_state(game, device)
        
        board_state = game.display()
        boards.append(board_state)
        
        steps.append(f"Move {i}: {action}\n" + _board_str(game))
        reward = game.reward()
        i += 1

        if reward == 10:
            return True, steps, boards, q_vals
        if reward == -10:
            return False, steps, boards, q_vals
        if i > max_steps:
            return False, steps, boards, q_vals

    return False, steps, boards, q_vals


def _board_str(game):
    return '\n'.join(' '.join(row) for row in game.display())

def evaluate_model(model, mode='static', episodes=10, max_steps=30, device='cpu'):
    model.to(device)
    model.eval()
    wins = 0
    total_rewards = 0.0
    for _ in range(episodes):
        game = Gridworld(size=4, mode=mode)
        state = _get_state(game).to(device)
        status = 1
        i = 0
        episode_reward = 0
        while status == 1 and i < max_steps:
            with torch.no_grad():
                qval = model(state)
            action_ = int(qval.cpu().data.numpy().argmax())
            game.makeMove(ACTION_SET[action_])
            state = _get_state(game).to(device)
            reward = game.reward()
            episode_reward += reward
            i += 1
            if abs(reward) == 10:
                if reward == 10:
                    wins += 1
                status = 0
        total_rewards += episode_reward
    model.train()
    return wins / episodes, total_rewards / episodes
