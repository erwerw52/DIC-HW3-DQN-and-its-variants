"""
HW3-1: Naive DQN (Static Mode)
Demonstrates basic DQN and Experience Replay on Gridworld.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import matplotlib.pyplot as plt
import torch

from src.models import DQN
from src.trainer import train_naive_dqn, train_replay_dqn, test_model

st.set_page_config(page_title="HW3-1 Naive DQN", layout="wide")
st.title("HW3-1: Naive DQN — Static Mode")

# ── Sidebar config ──────────────────────────────────────────────────────────
st.sidebar.header("Hyperparameters")
variant  = st.sidebar.radio("Variant", ["Naive DQN", "Experience Replay DQN"])
epochs   = st.sidebar.slider("Epochs", 500, 5000, 1000, step=500)
gamma    = st.sidebar.slider("Gamma (γ)", 0.5, 0.99, 0.9)
lr       = st.sidebar.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
mode     = "static" if variant == "Naive DQN" else "random"

# ── Architecture explanation ────────────────────────────────────────────────
with st.expander("Architecture", expanded=False):
    st.markdown("""
**Input**: 4×4×4 board flattened → 64 values (+ small noise)
**Network**: Linear(64→150) → ReLU → Linear(150→100) → ReLU → Linear(100→4)
**Output**: Q-values for 4 actions (up / down / left / right)

### Naive DQN Flow
```
state → Q-network → Q(s,a) for all actions
ε-greedy → pick action → observe reward, next state
target Y = r + γ·max Q(s') if not terminal, else r
loss = MSE(Q(s,a), Y)
```

### Experience Replay (改進)
Instead of updating every step, store transitions `(s, a, r, s', done)` in a
**replay buffer** (deque, size 1000), then sample a random mini-batch of 200
to break temporal correlations and stabilize training.
""")

# ── Train ────────────────────────────────────────────────────────────────────
if st.button("Start Training"):
    model = DQN()
    st.session_state["hw1_model"] = model
    st.session_state["hw1_mode"]  = mode

    progress = st.progress(0)
    status   = st.empty()
    chart    = st.empty()
    losses   = []

    trainer_fn = train_naive_dqn if variant == "Naive DQN" else train_replay_dqn

    for epoch, loss, all_losses in trainer_fn(model, epochs=epochs, gamma=gamma, lr=lr, mode=mode):
        losses = all_losses
        frac   = (epoch + 1) / epochs
        progress.progress(frac)
        status.text(f"Epoch {epoch+1}/{epochs}  Loss: {loss:.4f}")

        if (epoch + 1) % max(1, epochs // 50) == 0:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(losses, linewidth=0.8, color="#4C9BE8")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            chart.pyplot(fig)
            plt.close(fig)

    st.success(f"Training complete! Final loss: {losses[-1]:.4f}")

    # Final loss plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses, linewidth=0.8, color="#4C9BE8")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.set_title(f"{variant} — Training Loss ({mode} mode, {epochs} epochs)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Test ─────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Test Trained Model")

test_mode = st.radio("Test mode", ["static", "player", "random"], horizontal=True)
n_games   = st.slider("Games to test", 10, 500, 100, step=10)

if st.button("Run Test"):
    model = st.session_state.get("hw1_model")
    if model is None:
        st.warning("Please train a model first.")
    else:
        wins = sum(test_model(model, mode=test_mode)[0] for _ in range(n_games))
        win_pct = 100.0 * wins / n_games
        col1, col2, col3 = st.columns(3)
        col1.metric("Games Played", n_games)
        col2.metric("Wins", wins)
        col3.metric("Win Rate", f"{win_pct:.1f}%")

        # Show one game replay
        st.subheader("Sample Game Replay")
        _, steps = test_model(model, mode=test_mode)
        for step in steps:
            st.code(step)
