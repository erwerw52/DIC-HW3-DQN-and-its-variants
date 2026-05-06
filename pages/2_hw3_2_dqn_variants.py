"""
HW3-2: Double DQN vs Dueling DQN (Player Mode)
Side-by-side comparison of both variants.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import matplotlib.pyplot as plt
import torch
import numpy as np

from src.models import DoubleDQN, DuelingDQN, make_target_network
from src.trainer import train_double_dqn, train_dueling_dqn, test_model

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

st.set_page_config(page_title="HW3-2 DQN Variants", layout="wide")
st.title("HW3-2: Double DQN vs Dueling DQN — Player Mode")

st.sidebar.success(f"Using device: {device}")

# ── Architecture explanation ────────────────────────────────────────────────
with st.expander("How Double DQN differs from Vanilla DQN", expanded=False):
    st.markdown("""
### Problem: Q-value overestimation
Vanilla DQN computes the target using:
```
Y = r + γ · max_a Q_target(s', a)
```
The same network both **selects** and **evaluates** the best action → systematically
overestimates Q-values.

### Double DQN fix (one-line change)
```python
# Vanilla:
best_Q = target_net(s').max(dim=1)[0]

# Double DQN:
best_a  = online_net(s').argmax(dim=1)   # online selects action
best_Q  = target_net(s').gather(1, best_a)  # target evaluates it
```
The two networks decorrelate selection from evaluation, reducing overestimation.
""")

with st.expander("How Dueling DQN differs from Vanilla DQN", expanded=False):
    st.markdown("""
### Problem: Unnecessary state-action coupling
For many states, the choice of action barely matters — the value of the state
itself dominates. Vanilla DQN must learn Q(s,a) for every (state, action) pair.

### Dueling DQN: split Q into V + A
```
Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)
```
- **V(s)** — value of being in state s (shared backbone)
- **A(s, a)** — advantage of taking action a over the average

The network learns *which states are valuable* independently from
*which actions are better*, leading to faster and more stable learning,
especially when most actions don't change the outcome much.

**Architecture difference**:
```
shared layers → ┬→ value_stream     → V(s)  [1 output]
                └→ advantage_stream → A(s,a) [4 outputs]
                         ↓
                  Q = V + (A - mean(A))
```
""")

# ── Sidebar config ──────────────────────────────────────────────────────────
st.sidebar.header("Hyperparameters")
epochs     = st.sidebar.slider("Epochs", 100, 800, 500, step=100)
gamma      = st.sidebar.slider("Gamma (γ)", 0.5, 0.99, 0.9)
lr         = st.sidebar.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
sync_freq  = st.sidebar.slider("Target Sync Freq (steps)", 100, 1000, 500, step=100)
force_retrain = st.sidebar.checkbox("Force Retrain (Ignore Cache)", value=False)
run_both   = st.sidebar.checkbox("Train both simultaneously", value=True)
variant    = st.sidebar.radio("Or train one:", ["Double DQN", "Dueling DQN"],
                               disabled=run_both)

def get_model_cache_path(variant_name):
    clean_name = variant_name.replace(" ", "_").lower()
    return os.path.join(os.path.dirname(__file__), "..", "models_cache", f"hw2_{clean_name}_{epochs}.pt")

# ── Train ────────────────────────────────────────────────────────────────────
if st.button("Load / Start Training"):
    variants_to_run = (["Double DQN", "Dueling DQN"] if run_both
                       else [variant])

    for v in variants_to_run:
        cache_path = get_model_cache_path(v)
        st.subheader(f"Training: {v}")
        
        if v == "Double DQN":
            online = DoubleDQN().to(device)
            target = make_target_network(online).to(device)
        else:
            online = DuelingDQN().to(device)
            target = make_target_network(online).to(device)
            
        if os.path.exists(cache_path) and not force_retrain:
            st.info("Loading cached model...")
            checkpoint = torch.load(cache_path, map_location=device, weights_only=True)
            online.load_state_dict(checkpoint["model_state_dict"])
            losses = checkpoint["loss_history"]
            st.success("Loaded model from cache perfectly!")
            
            key = "hw2_double" if v == "Double DQN" else "hw2_dueling"
            st.session_state[key] = (online, losses)
            continue
            
        st.warning("Training from scratch...")
        if v == "Double DQN":
            trainer = train_double_dqn(online, target, epochs=epochs,
                                       gamma=gamma, lr=lr, sync_freq=sync_freq,
                                       mode='player', device=device)
        else:
            online = DuelingDQN().to(device)
            target = make_target_network(online).to(device)
            trainer = train_dueling_dqn(online, target, epochs=epochs,
                                        gamma=gamma, lr=lr, sync_freq=sync_freq,
                                        mode='player', device=device)

        progress = st.progress(0)
        status   = st.empty()
        chart    = st.empty()
        losses   = []
        rewards  = []

        for epoch, loss, all_losses, all_rewards in trainer:
            losses = all_losses
            rewards = all_rewards
            frac   = (epoch + 1) / epochs
            progress.progress(frac)
            status.text(f"Epoch {epoch+1}/{epochs}  Loss: {loss:.4f}")

            if (epoch + 1) % max(1, epochs // 10) == 0:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
                color = "#E8784C" if v == "Double DQN" else "#4CE89B"
                ax1.plot(losses, linewidth=0.8, color=color)
                ax1.set_title(f"{v} — Training Loss")
                ax2.plot(rewards, linewidth=0.8, color="green")
                ax2.set_title(f"{v} — Episode Reward")
                chart.pyplot(fig)
                plt.close(fig)

        st.success(f"{v} done! Final loss: {losses[-1]:.4f}")
        
        checkpoint = {
            "model_state_dict": online.state_dict(),
            "loss_history": losses,
            "rewards_history": rewards
        }
        torch.save(checkpoint, cache_path)

        key = "hw2_double" if v == "Double DQN" else "hw2_dueling"
        st.session_state[key] = (online, losses)

# ── Side-by-side comparison chart if both trained ──────────────────────
if run_both and "hw2_double" in st.session_state and "hw2_dueling" in st.session_state:
    st.divider()
    st.subheader("Comparison: Double DQN vs Dueling DQN")
    _, d_losses = st.session_state["hw2_double"]
    _, u_losses = st.session_state["hw2_dueling"]

    # smooth with rolling mean
    def smooth(x, w=50):
        return np.convolve(x, np.ones(w)/w, mode='valid')

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(smooth(d_losses), label="Double DQN", color="#E8784C", linewidth=1.2)
    ax.plot(smooth(u_losses), label="Dueling DQN", color="#4CE89B", linewidth=1.2)
    ax.set_xlabel("Episode (smoothed)")
    ax.set_ylabel("Loss")
    ax.set_title("Double DQN vs Dueling DQN — Loss Comparison (player mode)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Test ─────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Model Evaluation & Visualization")
import time
from src.trainer import evaluate_model

if "hw2_double" in st.session_state or "hw2_dueling" in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        test_mode = st.selectbox("Visualize 1 game & Evaluate on:", ["static", "player", "random"])
    with col2:
        n_eval_games = st.slider("Episodes for Evaluation", 10, 100, 20, step=10)

    if st.button("Watch Game & Evaluate"):
        available_models = []
        if "hw2_double" in st.session_state: available_models.append(("Double DQN", st.session_state["hw2_double"][0]))
        if "hw2_dueling" in st.session_state: available_models.append(("Dueling DQN", st.session_state["hw2_dueling"][0]))
        
        for m_name, model in available_models:
            st.divider()
            st.markdown(f"### Testing {m_name}")
            win, steps, boards, q_vals = test_model(model, mode=test_mode, device=device)
            
            st.write(f"**Game Result:** {'🏆 Won' if win else '💀 Lost/Timeout'}")
            
            col_board, col_q = st.columns([1, 1])
            board_placeholder = col_board.empty()
            q_placeholder = col_q.empty()
            
            if len(q_vals) < len(boards):
                q_vals.insert(0, [0.0, 0.0, 0.0, 0.0])
                
            for step_idx in range(len(boards)):
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.set_xlim(-0.5, 3.5)
                ax.set_ylim(-0.5, 3.5)
                ax.set_xticks(np.arange(-0.5, 4, 1))
                ax.set_yticks(np.arange(-0.5, 4, 1))
                ax.grid(color='black', linestyle='-', linewidth=2)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                
                board_grid = boards[step_idx]
                for row in range(4):
                    for col in range(4):
                        cell_item = board_grid[row][col]
                        plt_y = 3 - row
                        plt_x = col
                        if cell_item == 'P':
                            ax.text(plt_x, plt_y, '🤖', fontsize=40, ha='center', va='center')
                        elif cell_item == '+':
                            ax.text(plt_x, plt_y, '🏆', fontsize=40, ha='center', va='center')
                        elif cell_item == '-':
                            ax.text(plt_x, plt_y, '🔥', fontsize=40, ha='center', va='center')
                        elif cell_item == 'W':
                            ax.add_patch(plt.Rectangle((plt_x - 0.5, plt_y - 0.5), 1, 1, color='gray'))
                            
                ax.set_title(f"{m_name} — Step {step_idx}")
                board_placeholder.pyplot(fig)
                plt.close(fig)
                
                if step_idx > 0 or len(q_vals) > 0:
                    fig2, ax2 = plt.subplots(figsize=(4, 4))
                    actions = ['Up', 'Down', 'Left', 'Right']
                    current_q = q_vals[step_idx]
                    colors = ['#4C9BE8' if i != np.argmax(current_q) else '#FF6B6B' for i in range(4)]
                    ax2.bar(actions, current_q, color=colors)
                    ax2.set_title(f"Q-values (Move {step_idx})")
                    ax2.set_ylim(min(current_q) - 1, max(current_q) + 1)
                    q_placeholder.pyplot(fig2)
                    plt.close(fig2)
                    
                time.sleep(0.5)

            st.subheader(f"Evaluation Results ({test_mode} mode)")
            with st.spinner(f"Running {m_name} evaluation episodes..."):
                win_rate, avg_reward = evaluate_model(model, mode=test_mode, episodes=n_eval_games, device=device)
            
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Win Rate", f"{win_rate * 100:.1f} %")
            m_col2.metric("Avg Reward", f"{avg_reward:.1f}")

else:
    st.info("Train or load a model first.")
