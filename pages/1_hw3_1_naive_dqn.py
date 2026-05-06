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
from src.trainer import train_naive_dqn, train_replay_dqn, evaluate_model, test_model

st.set_page_config(page_title="HW3-1 Naive DQN", layout="wide")
st.title("HW3-1: Naive DQN — Static Mode")

# Auto Detect Device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
st.sidebar.success(f"Using device: {device}")

# ── Sidebar config ──────────────────────────────────────────────────────────
st.sidebar.header("Hyperparameters")
variant  = st.sidebar.radio("Variant", ["Naive DQN", "Experience Replay DQN"])
epochs   = st.sidebar.slider("Epochs", 100, 800, 500, step=100) # Default to 500
gamma    = st.sidebar.slider("Gamma (γ)", 0.5, 0.99, 0.9)
lr       = st.sidebar.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
force_retrain = st.sidebar.checkbox("Force Retrain (Ignore Cache)", value=False)
mode     = "static" if variant == "Naive DQN" else "random"

# ── Architecture explanation ────────────────────────────────────────────────
with st.expander("Architecture & Understanding Report", expanded=False):
    st.markdown("""
### Short Understanding Report

**1. Environment Difficulty**
The provided Gridworld environment (especially `random` mode) requires the agent to navigate to find the `+` goal while avoiding the `-` pit and `W` wall. In `static` mode, the goal and obstacles are fixed, making it quite easy. In `random` mode, the start and objects change randomly, which is much harder to generalize.

**2. Training Instability Symptoms & Why Current Mechanisms Fail**
Without Experience Replay buffer, the sequence of states observed is highly correlated. The agent tends to "forget" past lessons or overfit to a particular trajectory. As a result, Naive DQN often diverges, especially in the `random` environment.

**3. Why Experience Replay Solves the Problem**
Experience Replay stores transition tuples `(s, a, r, s')`. By randomly sampling mini-batches, it:
1. **Breaks temporal correlations** in the data.
2. **Reuses past experiences**, ensuring rare successes or failures aren't forgotten.
3. Stabilizes gradient updates.
""")

def get_model_cache_path(variant_name):
    clean_name = variant_name.replace(" ", "_").lower()
    return os.path.join(os.path.dirname(__file__), "..", "models_cache", f"hw1_{clean_name}_{epochs}.pt")

# ── Train ────────────────────────────────────────────────────────────────────
if st.button("Load / Start Training"):
    cache_path = get_model_cache_path(variant)
    
    model = DQN().to(device)
    
    if os.path.exists(cache_path) and not force_retrain:
        st.info("Loading cached model...")
        checkpoint = torch.load(cache_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        losses = checkpoint["loss_history"]
        rewards_list = checkpoint["rewards_history"]
        st.success("Loaded model from cache perfectly!")
    else:
        st.warning("Training from scratch...")
        progress = st.progress(0)
        status   = st.empty()
        chart    = st.empty()
        
        trainer_fn = train_naive_dqn if variant == "Naive DQN" else train_replay_dqn
        
        for epoch, loss, losses, rewards_list in trainer_fn(model, epochs=epochs, gamma=gamma, lr=lr, mode=mode, device=device):
            frac   = (epoch + 1) / epochs
            progress.progress(frac)
            status.text(f"Epoch {epoch+1}/{epochs}  Loss: {loss:.4f}")

            if (epoch + 1) % max(1, epochs // 10) == 0: 
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
                ax1.plot(losses, linewidth=0.8, color="#4C9BE8")
                ax1.set_title("Training Loss")
                ax2.plot(rewards_list, linewidth=0.8, color="green")
                ax2.set_title("Episode Reward")
                chart.pyplot(fig)
                plt.close(fig)

        st.success(f"Training complete! Final loss: {losses[-1]:.4f}")
        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "loss_history": losses,
            "rewards_history": rewards_list
        }
        torch.save(checkpoint, cache_path)

    st.session_state["hw1_model"] = model
    st.session_state["hw1_mode"]  = mode
    st.session_state["hw1_losses"] = losses
    st.session_state["hw1_rewards"] = rewards_list
    st.session_state["hw1_variant"] = variant
    st.session_state["hw1_epochs"] = epochs

if "hw1_losses" in st.session_state and "hw1_rewards" in st.session_state:
    loss_data = st.session_state["hw1_losses"]
    reward_data = st.session_state["hw1_rewards"]
    v_name = st.session_state["hw1_variant"]
    v_mode = st.session_state["hw1_mode"]
    v_epochs = st.session_state["hw1_epochs"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(loss_data, linewidth=0.8, color="#4C9BE8")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{v_name} — Loss ({v_mode}, {v_epochs} ep)")
    ax1.grid(True, alpha=0.3)
    
    import numpy as np
    smoothed_rewards = np.convolve(reward_data, np.ones(10)/10, mode='valid') if len(reward_data) > 10 else reward_data
    ax2.plot(smoothed_rewards, linewidth=1.2, color="green")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward (Smoothed)")
    ax2.set_title(f"{v_name} — Reward ({v_mode})")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Test ─────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Model Evaluation & Visualization")

if "hw1_model" in st.session_state:
    import time

    col1, col2 = st.columns(2)
    with col1:
        test_mode = st.selectbox("Visualize 1 game & Evaluate on:", ["static", "player", "random"])
    with col2:
        n_eval_games = st.slider("Episodes for Evaluation", 10, 100, 20, step=10)

    if st.button("Watch Game & Evaluate"):
        model = st.session_state["hw1_model"]
        win, steps, boards, q_vals = test_model(model, mode=test_mode, device=device)
        
        st.write(f"**Game Result:** {'🏆 Won' if win else '💀 Lost/Timeout'}")
        
        col_board, col_q = st.columns([1, 1])
        board_placeholder = col_board.empty()
        q_placeholder = col_q.empty()
        
        # Pre-pad Q_vals with zero array for the initial state before move
        if len(q_vals) < len(boards):
            q_vals.insert(0, [0.0, 0.0, 0.0, 0.0])
            
        for step_idx in range(len(boards)):
            # 1. Render Fancy Board
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
                    # Y is inverted in matplotlib (0 is bottom), so 3 - row
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
                        
            ax.set_title(f"Step {step_idx}")
            board_placeholder.pyplot(fig)
            plt.close(fig)
            
            # 2. Render Q-Value Bar Chart for this step
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

        st.divider()
        st.subheader(f"Evaluation Results ({test_mode} mode)")
        with st.spinner("Running evaluation episodes..."):
            win_rate, avg_reward = evaluate_model(model, mode=test_mode, episodes=n_eval_games, device=device)
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Win Rate", f"{win_rate * 100:.1f} %")
        m_col2.metric("Avg Reward", f"{avg_reward:.1f}")

else:
    st.info("Train or load a model first.")
