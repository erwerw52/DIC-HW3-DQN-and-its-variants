"""
HW3-2: Double DQN vs Dueling DQN (Player Mode)
Side-by-side comparison of both variants.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import matplotlib.pyplot as plt
import torch

from src.models import DoubleDQN, DuelingDQN, make_target_network
from src.trainer import train_double_dqn, train_dueling_dqn, test_model

st.set_page_config(page_title="HW3-2 DQN Variants", layout="wide")
st.title("HW3-2: Double DQN vs Dueling DQN — Player Mode")

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
epochs     = st.sidebar.slider("Epochs", 1000, 8000, 2000, step=500)
gamma      = st.sidebar.slider("Gamma (γ)", 0.5, 0.99, 0.9)
lr         = st.sidebar.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
sync_freq  = st.sidebar.slider("Target Sync Freq (steps)", 100, 1000, 500, step=100)
run_both   = st.sidebar.checkbox("Train both simultaneously", value=True)
variant    = st.sidebar.radio("Or train one:", ["Double DQN", "Dueling DQN"],
                               disabled=run_both)

# ── Train ────────────────────────────────────────────────────────────────────
if st.button("Start Training"):
    variants_to_run = (["Double DQN", "Dueling DQN"] if run_both
                       else [variant])

    for v in variants_to_run:
        st.subheader(f"Training: {v}")
        if v == "Double DQN":
            online = DoubleDQN()
            target = make_target_network(online)
            trainer = train_double_dqn(online, target, epochs=epochs,
                                       gamma=gamma, lr=lr, sync_freq=sync_freq,
                                       mode='player')
        else:
            online = DuelingDQN()
            target = make_target_network(online)
            trainer = train_dueling_dqn(online, target, epochs=epochs,
                                        gamma=gamma, lr=lr, sync_freq=sync_freq,
                                        mode='player')

        progress = st.progress(0)
        status   = st.empty()
        chart    = st.empty()
        losses   = []

        for epoch, loss, all_losses in trainer:
            losses = all_losses
            frac   = (epoch + 1) / epochs
            progress.progress(frac)
            status.text(f"Epoch {epoch+1}/{epochs}  Loss: {loss:.4f}")

            if (epoch + 1) % max(1, epochs // 50) == 0:
                fig, ax = plt.subplots(figsize=(8, 3))
                color = "#E8784C" if v == "Double DQN" else "#4CE89B"
                ax.plot(losses, linewidth=0.8, color=color)
                ax.set_xlabel("Episode")
                ax.set_ylabel("Loss")
                ax.set_title(f"{v} — Training Loss")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                chart.pyplot(fig)
                plt.close(fig)

        st.success(f"{v} done! Final loss: {losses[-1]:.4f}")

        key = "hw2_double" if v == "Double DQN" else "hw2_dueling"
        st.session_state[key] = (online, losses)

    # ── Side-by-side comparison chart if both trained ──────────────────────
    if run_both and "hw2_double" in st.session_state and "hw2_dueling" in st.session_state:
        st.divider()
        st.subheader("Comparison: Double DQN vs Dueling DQN")
        _, d_losses = st.session_state["hw2_double"]
        _, u_losses = st.session_state["hw2_dueling"]

        # smooth with rolling mean
        import numpy as np
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
st.subheader("Test Trained Models")
test_mode = st.radio("Test mode", ["static", "player", "random"], horizontal=True)
n_games   = st.slider("Games to test", 10, 500, 100, step=10)

if st.button("Run Test"):
    col1, col2 = st.columns(2)

    for col, key, label in [
        (col1, "hw2_double", "Double DQN"),
        (col2, "hw2_dueling", "Dueling DQN"),
    ]:
        with col:
            st.markdown(f"**{label}**")
            entry = st.session_state.get(key)
            if entry is None:
                st.warning("Not trained yet.")
            else:
                model, _ = entry
                wins = sum(test_model(model, mode=test_mode)[0] for _ in range(n_games))
                win_pct = 100.0 * wins / n_games
                st.metric("Win Rate", f"{win_pct:.1f}%")
                st.metric("Wins", f"{wins}/{n_games}")

                st.markdown("**Sample game:**")
                _, steps = test_model(model, mode=test_mode)
                for step in steps:
                    st.code(step)
