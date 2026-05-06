"""
HW3-3: PyTorch Lightning DQN — Random Mode
Converts the DQN training loop to a LightningModule and adds:
  - Gradient clipping
  - Learning rate scheduling (CosineAnnealingLR)
  - Target network sync
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import streamlit as st
import matplotlib.pyplot as plt

from Gridworld import Gridworld
from src.models import DQN, make_target_network
from src.trainer import test_model, evaluate_model, ACTION_SET

st.set_page_config(page_title="HW3-3 Lightning DQN", layout="wide")
st.title("HW3-3: PyTorch Lightning DQN — Random Mode")

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
st.sidebar.success(f"Using device: {device}")

# ── Architecture explanation ─────────────────────────────────────────────────
with st.expander("PyTorch Lightning conversion explained", expanded=False):
    st.markdown("""
### What changes when using PyTorch Lightning

| Vanilla PyTorch | PyTorch Lightning |
|---|---|
| Manual training loop | `training_step()` method |
| Manual `optimizer.zero_grad()` / `.backward()` / `.step()` | Handled by Trainer |
| Manual gradient clipping | `gradient_clip_val=1.0` in Trainer |
| Manual LR scheduling | Return scheduler from `configure_optimizers()` |
| Manual device management | `Trainer(accelerator='auto')` |

### Training techniques added
- **Gradient Clipping** (`gradient_clip_val=1.0`): Prevents exploding gradients by
  capping the gradient norm. Especially useful in early training when Q-values are
  noisy.
- **Cosine Annealing LR**: Smoothly decays the learning rate following a cosine
  curve, allowing large updates early and fine-tuning later.
- **Target Network**: Separate frozen network for computing Q-targets, synced every
  `sync_freq` steps to break the deadly triad (bootstrapping + off-policy + function
  approximation).

### Code structure
```
DQNLightning(LightningModule)
├── __init__       — model, target, replay buffer, epsilon
├── forward        — pass through online network
├── training_step  — sample minibatch, compute Double-DQN loss
├── configure_optimizers — Adam + CosineAnnealingLR
└── on_train_epoch_end   — sync target network, decay epsilon
```
""")

# ── Lightning Module (manual_optimization) ───────────────────────────────────
# We use manual_optimization so that:
#   - We control the full episode/step loop inside training_step
#   - Lightning still handles gradient_clip_val and LR scheduling
#   - A single "epoch" = one full Gridworld episode
class DQNLightning(pl.LightningModule):
    def __init__(self, lr=1e-3, gamma=0.9, epsilon_start=1.0,
                 mem_size=1000, batch_size=200, sync_freq=500,
                 max_moves=50, mode='random', total_epochs=2000, grad_clip=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # manual_optimization mode

        self.online  = DQN()
        self.target  = make_target_network(self.online)
        self.replay  = deque(maxlen=mem_size)
        self.epsilon = epsilon_start
        self._step   = 0
        self._losses = []

    def forward(self, x):
        return self.online(x)

    def _get_state(self, game):
        arr = game.board.render_np().reshape(1, 64).astype(np.float32)
        arr += np.random.rand(1, 64).astype(np.float32) / 100.0
        return torch.from_numpy(arr)

    def training_step(self, batch, batch_idx):
        """One epoch = one full episode with multiple gradient updates."""
        opt = self.optimizers()
        sch = self.lr_schedulers()

        game   = Gridworld(size=4, mode=self.hparams.mode)
        state1 = self._get_state(game)
        mov    = 0
        loss   = torch.tensor(0.0)

        # Collect one full episode of experience
        while True:
            mov += 1
            self._step += 1
            qval    = self.online(state1)
            action_ = np.random.randint(0, 4) if random.random() < self.epsilon \
                      else int(qval.data.numpy().argmax())
            game.makeMove(ACTION_SET[action_])
            state2 = self._get_state(game)
            reward = game.reward()
            done   = reward != -1
            self.replay.append((state1, action_, reward, state2, done))
            state1 = state2
            if done or mov > self.hparams.max_moves:
                break

        # One gradient update per episode (same cadence as trainer.py)
        if len(self.replay) >= self.hparams.batch_size:
            mb   = random.sample(self.replay, self.hparams.batch_size)
            s1_b = torch.cat([s for s, *_ in mb])
            a_b  = torch.tensor([a for _, a, *_ in mb], dtype=torch.long)
            r_b  = torch.tensor([r for _, _, r, *_ in mb], dtype=torch.float)
            s2_b = torch.cat([s2 for _, _, _, s2, _ in mb])
            d_b  = torch.tensor([d for *_, d in mb], dtype=torch.float)

            Q1 = self.online(s1_b)
            with torch.no_grad():
                best_a = self.online(s2_b).argmax(dim=1, keepdim=True)
                Q2     = self.target(s2_b).gather(1, best_a).squeeze()
            Y    = r_b + self.hparams.gamma * (1 - d_b) * Q2
            X    = Q1.gather(1, a_b.unsqueeze(1)).squeeze()
            loss = nn.MSELoss()(X, Y.detach())

            opt.zero_grad()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(
                self.online.parameters(), self.hparams.grad_clip
            )
            opt.step()
            self._losses.append(loss.item())

            if self._step % self.hparams.sync_freq == 0:
                self.target.load_state_dict(self.online.state_dict())

        # Step LR scheduler once per epoch
        if sch is not None:
            sch.step()

        # Decay epsilon
        if self.epsilon > 0.1:
            self.epsilon -= 1.0 / self.hparams.total_epochs

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.online.parameters(),
                                     lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.total_epochs
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def train_dataloader(self):
        # Dummy single-item loader — real data comes from the env inside training_step
        from torch.utils.data import DataLoader, TensorDataset
        dummy = TensorDataset(torch.zeros(self.hparams.total_epochs, 1))
        return DataLoader(dummy, batch_size=1)


# ── Streamlit callback for live updates ──────────────────────────────────────
class StreamlitCallback(pl.Callback):
    def __init__(self, total_epochs, progress_bar, status_text, chart_placeholder):
        self.total   = total_epochs
        self.pbar    = progress_bar
        self.status  = status_text
        self.chart   = chart_placeholder

    def on_train_epoch_end(self, trainer, pl_module):
        epoch  = trainer.current_epoch
        losses = pl_module._losses
        frac   = (epoch + 1) / self.total
        self.pbar.progress(min(frac, 1.0))

        if losses:
            last_loss = losses[-1]
            self.status.text(
                f"Epoch {epoch+1}/{self.total}  Loss: {last_loss:.4f}"
                f"  ε: {pl_module.epsilon:.3f}"
                f"  LR: {trainer.optimizers[0].param_groups[0]['lr']:.2e}"
            )

        if (epoch + 1) % max(1, self.total // 50) == 0 and len(losses) > 1:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(losses, linewidth=0.8, color="#A44CE8")
            ax.set_xlabel("Update Step")
            ax.set_ylabel("Loss")
            ax.set_title("Lightning DQN — Training Loss")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            self.chart.pyplot(fig)
            plt.close(fig)


# ── Sidebar config ────────────────────────────────────────────────────────────
st.sidebar.header("Hyperparameters")
epochs      = st.sidebar.slider("Epochs", 100, 800, 500, step=100)
gamma       = st.sidebar.slider("Gamma (γ)", 0.5, 0.99, 0.9)
lr          = st.sidebar.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)
sync_freq   = st.sidebar.slider("Target Sync Freq", 100, 1000, 500, step=100)
grad_clip   = st.sidebar.slider("Gradient Clip Value", 0.1, 5.0, 1.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("**Training Techniques**")
st.sidebar.markdown("✅ Gradient Clipping")
st.sidebar.markdown("✅ Cosine Annealing LR")
st.sidebar.markdown("✅ Target Network")
st.sidebar.markdown("✅ Experience Replay")
st.sidebar.markdown("✅ Double DQN target")

force_retrain = st.sidebar.checkbox("Force Retrain (Ignore Cache)", value=False)

def get_model_cache_path():
    return os.path.join(os.path.dirname(__file__), "..", "models_cache", f"hw3_lightning_{epochs}.pt")

# ── Train ─────────────────────────────────────────────────────────────────────
if st.button("Load / Start Training"):
    cache_path = get_model_cache_path()
    
    lightning_model = DQNLightning(
        lr=lr, gamma=gamma, sync_freq=sync_freq,
        total_epochs=epochs, mode='random', grad_clip=grad_clip
    )
    
    if os.path.exists(cache_path) and not force_retrain:
        st.info("Loading cached model...")
        checkpoint = torch.load(cache_path, map_location=device, weights_only=True)
        lightning_model.load_state_dict(checkpoint["model_state_dict"])
        losses = checkpoint["loss_history"]
        
        # Override the module's loss to maintain UI compatibility
        lightning_model._losses = losses
        st.success("Loaded PyTorch Lightning model from cache perfectly!")
    else:
        st.warning("Training PyTorch Lightning model from scratch...")
        progress   = st.progress(0)
        status     = st.empty()
        chart      = st.empty()

        cb = StreamlitCallback(epochs, progress, status, chart)

        # Let Lightning use the requested device if possible
        accelerator = "gpu" if device.type in ["cuda", "mps"] else "cpu"
        # However, for this tiny model with manual_optimization, CPU is often faster
        
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="cpu",   # MPS dispatch overhead > gain for this tiny model
            callbacks=[cb],
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        with st.spinner("Training via Lightning..."):
            trainer.fit(lightning_model)

        losses = lightning_model._losses
        final_loss = losses[-1] if losses else float('nan')
        st.success(f"Training complete! Final loss: {final_loss:.4f}")
        
        checkpoint = {
            "model_state_dict": lightning_model.state_dict(),
            "loss_history": losses
        }
        torch.save(checkpoint, cache_path)

    st.session_state["hw3_model"] = lightning_model

if "hw3_model" in st.session_state:
    lightning_model = st.session_state["hw3_model"]
    losses = lightning_model._losses

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses, linewidth=0.7, color="#A44CE8", alpha=0.6, label="raw")
    import numpy as np
    if len(losses) > 50:
        smoothed = np.convolve(losses, np.ones(50)/50, mode='valid')
        ax.plot(smoothed, linewidth=1.5, color="#6B1FBF", label="smoothed (50)")
    ax.set_xlabel("Update Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Lightning DQN — random mode, min({len(losses)}) updates, clip={grad_clip}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Test ──────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Model Evaluation & Visualization")
import time

if "hw3_model" in st.session_state:
    col1, col2 = st.columns(2)
    with col1:
        test_mode = st.selectbox("Visualize 1 game & Evaluate on:", ["static", "player", "random"])
    with col2:
        n_eval_games = st.slider("Episodes for Evaluation", 10, 100, 20, step=10)

    if st.button("Watch Game & Evaluate"):
        entry = st.session_state.get("hw3_model")
        model = entry.online
        
        # Ensure device compatibility
        model = model.to(device)
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
                        
            ax.set_title(f"Lightning DQN — Step {step_idx}")
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

        st.divider()
        st.subheader(f"Evaluation Results ({test_mode} mode)")
        with st.spinner("Running evaluation episodes..."):
            win_rate, avg_reward = evaluate_model(model, mode=test_mode, episodes=n_eval_games, device=device)
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Win Rate", f"{win_rate * 100:.1f} %")
        m_col2.metric("Avg Reward", f"{avg_reward:.1f}")

else:
    st.info("Train or load a PyTorch Lightning model first.")
