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
from src.trainer import test_model, ACTION_SET

st.set_page_config(page_title="HW3-3 Lightning DQN", layout="wide")
st.title("HW3-3: PyTorch Lightning DQN — Random Mode")

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
epochs      = st.sidebar.slider("Epochs", 500, 5000, 1000, step=500)
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

# ── Train ─────────────────────────────────────────────────────────────────────
if st.button("Start Training"):
    lightning_model = DQNLightning(
        lr=lr, gamma=gamma, sync_freq=sync_freq,
        total_epochs=epochs, mode='random', grad_clip=grad_clip
    )
    st.session_state["hw3_model"] = lightning_model

    progress   = st.progress(0)
    status     = st.empty()
    chart      = st.empty()

    cb = StreamlitCallback(epochs, progress, status, chart)

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="cpu",   # MPS dispatch overhead > gain for this tiny model
        callbacks=[cb],
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )

    with st.spinner("Training..."):
        trainer.fit(lightning_model)

    losses = lightning_model._losses
    final_loss = losses[-1] if losses else float('nan')
    st.success(f"Training complete! Final loss: {final_loss:.4f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses, linewidth=0.7, color="#A44CE8", alpha=0.6, label="raw")
    import numpy as np
    if len(losses) > 50:
        smoothed = np.convolve(losses, np.ones(50)/50, mode='valid')
        ax.plot(smoothed, linewidth=1.5, color="#6B1FBF", label="smoothed (50)")
    ax.set_xlabel("Update Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"Lightning DQN — random mode, {epochs} epochs, clip={grad_clip}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Test ──────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Test Trained Model")
test_mode = st.radio("Test mode", ["static", "player", "random"], horizontal=True)
n_games   = st.slider("Games to test", 10, 500, 100, step=10)

if st.button("Run Test"):
    entry = st.session_state.get("hw3_model")
    if entry is None:
        st.warning("Please train a model first.")
    else:
        model = entry.online
        wins  = sum(test_model(model, mode=test_mode)[0] for _ in range(n_games))
        win_pct = 100.0 * wins / n_games
        col1, col2, col3 = st.columns(3)
        col1.metric("Games Played", n_games)
        col2.metric("Wins", wins)
        col3.metric("Win Rate", f"{win_pct:.1f}%")

        st.subheader("Sample Game Replay")
        _, steps = test_model(model, mode=test_mode)
        for step in steps:
            st.code(step)
