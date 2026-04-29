"""
HW3 DQN Variants — Main Streamlit App
Navigate via the sidebar to each homework section.
"""
import streamlit as st

st.set_page_config(
    page_title="HW3: DQN and Variants",
    page_icon="🧠",
    layout="wide",
)

st.title("HW3: DQN and its Variants")
st.markdown("**Deep Reinforcement Learning — Gridworld (4×4)**")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("HW3-1 · Naive DQN")
    st.markdown("""
**Mode**: Static / Random

- Basic DQN training loop
- ε-greedy exploration
- Experience Replay Buffer
- Visualize loss curve & test win rate
""")
    st.page_link("pages/1_hw3_1_naive_dqn.py", label="Go to HW3-1 →")

with col2:
    st.subheader("HW3-2 · Enhanced DQN Variants")
    st.markdown("""
**Mode**: Player

- **Double DQN** — decouple action selection from Q evaluation to reduce overestimation
- **Dueling DQN** — split Q into V(s) + A(s,a) for better state-value learning
- Side-by-side loss comparison chart
""")
    st.page_link("pages/2_hw3_2_dqn_variants.py", label="Go to HW3-2 →")

with col3:
    st.subheader("HW3-3 · PyTorch Lightning")
    st.markdown("""
**Mode**: Random

- Full DQN converted to `LightningModule`
- **Gradient Clipping** (configurable)
- **Cosine Annealing LR** scheduler
- Target network + Double DQN target
- Live loss + ε + LR monitoring
""")
    st.page_link("pages/3_hw3_3_lightning.py", label="Go to HW3-3 →")

st.divider()
st.subheader("Environment: Gridworld 4×4")
st.markdown("""
```
+ = Goal   (reward +10)
- = Trap   (reward −10)
W = Wall
P = Player (reward −1 per step)
```

**Modes**
| Mode | Description |
|------|-------------|
| `static` | Fixed layout — P top-right, + top-left |
| `player` | Player starts at random position |
| `random` | All pieces (P, +, −, W) placed randomly |
""")
