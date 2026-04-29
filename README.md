# HW3：DQN 及其變體

深度強化學習作業，在 Gridworld 環境上實作 DQN 各種變體，以互動式 Streamlit 應用程式呈現。

## 遊戲環境

**Gridworld 4×4** — 格狀導航任務，Agent 學習抵達目標並避開陷阱。

```
+ = 目標  (reward +10)
- = 陷阱  (reward −10)
W = 牆壁
P = 玩家  (每步 reward −1)
```

| 模式 | 說明 |
|------|------|
| `static` | 固定佈局 — P 在右上，+ 在左上 |
| `player` | 玩家位置隨機，其餘固定 |
| `random` | 所有元素（P、+、−、W）全部隨機擺放 |

---

## 專案結構

```
HW3/
├── app.py                          # Streamlit 首頁導覽
├── pages/
│   ├── 1_hw3_1_naive_dqn.py       # HW3-1：Naive DQN + Experience Replay
│   ├── 2_hw3_2_dqn_variants.py    # HW3-2：Double DQN vs Dueling DQN
│   └── 3_hw3_3_lightning.py       # HW3-3：PyTorch Lightning DQN
├── src/
│   ├── models.py                   # DQN、DoubleDQN、DuelingDQN 模型定義
│   └── trainer.py                  # 訓練邏輯（generator 形式，供 page 1、2 使用）
├── Gridworld.py                    # 遊戲環境
└── GridBoard.py                    # 棋盤渲染
```

---

## 環境安裝

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch streamlit pytorch-lightning numpy matplotlib
```

下載遊戲環境檔案：
```bash
python3 -c "
import urllib.request, ssl
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
for fname, url in [
    ('Gridworld.py', 'https://raw.githubusercontent.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/master/Errata/Gridworld.py'),
    ('GridBoard.py',  'https://raw.githubusercontent.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/master/Errata/GridBoard.py'),
]:
    open(fname, 'wb').write(opener.open(url).read())
"
```

啟動應用程式：
```bash
venv/bin/streamlit run app.py
```

---

## HW3-1：Naive DQN — Static Mode（30%）

### 網路架構

3 層全連接網路（MLP）：

```
輸入層 (64) → 隱藏層 (150) → 隱藏層 (100) → 輸出層 (4)
```

- **輸入**：4×4×4 棋盤狀態攤平為 64 維向量（加入小量雜訊以利探索）
- **輸出**：4 個動作（上/下/左/右）各自的 Q 值

### Naive DQN 運作流程

每一步：

1. 以 **ε-greedy** 策略選擇動作（ε 從 1.0 線性遞減到 0.1）
2. 執行動作，觀察回饋值 `r` 和下一個狀態 `s'`
3. 計算目標 Q 值：`Y = r + γ · max Q(s')`（若遊戲結束則 `Y = r`）
4. 計算損失並反向傳播：`loss = MSE(Q(s, a), Y)`

**問題**：每次更新只使用當前這一筆資料，樣本之間高度相關，訓練不穩定、樣本效率低。

### Experience Replay（改進）

將每筆轉移 `(s, a, r, s', done)` 存入**回放緩衝區**（deque，容量 1000）。  
每步從緩衝區隨機抽取 **mini-batch（200 筆）** 進行梯度更新。

改進效果：
- 打破時間上的相關性，讓訓練更穩定
- 每筆經驗可被重複使用，提升樣本效率
- 減少訓練過程中的 loss 震盪

**預計訓練時間**：約 30 秒（1000 epochs，static mode）

---

## HW3-2：Enhanced DQN Variants — Player Mode（40%）

### Double DQN

**Vanilla DQN 的問題：Q 值高估（Overestimation）**

Vanilla DQN 計算目標 Q 值時：

```python
Y = r + γ · max_a Q_target(s', a)
```

同一個網路既**選擇**動作又**評估**該動作的 Q 值，這會系統性地高估 Q 值，導致 Agent 過於樂觀、策略不穩定。

**Double DQN 修正：將選擇與評估分開**

```python
# Vanilla DQN：
Y = r + γ · max_a Q_target(s', a)

# Double DQN：
a* = argmax_a Q_online(s', a)    # 用 online network 選動作
Y  = r + γ · Q_target(s', a*)   # 用 target network 評估 Q 值
```

只改一行，就能有效降低 Q 值高估的偏差。

### Dueling DQN

**Vanilla DQN 的問題：狀態價值與動作優勢耦合**

對很多狀態來說，**「在這個狀態有多好」**遠比**「選哪個動作比較好」**更重要。  
Vanilla DQN 必須對每一個 (state, action) 組合獨立學習 Q 值，效率較低。

**Dueling DQN 修正：將 Q 值拆解為兩個分支**

```
Q(s, a) = V(s) + A(s, a) − mean_a A(s, a)
```

- **V(s)**：這個狀態本身有多好（State Value）
- **A(s, a)**：在這個狀態下，動作 a 比平均好多少（Advantage）
- 減去 `mean(A)` 是為了讓分解具有唯一性（identifiability）

**網路架構**：
```
輸入 (64)
    ↓
共享層（64 → 150 → 100）
    ↓               ↓
value_stream    advantage_stream
  (100 → 1)        (100 → 4)
    ↓               ↓
       Q = V + (A − mean(A))
```

Agent 可以獨立學習「哪些狀態是有價值的」，不需要對每個動作都有足夠的探索，特別適合動作選擇影響不大的狀態。

app 會同時訓練兩個變體，並顯示平滑後的 loss 對比圖。

**預計訓練時間**：每個變體約 2 分鐘（2000 epochs，player mode）

---

## HW3-3：PyTorch Lightning — Random Mode（30%）

將 Vanilla DQN 的訓練迴圈改寫為 `LightningModule`，並加入多項訓練穩定化技巧。

### PyTorch vs PyTorch Lightning 對照

| Vanilla PyTorch | PyTorch Lightning |
|---|---|
| 手動 `zero_grad / backward / step` | `manual_backward()` + `opt.step()` |
| 手動管理 LR scheduler | `configure_optimizers()` 統一設定 |
| 手動管理裝置（CPU/GPU） | `accelerator="cpu"` 指定 |
| 自己寫訓練迴圈 | `training_step()` 封裝邏輯 |

### 訓練技巧

**Gradient Clipping（梯度裁剪）**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
在每次 optimizer.step() 前，將梯度的 L2 norm 限制在指定值內。  
防止訓練初期 Q 值估計不穩定時發生梯度爆炸。可透過側邊欄調整裁剪值。

**Cosine Annealing LR（餘弦退火學習率）**
```python
CosineAnnealingLR(optimizer, T_max=total_epochs)
```
學習率依餘弦曲線從初始值平滑衰減至接近 0。  
訓練前期步伐大、後期微調，比固定學習率通常能收斂到更好的結果。

**Target Network（目標網路）**  
保留一份凍結的網路副本，每 `sync_freq` 步（預設 500）才同步一次參數。  
避免目標 Q 值每步都在移動，解決「致命三角」（deadly triad）不穩定問題。

**Double DQN 目標**  
Online network 選動作、target network 評估 Q 值，減少高估偏差（同 HW3-2）。

### 實作說明

使用 `automatic_optimization = False`（手動優化模式），讓完整的 episode 收集迴圈可以在 `training_step` 內執行，Lightning 的 LR scheduler 和參數追蹤仍正常運作。  
用 dummy `TensorDataset` 驅動 epoch 迴圈。

> 注意：此模型（25K 參數）在 Apple Silicon MPS 上的 dispatch overhead 大於計算本身，  
> 因此強制使用 `accelerator="cpu"`，速度約快 9 倍。

**預計訓練時間**：約 3 分鐘（1000 epochs，random mode，CPU）

---

## 參考資料

- Mnih et al. (2015) — [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- van Hasselt et al. (2016) — [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- Wang et al. (2016) — [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
- Zai & Brown (2020) — *Deep Reinforcement Learning in Action*，第 3 章
