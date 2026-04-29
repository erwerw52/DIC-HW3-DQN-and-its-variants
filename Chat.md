# HW3 開發對話紀錄

---

## 對話 1

**User:**
📘 Homework 3: DQN and its variants

Total: 100%

1. 📂 Setup & Reference

Base your work on the DRL in Action (English) GitHub repo:
🔗 https://github.com/DeepReinforcementLearning/DeepReinforcementLearningInAction/tree/master
Use the updated starter code provided by the instructor as your baseline.

2. 🧠 HW3-1: Naive DQN for static mode[30%]

✅ Run the provided code naive or Experience buffer reply
💬 Chat with ChatGPT about the code to clarify your understanding
📝 Submit a short understanding report
Includes:
Basic DQN implementation for an easy environment
Experience Replay Buffer

3. ⚖️ HW3-2: Enhanced DQN Variants for player mode [40%]

Implement and compare the following:
Double DQN
Dueling DQN
💡 Focus on how they improve upon the basic DQN approach

4. 🔁 HW3-3: Enhance DQN for random mode WITH Training Tips [30%]

Convert the DQN model from PyTorch to either:
Keras, or
PyTorch Lightning
Bonus points for integrating training techniques to stabilize/improve learning
(e.g., gradient clipping, learning rate scheduling, etc.)

幫我確認一下需求，並列出你的想法

**Assistant:**
讓我先看看當前工作目錄的內容。

（檢視 HW1、HW2 目錄後）

目前 HW1 和 HW2 都是空的，HW3 資料夾也還沒建立。以下是我對作業需求的確認與想法：

### 需求確認

**HW3-1: Naive DQN (Static Mode) — 30%**
- 跑提供的 Naive DQN 或 Experience Replay Buffer 版本
- 撰寫理解報告（不需要實作新東西，主要是理解現有 code）
- 環境：static mode（固定地圖）

**HW3-2: Enhanced DQN Variants (Player Mode) — 40%**
- 實作並比較兩個變體：
  - Double DQN：用兩個網路分離 action 選擇與 Q 值評估，減少高估問題
  - Dueling DQN：將 Q(s,a) 分解為 V(s) + A(s,a)，更高效學習 state value
- 環境：player mode

**HW3-3: Enhanced DQN (Random Mode) — 30%**
- 將 PyTorch 模型轉換成 Keras 或 PyTorch Lightning
- 環境：random mode（隨機地圖，難度最高）
- Bonus：加入訓練技巧（Gradient clipping、Learning rate scheduling 等）

### 關鍵問題需要你確認：
1. starter code 在哪？
2. 環境是什麼遊戲？
3. 理解報告格式要求？中文還是英文？

### 實作規劃（待確認 starter code 後）：

```
HW3/
├── hw3_1_naive_dqn.ipynb
├── hw3_2_double_dqn.ipynb
├── hw3_2_dueling_dqn.ipynb
├── hw3_3_pytorch_lightning.ipynb
└── report.md
```

請提供 starter code，我可以馬上開始分析並規劃實作細節。

---

## 對話 2

**User:**
我把它放在 @HW3/第3章程式_ALL_IN_ONE (1).ipynb，你看一下

**Assistant:**
讀取 notebook 後確認內容：

**環境：Gridworld (4x4 格狀遊戲)**
- `+` = 目標（reward +10）
- `-` = 陷阱（reward -10）
- `W` = 牆壁
- `P` = 玩家
- 每步 reward = -1，最多走 50 步
- 三種模式：static、player、random

**Starter Code 包含的程式：**

| 程式 | 內容 |
|------|------|
| 3.2 | Naive DQN 模型定義（3層 MLP） |
| 3.3 | Naive DQN 訓練迴圈（static mode） |
| 3.4 | 測試函式 test_model() |
| 3.5 | Experience Replay DQN（random mode） |
| 3.7 | Target Network 模型定義 |
| 3.8 | Experience Replay + Target Network（random mode） |
| 3.5改 | 加入撞牆懲罰（-5）的改良版 |

**對應作業的規劃：**

- HW3-1：直接使用程式 3.2 + 3.3，重點在理解報告
- HW3-2：以程式 3.8 為基礎，實作 Double DQN 和 Dueling DQN
- HW3-3：把程式 3.8 包裝成 LightningModule

---

## 對話 3

**User:**
可以用 python 寫，這個應該不用跑在 colab，streamlit 可以呈現嗎？

**Assistant:**
確認本地環境只有 numpy，建立 venv 並安裝套件：

```bash
pip install torch streamlit pytorch-lightning numpy matplotlib
```

**Streamlit 完全可以呈現，而且效果很好：**

可以做到的：
- 即時顯示訓練 loss 曲線
- 用進度條顯示訓練進度
- 側邊欄切換 HW3-1 / 3-2 / 3-3
- 訓練完後動態展示 Gridworld 棋盤走步過程
- 比較 Double DQN vs Dueling DQN 的 loss 曲線

**建議的架構：**

```
HW3/
├── venv/
├── app.py
├── pages/
│   ├── 1_hw3_1_naive_dqn.py
│   ├── 2_hw3_2_dqn_variants.py
│   └── 3_hw3_3_lightning.py
├── src/
│   ├── gridworld.py
│   ├── models.py
│   └── trainer.py
└── README.md
```

---

## 對話 4

**User:**
請實作

**Assistant:**
完成所有實作，Smoke test 通過。

**建立的檔案：**

```
HW3/
├── app.py                          # 首頁導覽
├── pages/
│   ├── 1_hw3_1_naive_dqn.py       # HW3-1: Naive DQN + Experience Replay
│   ├── 2_hw3_2_dqn_variants.py    # HW3-2: Double DQN vs Dueling DQN
│   └── 3_hw3_3_lightning.py       # HW3-3: PyTorch Lightning
├── src/
│   ├── models.py                   # DQN, DoubleDQN, DuelingDQN
│   └── trainer.py                  # 所有訓練邏輯（generator 形式）
├── Gridworld.py / GridBoard.py
└── venv/
```

啟動方式：
```bash
cd HW3
venv/bin/streamlit run app.py
```

---

## 對話 5

**User:**
IndexError: list index out of range
Traceback:
File ".../pages/3_hw3_3_lightning.py", line 261, in <module>
    st.success(f"Training complete! Final loss: {losses[-1]:.4f}")

**Assistant:**
找到三個問題並修復：

1. **`losses[-1]` IndexError** — training_step 在 buffer 未滿時回傳 None，losses 是空的。改用 manual_optimization 模式，在 training_step 內自己跑完整 episode loop。

2. **`gradient_clip_val` 與 `manual_optimization` 衝突** — Lightning 不允許同時使用，改成手動呼叫 `torch.nn.utils.clip_grad_norm_()`。

3. **MPS device mismatch** — Apple Silicon GPU 把 model 移上 MPS，但 numpy 產生的 state tensor 在 CPU。加上 `.to(self.device)` 和 replay buffer 存 `.cpu()` 解決。

---

## 對話 6

**User:**
他算很久誒，你幫我確認一下

**Assistant:**
發現兩個效能問題並修復：

**根本原因 1：MPS overhead**
- 模型只有 25K 參數，MPS GPU dispatch overhead 反而比計算本身還慢
- 修正：`accelerator="cpu"`，速度快 9 倍（163s → 17s per 100 epochs）

**根本原因 2：每步都訓練**
- 原本每個 episode 的每一步都做一次 minibatch 更新
- 修正：改成每個 episode 結束後只更新一次，與 trainer.py 邏輯一致

**訓練時間改善後：**
- HW3-1 Naive DQN (1000 epochs)：約 30 秒
- HW3-2 Double/Dueling DQN (2000 epochs)：約 2 分鐘
- HW3-3 Lightning DQN (1000 epochs)：約 3 分鐘

---

## 對話 7

**User:**
幫我寫 README.md

**Assistant:**
建立英文版 README.md，包含：
- Setup 指令
- 三個 HW 的理論說明
- 程式碼對照（Double DQN 的一行差異、Dueling 的架構圖）
- Lightning 的對照表
- 預計訓練時間
- 參考文獻

---

## 對話 8

**User:**
我要中文的

**Assistant:**
將 README.md 全部改為中文，內容完全對應。

---

## 對話 9

**User:**
幫我推到 github https://github.com/erwerw52/DIC-HW3-DQN-and-its-variants

**Assistant:**
建立 .gitignore，初始化 git repo，推送到 GitHub：

```bash
git init
git remote add origin https://github.com/erwerw52/DIC-HW3-DQN-and-its-variants.git
git add .
git commit -m "HW3: DQN and variants on Gridworld with Streamlit UI"
git push -u origin main
```

---

## 對話 10

**User:**
我有新的 demo site domain https://dic-hw3-dqn-and-its-variants-q9emxf7c4ec6bem2ykavsn.streamlit.app/ 幫我補到 README.md

**Assistant:**
在 README.md 頂部加入 demo 連結並推上 GitHub。

同時偵測到 app.py 有異動（移除各頁面的 Weight 標示），一併 commit。

---

## 對話 11

**User:**
ModuleNotFoundError: This app has encountered an error.
File ".../pages/1_hw3_1_naive_dqn.py", line 9, in <module>
    import matplotlib.pyplot as plt

這個原因是什麼，是不是要給我 requirements.txt?

**Assistant:**
對，Streamlit Cloud 需要 requirements.txt 才知道要裝什麼套件。

建立並推送：

```
torch==2.11.0
pytorch-lightning==2.6.1
streamlit==1.57.0
numpy==2.4.4
matplotlib==3.10.9
```

---

## 對話 12

**User:**
把我們的對話完整的放到 Chat.md

**Assistant:**
（本文件）
