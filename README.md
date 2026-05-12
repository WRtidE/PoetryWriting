# PoetryWriting — 自动写诗

基于 PyTorch 的唐诗自动生成项目，支持**双层 LSTM** 与**双层 GRU** 两种模型架构，在海量唐诗数据上训练，支持**温度采样**和 **Top-K 采样**生成诗句，并提供藏头诗生成功能。

---

## 目录

- [项目概览](#项目概览)
- [目录结构](#目录结构)
- [数据集](#数据集)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
  - [图形界面（推荐）](#图形界面推荐)
  - [训练模型](#训练模型)
  - [命令行生成诗句](#命令行生成诗句)
  - [命令行生成藏头诗](#命令行生成藏头诗)
- [模型架构](#模型架构)
- [配置说明](#配置说明)
- [生成策略](#生成策略)
- [改进方向](#改进方向)
- [License](#license)

---

## 项目概览

本项目实现了一个端到端的唐诗生成系统，包含数据加载、模型训练、诗歌生成三大模块。模型以字符级语言模型的方式逐字生成诗句，支持自由创作和藏头诗两种模式。

| 特性 | 说明 |
|------|------|
| 模型 | 双层 LSTM / 双层 GRU + Embedding + 全连接 |
| 参数 | embedding_dim=256, hidden_dim=512 |
| 数据 | 唐诗数据集，57,580 首，8,293 字 |
| 解码 | 温度采样 + Top-K 采样 |
| 功能 | 续写诗句 / 藏头诗生成 |
| 框架 | PyTorch |

---

## 目录结构

```
PoetryWriting/
├── model/
│   ├── __init__.py      # 模型工厂函数（get_model）
│   ├── model_lstm.py    # 双层 LSTM 模型定义
│   └── model_gru.py     # 双层 GRU 模型定义
├── train.py             # 模型训练脚本（支持 --model lstm/gru）
├── main.py              # PyQt5 图形界面
├── test.py              # 命令行生成（续写 + 藏头诗，支持 --model/--mode）
├── dataset/
│   └── tang.npz         # 已处理的唐诗数据集
├── result/
│   ├── poetry_model_LSTM.pth  # LSTM 预训练模型
│   └── poetry_model_GRU.pth   # GRU 预训练模型
└── README.md
```

| 文件 | 用途 |
|------|------|
| `model/__init__.py` | `get_model('lstm'/'gru')` 工厂函数，统一创建模型 |
| `model/model_lstm.py` | 双层 LSTM：Embedding → LSTM×2 → Linear |
| `model/model_gru.py` | 双层 GRU：Embedding → GRU×2 → Linear |
| `train.py` | 训练流程：`--model lstm/gru` 切换架构，支持自定义 batch_size/lr/epochs |
| `main.py` | **PyQt5 图形界面**，修改 `Config.model_type` 即可切换模型 |
| `test.py` | 命令行生成：`--mode continue` 续写，`--mode acrostic` 藏头诗，`--model lstm/gru` 切换模型 |

---

## 数据集

- **格式**: `.npz` 压缩的 NumPy 数据
- **内容**: 
  - `data`: 已数字化的唐诗序列（共 57,580 首）
  - `ix2word`: 索引→字符的映射字典
  - `word2ix`: 字符→索引的映射字典
- **规模**: 序列长度 125，词汇量 8,293
- **特殊标记**: `<START>`（开始）、`<EOP>`（结束）

---

## 环境依赖

- Python >= 3.8
- PyTorch >= 1.8
- NumPy

```bash
# CPU 版本
pip install torch numpy PyQt5

# GPU 版本（CUDA 11.8）
pip install torch numpy PyQt5 --index-url https://download.pytorch.org/whl/cu118
```

推荐使用 conda 或 venv 创建隔离环境：

```bash
python -m venv poetry_env
source poetry_env/bin/activate  # Linux/macOS
pip install torch numpy PyQt5
```

---

## 快速开始

### 图形界面（推荐）

```bash
python main.py
```

启动后将看到中式风格的 PyQt5 图形界面，包含两个标签页：

| 标签 | 功能 | 可调参数 |
|------|------|----------|
| **续写诗句** | 输入开头文字，生成完整诗句 | 温度、Top-K |
| **藏头诗** | 输入藏头字，生成藏头七言诗 | 温度 |

启动后自动加载模型，加载完成后即可点击按钮生成。生成结果可一键复制到剪贴板。

> 界面采用中国传统配色（宣纸底色、朱砂红点缀），搭配宋体系列字体。

### 训练模型

```bash
# 训练 GRU（默认）
python train.py

# 训练 LSTM
python train.py --model lstm

# 自定义训练参数
python train.py --model gru --epochs 200 --lr 0.001 --batch_size 128
```

训练时控制台会输出每个 step 的 loss 以及每轮的平均 loss。模型仅保存验证 loss 最低的版本。

**训练日志示例：**

```
使用设备: cpu
模型类型: GRU | embedding_dim=256 | hidden_dim=512
词表大小: 8293 | 批次数量: 900
模型参数量: 9,135,973

开始训练...
  Epoch   1 | Step    0 | Loss: 9.0583
  Epoch   1 | Step  100 | Loss: 3.4217

>>> 第 1 轮 | 平均损失: 2.8743 | 学习率: 0.000500

  ★ 发现更好的模型，已保存！Best Loss: 2.8743
...
训练结束！最佳模型已保存到: result/poetry_model_GRU.pth
最佳损失: 1.3569
```

### 命令行生成诗句

使用 `test.py` 进行诗句续写或藏头诗生成：

```bash
# 续写诗句（GRU，交互式输入）
python test.py --mode continue

# 续写诗句（LSTM，命令行直接输入）
python test.py --model lstm --mode continue --input "床前明月光"

# 藏头诗（GRU）
python test.py --mode acrostic

# 藏头诗（LSTM + 自定义温度）
python test.py --model lstm --mode acrostic --input "我爱编程" --temperature 0.8
```

续写示例：

```
$ python test.py --mode continue --input "床前明月光"

生成的诗句：
床前明月光，疑是地上霜。举头望明月，低头思故乡。
```

藏头诗示例：

```
$ python test.py --mode acrostic --input "我爱编程"

生成的藏头诗：
我本山中一散仙，爱向松间枕石眠。编茅为屋云为幕，程远何妨到日边。
```

> **提示**：可通过 `--temperature` 和 `--topk` 参数调整生成风格。温度越低越保守，越高越随机。

---

## 模型架构

两种模型架构相同，仅在循环神经网络单元上有区别：

```
输入 (字符索引)
    │
    ▼
Embedding (vocab_size → 256)
    │
    ▼
双层 LSTM / 双层 GRU (256 → 512, num_layers=2, dropout=0.3/0.5)
    │
    ▼
Linear (512 → vocab_size)
    │
    ▼
输出 (每个字符在词表中的概率分布)
```

**LSTM 与 GRU 对比：**

| | LSTM | GRU |
|------|------|------|
| `embedding_dim` | 256 | 256 |
| `hidden_dim` | 512 | 512 |
| `dropout` | 0.5 | 0.3 |
| 参数量 | ~913 万 | ~914 万 |
| 模型路径 | `result/poetry_model_LSTM.pth` | `result/poetry_model_GRU.pth` |

**关键参数：**

| 参数 | 值 | 说明 |
|------|------|------|
| `vocab_size` | 8,293 | 词表大小 |
| `embedding_dim` | 256 | 词嵌入维度 |
| `hidden_dim` | 512 | 隐藏层维度 |
| `num_layers` | 2 | 循环网络层数 |

---

## 配置说明

### 训练配置 (`train.py` 命令行参数)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `gru` | 模型类型：`lstm` 或 `gru` |
| `--batch_size` | 64 | 批次大小 |
| `--lr` | 0.0005 | 初始学习率 |
| `--epochs` | 100 | 训练轮数 |
| `--model_path` | 自动 | 模型保存路径（默认保存到 `result/` 下） |

**学习率调度**：使用 `StepLR`，每 20 轮衰减为原来的 0.5 倍。

### 生成配置 (`test.py` 命令行参数)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_gen_len` | 64 | 最大生成长度 |
| `temperature` | 1.0 | 温度参数（越高越随机） |
| `top_k` | 50 | Top-K 采样范围 |

---

## 生成策略

项目实现了两种解码策略，用于提升生成质量：

### 温度采样 (Temperature Sampling)

通过温度参数 $T$ 调整概率分布的平滑程度：

$$P_i = \frac{\exp(\text{logits}_i / T)}{\sum_j \exp(\text{logits}_j / T)}$$

| 温度 | 效果 |
|------|------|
| $T < 1.0$ | 更保守，生成结果更稳定 |
| $T = 1.0$ | 原始分布 |
| $T > 1.0$ | 更随机，生成更多样 |

### Top-K 采样

每次生成时仅从概率最高的 $K$ 个候选词中采样，避免低概率词的干扰。

---

## 改进方向

- [ ] **更大数据集**：引入宋词、元曲等更多古典文学数据
- [ ] **Transformer / GPT 架构**：替换 LSTM 以获得更好的长程依赖建模能力
- [ ] **预训练词向量**：使用中文 BERT 或 Word2Vec 预训练嵌入
- [ ] **韵律约束**：引入平仄、押韵规则，提升古诗的格律规范性
- [ ] **Beam Search**：实现束搜索解码以替代贪心/采样解码
- [ ] **主题控制**：引导模型生成特定主题（山水、离别、边塞等）的诗句

---

## License

本项目仅用于学习和研究目的。