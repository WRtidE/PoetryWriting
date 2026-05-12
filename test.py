"""
PoetryWriting 诗句生成脚本
支持续写诗句与藏头诗两种模式，以及 LSTM / GRU 模型切换。

用法:
    python test.py                              # 默认 GRU，续写模式
    python test.py --model lstm                 # 使用 LSTM
    python test.py --mode acrostic              # 藏头诗模式
    python test.py --temperature 1.2 --topk 80  # 调整生成参数
"""

import argparse
import numpy as np
import torch

from model import get_model


# ========== 各模型默认超参数 ==========
# 注：train.py 保存模型时会在文件名后追加损失率（如 poetry_model_GRU_1.3569.pth），
#     请将最佳模型重命名为以下路径，或通过 --model_path 指定实际文件名。

MODEL_CONFIGS = {
    'lstm': {
        'embedding_dim': 256,
        'hidden_dim': 512,
        'model_path': 'result/poetry_model_LSTM.pth',
    },
    'gru': {
        'embedding_dim': 256,
        'hidden_dim': 512,
        'model_path': 'result/poetry_model_GRU.pth',
    },
}

MAX_GEN_LEN = 64


# ========== 数据 & 模型加载 ==========

def load_data():
    """加载唐诗数据集字典"""
    datas = np.load('dataset/tang.npz', allow_pickle=True)
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()
    return ix2word, word2ix


def load_model(model_type, vocab_size, model_path=None):
    """加载训练好的模型"""
    cfg = MODEL_CONFIGS[model_type]

    if model_path is None:
        model_path = cfg['model_path']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = get_model(
        model_type=model_type,
        vocab_size=vocab_size,
        embedding_dim=cfg['embedding_dim'],
        hidden_dim=cfg['hidden_dim'],
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f'已加载模型: {model_type.upper()} | 路径: {model_path}')
    return model


# ========== 续写诗句 ==========

def generate(model, start_words, ix2word, word2ix, temperature=1.0, top_k=50):
    """
    温度 + Top-K 采样续写诗句。

    参数:
        model:       PoetryModel 实例
        start_words: 诗句开头字符串
        ix2word:     index → word 映射
        word2ix:     word → index 映射
        temperature: 温度参数（越高越随机）
        top_k:       Top-K 采样范围
    """
    device = next(model.parameters()).device
    results = list(start_words)
    start_len = len(start_words)

    input_tensor = torch.Tensor([word2ix['<START>']]).view(1, 1).long().to(device)
    hidden = None

    with torch.no_grad():
        for i in range(MAX_GEN_LEN):
            output, hidden = model(input_tensor, hidden)

            if i < start_len:
                word = results[i]
                input_tensor = torch.Tensor([word2ix[word]]).view(1, 1).long().to(device)
            else:
                # 温度 + Top-K 采样
                logits = output.data[0] / temperature
                top_k_logits, top_k_indices = logits.topk(min(top_k, logits.size(-1)))
                probs = torch.softmax(top_k_logits, dim=-1)
                top_index = top_k_indices[torch.multinomial(probs, 1).item()].item()

                word = ix2word[top_index]

                if word == '<EOP>' and len(results) > 20:
                    break

                results.append(word)
                input_tensor = torch.Tensor([top_index]).view(1, 1).long().to(device)

    return ''.join(results)


# ========== 藏头诗生成 ==========

def generate_acrostic(model, head_words, ix2word, word2ix, temperature=0.8):
    """
    生成藏头诗（七言）。

    参数:
        model:      PoetryModel 实例
        head_words: 藏头字（字符串）
        ix2word:    index → word 映射
        word2ix:    word → index 映射
        temperature: 采样温度
    """
    device = next(model.parameters()).device
    results = []
    hidden = None
    head_index = 0
    sentence_len = 0

    input_tensor = torch.Tensor([word2ix['<START>']]).view(1, 1).long().to(device)

    with torch.no_grad():
        while True:
            output, hidden = model(input_tensor, hidden)

            # 每句开头使用藏头字
            if sentence_len == 0:
                if head_index >= len(head_words):
                    break
                word = head_words[head_index]
                head_index += 1
            else:
                p = torch.softmax(output / temperature, dim=1)
                top_index = torch.multinomial(p, 1).item()
                word = ix2word[top_index]

                # 跳过特殊字符
                if word in ['<EOP>', '<START>', '</s>', '，', '。']:
                    continue

            results.append(word)
            sentence_len += 1

            # 七言句加标点
            if sentence_len == 7:
                results.append('，' if head_index % 2 == 1 else '。')
                sentence_len = 0

            input_tensor = torch.Tensor([word2ix[word]]).view(1, 1).long().to(device)

    return ''.join(results)


# ========== 主入口 ==========

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PoetryWriting 诗句生成')
    parser.add_argument('--model', type=str, default='gru',
                        choices=['lstm', 'gru'], help='模型架构 (默认: gru)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型权重路径（默认使用 result/ 下的对应文件）')
    parser.add_argument('--mode', type=str, default='continue',
                        choices=['continue', 'acrostic'], help='生成模式')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='温度参数 (默认: 1.0)')
    parser.add_argument('--topk', type=int, default=50,
                        help='Top-K 采样范围 (默认: 50)')
    parser.add_argument('--input', type=str, default=None,
                        help='诗句开头 / 藏头字（不提供则交互式输入）')

    args = parser.parse_args()

    # 加载
    ix2word, word2ix = load_data()
    model = load_model(args.model, len(word2ix), model_path=args.model_path)

    # 执行
    if args.mode == 'continue':
        start_words = args.input or input('请输入诗句开头：')
        result = generate(model, start_words, ix2word, word2ix,
                          temperature=args.temperature, top_k=args.topk)
        print('\n生成的诗句：')
        print(result)

    elif args.mode == 'acrostic':
        head_words = args.input or input('请输入藏头字：')
        result = generate_acrostic(model, head_words, ix2word, word2ix,
                                   temperature=args.temperature)
        print('\n生成的藏头诗：')
        print(result)