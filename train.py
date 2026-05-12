"""
PoetryWriting 训练脚本
支持双层 LSTM / 双层 GRU 两种模型架构，通过命令行参数切换。

用法:
    python train.py                  # 默认使用 GRU
    python train.py --model lstm     # 使用 LSTM
    python train.py --model gru --epochs 200 --lr 0.001
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import get_model


# ========== 各模型的默认超参数 ==========

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


def prepare_data(batch_size):
    """加载唐诗数据集，返回 DataLoader 和字典"""
    datas = np.load('dataset/tang.npz', allow_pickle=True)

    data = datas['data']
    ix2word = datas['ix2word'].item()
    word2ix = datas['word2ix'].item()

    data = torch.from_numpy(data)

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    return dataloader, ix2word, word2ix


def train(model_type='gru', batch_size=64, lr=0.0005, epochs=100, model_path=None):
    """
    训练诗歌生成模型。

    参数:
        model_type: 模型类型 'lstm' 或 'gru'
        batch_size: 批次大小
        lr:         学习率
        epochs:     训练轮数
        model_path: 模型保存路径（None 则使用默认路径）
    """
    cfg = MODEL_CONFIGS[model_type]

    if model_path is None:
        model_path = cfg['model_path']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    print(f'模型类型: {model_type.upper()} | embedding_dim={cfg["embedding_dim"]} | hidden_dim={cfg["hidden_dim"]}')

    # 数据
    dataloader, ix2word, word2ix = prepare_data(batch_size)
    print(f'词表大小: {len(word2ix)} | 批次数量: {len(dataloader)}')

    # 模型
    model = get_model(
        model_type=model_type,
        vocab_size=len(word2ix),
        embedding_dim=cfg['embedding_dim'],
        hidden_dim=cfg['hidden_dim'],
    ).to(device)

    print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')

    print('\n开始训练...\n')

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for step, batch_data in enumerate(dataloader):
            batch_data = batch_data.long().to(device)

            input_data = batch_data[:, :-1]          # [B, seq_len-1]
            target_data = batch_data[:, 1:].reshape(-1)  # [B*(seq_len-1)]

            optimizer.zero_grad()
            output, _ = model(input_data)
            loss = criterion(output, target_data)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0:
                print(f'  Epoch {epoch + 1:3d} | Step {step:4d} | Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'\n>>> 第 {epoch + 1} 轮 | 平均损失: {avg_loss:.4f} | 学习率: {current_lr:.6f}\n')

        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            # 文件名包含损失率，避免覆盖已训练好的模型
            loss_str = f'{best_loss:.4f}'
            save_path = model_path.replace('.pth', f'_{loss_str}.pth')
            torch.save(model.state_dict(), save_path)
            print(f'  ★ 发现更好的模型，已保存！Best Loss: {best_loss:.4f} → {save_path}\n')

    print(f'训练结束！最佳模型已保存到: {save_path}')
    print(f'最佳损失: {best_loss:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PoetryWriting 模型训练')
    parser.add_argument('--model', type=str, default='gru',
                        choices=['lstm', 'gru'], help='模型架构 (默认: gru)')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.0005, help='学习率')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--model_path', type=str, default=None, help='模型保存路径')

    args = parser.parse_args()
    train(
        model_type=args.model,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        model_path=args.model_path,
    )