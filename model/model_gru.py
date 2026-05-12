import torch
import torch.nn as nn


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512):
        super(PoetryModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = 2

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 双层 GRU
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.3
        )

        # 全连接输出层
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):

        embeds = self.embedding(x)

        batch_size, seq_len = x.size()

        # 初始化 hidden
        if hidden is None:
            hidden = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_dim
            ).to(x.device)

        # GRU 前向传播
        output, hidden = self.gru(embeds, hidden)

        # 输出层
        output = self.fc(output)

        # reshape 用于交叉熵
        output = output.reshape(batch_size * seq_len, -1)

        return output, hidden