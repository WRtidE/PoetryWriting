import torch
import torch.nn as nn

class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super(PoetryModel, self).__init__()

        self.hidden_dim = hidden_dim
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 双层 LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.5
        )
        # 全连接层
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x shape: [batch_size, seq_len]

        embeds = self.embedding(x)

        batch_size, seq_len = x.size()

        if hidden is None:
            h_0 = torch.zeros(2, batch_size, self.hidden_dim).to(x.device)
            c_0 = torch.zeros(2, batch_size, self.hidden_dim).to(x.device)
        else:
            h_0, c_0 = hidden

        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.fc(output)
        # reshape 方便计算交叉熵
        output = output.reshape(batch_size * seq_len, -1)

        return output, hidden
    