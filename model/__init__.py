"""
PoetryWriting 模型包
提供统一的模型工厂函数，支持 LSTM / GRU 两种架构切换。
"""

from .model_lstm import PoetryModel as LSTMModel
from .model_gru import PoetryModel as GRUModel


def get_model(model_type: str, vocab_size: int, embedding_dim: int = None, hidden_dim: int = None):
    """
    工厂函数：根据 model_type 创建对应的诗歌生成模型。

    参数:
        model_type: 'lstm' 或 'gru'
        vocab_size: 词表大小
        embedding_dim: 词嵌入维度（None 则使用模型默认值）
        hidden_dim:   隐藏层维度（None 则使用模型默认值）

    返回:
        PoetryModel 实例
    """
    model_type = model_type.lower().strip()

    if model_type == 'lstm':
        emb = embedding_dim if embedding_dim is not None else 256
        hid = hidden_dim if hidden_dim is not None else 512
        return LSTMModel(vocab_size, embedding_dim=emb, hidden_dim=hid)

    elif model_type == 'gru':
        emb = embedding_dim if embedding_dim is not None else 256
        hid = hidden_dim if hidden_dim is not None else 512
        return GRUModel(vocab_size, embedding_dim=emb, hidden_dim=hid)

    else:
        raise ValueError(f"不支持的模型类型: '{model_type}'，可选: 'lstm' 或 'gru'")
