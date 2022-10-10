import torch
from typing import *
from .multi_head_attention import Multi_Head_Attention_forward


class Multi_Head_Attention(torch.nn.Module):
    def __init__(self, input_size: int, n_head: int, device) -> None:
        '''
        多头KQV矩阵运算
        ----
        Parameter(s):
            input_size: int
                输入向量的维度(`d_model`)
            n_head: int
                头的个数，有多少次注意力计算
        '''
        super(Multi_Head_Attention, self).__init__()

        assert input_size % n_head == 0, "n_head应该要整除input_size"
        self.hidden_size = input_size // n_head
        self.n_head = n_head

        self.KProject = torch.nn.Linear(in_features=input_size,
                                        out_features=self.hidden_size * n_head)  # 一次性计算多头，避免使用循环 （pic1）
        self.QProject = torch.nn.Linear(in_features=input_size, out_features=self.hidden_size * n_head)
        self.VProject = torch.nn.Linear(in_features=input_size, out_features=self.hidden_size * n_head)
        self.Linear = torch.nn.Linear(in_features=self.hidden_size * n_head, out_features=self.hidden_size * n_head)

        self.device = device

    def forward(self,
                X: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                src_atten_mask: Optional[torch.Tensor] = None):
        '''
        X: bs, sequence_length, embedding_size
        key_padding_mask: bs, sequence_length
            any non-zero element in this tensor indicates that the token has the same index with the element should be ignored
            while performing attention mechanism.
        src_atten_mask: L_src,L_src
            atten_mask for source sequence
        '''

        Q = self.QProject(X)
        K = self.KProject(X)
        V = self.VProject(X)
        result = Multi_Head_Attention_forward(Q, K, V, hidden_size=self.hidden_size, n_head=self.n_head,
                                              key_padding_mask=key_padding_mask, device=self.device,attention_mask=src_atten_mask)

        return self.Linear(result)


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_head, ffnet_hidden_size, device) -> None:
        '''
        ffnet_hidden_size: the number of hidden units between the two linear transformation in Feed Forward Layer
        '''
        super().__init__()
        self.MHAttention = Multi_Head_Attention(input_size=d_model, n_head=n_head, device=device)
        self.norm_mha = torch.nn.LayerNorm(normalized_shape=d_model)

        self.FFLayer = torch.nn.Sequential(
            torch.nn.Linear(in_features=d_model, out_features=ffnet_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=ffnet_hidden_size, out_features=d_model)
        )

        self.norm_fcn = torch.nn.LayerNorm(normalized_shape=d_model)

    def forward(self,
                X,
                key_padding_mask: Optional[torch.tensor] = None,
                src_atten_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        X: bs, sequence_length, embedding_size
        key_padding_mask: bs, sequence_length
            any non-zero element in this tensor indicates that the token has the same index with the element should be ignored
            while performing attention mechanism.
        src_atten_mask: L_src,L_src
            atten_mask for source sequence
        '''
        mh_attention = self.MHAttention(X, key_padding_mask=key_padding_mask, src_atten_mask=src_atten_mask)
        output_mha = self.norm_mha(mh_attention + X)
        output = self.FFLayer(output_mha)
        return self.norm_fcn(output_mha + output)


class TransformerEncoder(torch.nn.Module):
    def __init__(self, d_model, n_head, n_layers, device, ffnet_hidden_size=2048) -> None:
        '''
        ffnet_hidden_size: the number of hidden units between the two linear transformation in Feed Forward Layer
        '''
        super().__init__()
        self.TransformerEncoder = torch.nn.ModuleList([
            TransformerEncoderLayer(d_model=d_model, n_head=n_head, ffnet_hidden_size=ffnet_hidden_size, device=device)
            for _ in range(n_layers)])

    def forward(self,
                X,
                key_padding_mask: Optional[torch.tensor] = None,
                src_atten_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        key_padding_mask: bs, sequence_length
            any non-zero element in this tensor indicates that the token has the same index with the element should be ignored
            while performing attention mechanism.
        src_atten_mask: L_src,L_src
            atten_mask for source sequence
        '''
        for layer in self.TransformerEncoder:
            X = layer(X, key_padding_mask, src_atten_mask)
        return X


if __name__ == "__main__":
    X = torch.randn(size=(2, 3, 512))
    model = TransformerEncoder(d_model=512, n_head=8, n_layers=6, ffnet_hidden_size=2048, device='cpu')
    result = model(X)
    print(result.shape)
