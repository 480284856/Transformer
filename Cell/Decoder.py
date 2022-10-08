import torch
from typing import *
from .Encoder import Multi_Head_Attention
from .multi_head_attention import Multi_Head_Attention_forward


class Masked_Multi_Head_Attention(torch.nn.Module):
    def __init__(self, input_size: int, n_head: int, target_sequence_length: int, device) -> None:
        '''
        多头KQV矩阵运算
        ----
        Parameter(s):
            input_size: int
                输入向量的维度(`d_model`)
            n_head: int
                头的个数，有多少次注意力计算
            target_sequence_length: int
                decoder 端的输入的序列的长度，产生掩码的时候需要用到
        '''
        super(Masked_Multi_Head_Attention, self).__init__()

        assert input_size % n_head == 0, "n_head应该要整除input_size"
        self.hidden_size = input_size // n_head
        self.n_head = n_head
        self.tgt_length = target_sequence_length

        self.KProject = torch.nn.Linear(in_features=input_size,
                                        out_features=self.hidden_size * n_head)  # 一次性计算多头，避免使用循环 （pic1）
        self.QProject = torch.nn.Linear(in_features=input_size, out_features=self.hidden_size * n_head)
        self.VProject = torch.nn.Linear(in_features=input_size, out_features=self.hidden_size * n_head)
        self.Linear = torch.nn.Linear(in_features=self.hidden_size * n_head, out_features=self.hidden_size * n_head)

        self.masked_matrix = self.generate_masked_matrix().to(device)
        self.device = device

    def generate_masked_matrix(self) -> torch.Tensor:
        '''
        产生注意力掩码，即产生一个和qkt一样的非零值为-inf的上三角矩阵
        '''

        mask_index = torch.triu(torch.ones(size=(self.tgt_length, self.tgt_length), dtype=torch.long), diagonal=1)
        masked_matrix = torch.zeros(size=(self.tgt_length, self.tgt_length))
        return masked_matrix.masked_fill(mask_index, value=-torch.inf)

    def forward(self, X: torch.tensor, key_padding_mask: torch.Tensor = None):
        '''
        X: bs, sequence_length, embedding_size
        key_padding_mask: bs, sequence_length
            any non-zero element in this tensor indocates that the token has the same index with the element should be ignored
            while performing attention mechanism.
        '''

        Q = self.QProject(X)
        K = self.KProject(X)
        V = self.VProject(X)
        attention_mask = self.masked_matrix

        result = Multi_Head_Attention_forward(Q, K, V, hidden_size=self.hidden_size, n_head=self.n_head,
                                              device=self.device,
                                              attention_mask=attention_mask, key_padding_mask=key_padding_mask)
        #  bs, L, embedding_size

        return self.Linear(result)


class Decoder_Multi_Head_Attention(Multi_Head_Attention):
    def __init__(self, input_size: int, n_head: int, device) -> None:
        super().__init__(input_size, n_head, device)
        self.device = device

    def forward(self, Y: torch.tensor, memory: torch.Tensor, key_padding_mask: Optional[torch.tensor] = None):
        Q = self.QProject(Y)
        K = self.KProject(memory)
        V = self.VProject(memory)

        result = Multi_Head_Attention_forward(Q, K, V, hidden_size=self.hidden_size, n_head=self.n_head,
                                              device=self.device, key_padding_mask=key_padding_mask)
        return self.Linear(result)


class TransformerDecoderLayer(torch.nn.Module):
    def __init__(self, input_size: int, n_head: int, target_sequence_length: int, ffnet_hidden_size: int,
                 device) -> None:
        '''
        Parameter(s):
            input_size: int
                输入向量的维度(`d_model`)
            n_head: int
                头的个数，有多少次注意力计算
            target_sequence_length: int
                decoder 端的输入的序列的长度，产生掩码的时候需要用到
            ffnet_hidden_size: the number of hidden units between the two linear transformation in Feed Forward Layer
        '''
        super().__init__()
        self.masked_multi_head_attention_layer = Masked_Multi_Head_Attention(input_size=input_size, n_head=n_head,
                                                                             target_sequence_length=target_sequence_length,
                                                                             device=device)
        self.norm_mmha_layer = torch.nn.LayerNorm(normalized_shape=input_size)

        self.encd_decd_attention_layer = Decoder_Multi_Head_Attention(input_size=input_size, n_head=n_head,
                                                                      device=device)
        self.norm_ecdc_layer = torch.nn.LayerNorm(normalized_shape=input_size)

        self.FFLayer = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_size, out_features=ffnet_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=ffnet_hidden_size, out_features=input_size)
        )

        self.norm_fnc = torch.nn.LayerNorm(normalized_shape=input_size)

    def forward(self, Y: torch.Tensor, memory: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Y: target sequence shifted right by one position
        memory: output of the TransformerEncoder
        key_padding_mask: bs, sequence_length
            any non-zero element in this tensor indocates that the token has the same index with the element should be ignored
            while performing attention mechanism.
        '''
        decoder_sfa_ = self.masked_multi_head_attention_layer.forward(Y, key_padding_mask)
        decoder_sfa_ = self.norm_mmha_layer(Y + decoder_sfa_)

        encoder_decoeder_attention = self.encd_decd_attention_layer.forward(decoder_sfa_, memory)
        encoder_decoeder_attention = self.norm_ecdc_layer(encoder_decoeder_attention + decoder_sfa_)

        ff_output = self.FFLayer(encoder_decoeder_attention)
        ff_output = self.norm_fnc(ff_output + encoder_decoeder_attention)

        return ff_output


class TransformerDecoder(torch.nn.Module):
    def __init__(self, d_model, n_head, n_layers, target_sequence_length, device, ffnet_hidden_size=2048) -> None:
        '''
        ffnet_hidden_size: the number of hidden units between the two linear transformation in Feed Forward Layer
        target_sequence_length: length of target sequence, in purpose of generating masked matrix
        '''
        super().__init__()
        self.TransformerDecoder = torch.nn.ModuleList([
            TransformerDecoderLayer(input_size=d_model, n_head=n_head, target_sequence_length=target_sequence_length,
                                    ffnet_hidden_size=ffnet_hidden_size, device=device)
            for _ in range(n_layers)])

    def forward(self, Y, memory, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Y: target sequence shifted right by one position
        memory: output of the TransformerEncoder
        key_padding_mask: bs, sequence_length
            any non-zero element in this tensor indocates that the token has the same index with the element should be ignored
            while performing attention mechanism.
        '''
        for layer in self.TransformerDecoder:
            Y = layer(Y, memory, key_padding_mask)
        return Y


if __name__ == "__main__":
    Y = torch.randn(size=(2, 30, 512))
    memory = torch.randn(size=(2, 5, 512))
    model = TransformerDecoder(d_model=512, n_head=8, n_layers=6, target_sequence_length=30)
    output = model.forward(Y, memory)
    print(output.shape)
