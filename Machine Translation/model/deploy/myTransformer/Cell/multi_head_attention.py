import torch
from typing import *


def Multi_Head_Attention_forward(Q: torch.Tensor,
                                 K: torch.Tensor,
                                 V: torch.Tensor,
                                 hidden_size: int,
                                 n_head: int,
                                 device,
                                 attention_mask: Optional[torch.Tensor] = None,
                                 key_padding_mask: Optional[torch.Tensor] = None,
                                 memory_key_padding_mask: Optional[torch.tensor] = None):
    '''
	Q: bs, L, hidden_size
	K: bs, L, hidden_size
	V: bs, L, hidden_size
	hidden_zie: 每个head中token的维度
	attention_mask:
		self-attention mask matrix
	key_padding_mask:
		indicate which token should be ignored while performing attention mechanism
	'''
    if attention_mask != None and key_padding_mask != None:
        attention_mask = merge2maskMatrices(attention_mask=attention_mask, key_padding_mask=key_padding_mask,
                                            n_head=n_head)
    else:
        if key_padding_mask == None:
            pass
        else:
            bs, L = key_padding_mask.shape
            key_padding_mask = key_padding_mask.view(bs, 1, 1, L). \
                expand(-1, n_head, -1, -1).reshape(bs * n_head, 1, L)

            zeros = torch.zeros(L, L).to(device)
            attention_mask = zeros.masked_fill(key_padding_mask.bool(), value=-torch.inf)
    if memory_key_padding_mask != None:
        src_L = K.shape[1]
        tgt_L = Q.shape[1]
        bs = Q.shape[0]
        memory_key_padding_mask = memory_key_padding_mask.view(bs, 1, 1, src_L). \
                expand(-1, n_head, -1, -1).reshape(bs * n_head, 1, src_L)
        zeros = torch.zeros(tgt_L, src_L).to(device)
        attention_mask = zeros.masked_fill(memory_key_padding_mask.bool(), value=-torch.inf)

    bs = Q.shape[0]
    Q: torch.tensor = Q.view(Q.shape[0], Q.shape[1], n_head, hidden_size)  # bs,L, n_head, hidden_size
    Q = Q.permute(0, 2, 1, 3).contiguous().view(Q.shape[0] * n_head, Q.shape[1],
                                                hidden_size)  # bs*n_head, L, hidden_size
    K: torch.tensor = K.view(K.shape[0], K.shape[1], n_head, hidden_size)
    K = K.permute(0, 2, 1, 3).contiguous().view(K.shape[0] * n_head, K.shape[1], hidden_size)
    V: torch.tensor = V.view(V.shape[0], V.shape[1], n_head, hidden_size)
    V = V.permute(0, 2, 1, 3).contiguous().view(V.shape[0] * n_head, V.shape[1], hidden_size)

    qkt = torch.bmm(Q, torch.permute(K, (0, 2, 1)))  # bs*n_head, L, hidden_size | bs*n_head, hidden_size, L
    # bs*n_head, L,L

    # permute: https://www.cnblogs.com/hisi-tech/p/16451240.html
    # Q中的每一行都是一个词向量q去内积K.T的每一列，相当于查看其它词向量，得到其他词向量对这个词的重要程度，
    # 得到的结果中，第i行是第i个词对其他的（包括自己）的词的注意力权重  （pic1)
    qkt = torch.divide(qkt, torch.sqrt(torch.tensor([hidden_size])).to(qkt.device))
    if attention_mask == None:
        pass
    else:
        qkt += attention_mask

    score = torch.softmax(qkt, dim=-1)
    result: torch.tensor = torch.bmm(score, V)  # bs*n_head,L,L |bs* n_head, L,hidden_size
    # bs*n_head,L, hidden_size
    result = result.permute(1, 0, 2)
    # L,bs*n_head, hidden_size
    result = result.contiguous().view(Q.shape[1], bs, hidden_size * n_head)

    return result.permute(1, 0, 2)


def merge2maskMatrices(attention_mask: torch.Tensor, key_padding_mask: torch.Tensor, n_head: int) -> torch.Tensor:
    '''
	attention_mask: tgt_length,tgt_length(src_length for encoder)
	key_padding_mask: bs,tgt_length(src_length for encoder)
	'''
    assert attention_mask.dim() == 2, f'Expected attention_mask\'s has dimension of 2, but get {attention_mask.dim()},attention_mask: tgt_length,tgt_length(src_length for encoder)'
    assert key_padding_mask.dim() == 2, f'Expected key_padding_mask\'s has dimension of 2, but get {key_padding_mask.dim()},key_padding_mask: bs,tgt_length(src_length for encoder)'
    bs, L = key_padding_mask.shape
    key_padding_mask = key_padding_mask.view(bs, 1, 1, L). \
        expand(-1, n_head, -1, -1).reshape(bs * n_head, 1, L)
    attention_mask = attention_mask.masked_fill(key_padding_mask.bool(), value=-torch.inf)
    return attention_mask
