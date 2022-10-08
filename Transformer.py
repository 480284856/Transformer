import torch
from typing import *

from Cell.Encoder import TransformerEncoder
from Cell.Decoder import TransformerDecoder


class Transfomer(torch.nn.Module):
    def __init__(self, d_model, n_head, n_layers, target_sequence_length, vocab_size: int, device,
                 encoder_num_embeddings: int,
                 decoder_num_embeddings: int, embedding_dim: int = 512, ffnet_hidden_size: int = 2048,
                 ) -> None:
        '''
        target_sequence_length: used in generating attention mask
        '''
        super().__init__()
        self.encoder_embedding = torch.nn.Embedding(encoder_num_embeddings, embedding_dim)
        self.decoder_embedding = torch.nn.Embedding(decoder_num_embeddings, embedding_dim)

        self.encoder = TransformerEncoder(d_model=d_model, n_head=n_head, n_layers=n_layers,
                                          ffnet_hidden_size=ffnet_hidden_size, device=device)
        self.decoder = TransformerDecoder(d_model=d_model, n_head=n_head, n_layers=n_layers, target_sequence_length= \
            target_sequence_length, ffnet_hidden_size=ffnet_hidden_size, device=device)
        self.classfier = torch.nn.Linear(in_features=d_model, out_features=vocab_size)
        self.activation = torch.nn.Softmax(dim=-1)

    def forward(self, X, Y, X_key_padding_mask: Optional[torch.Tensor] = None,
                Y_key_padding_mask: Optional[torch.Tensor] = None):
        '''
        X_key_padding_mask: key padding mask for encoder, which non zero element indicates that the token has the same
        index with the element should be ignored while performing self-attention mechanism.
        Y_key_padding_mask: bs,tgt_length(src_length for encoder)
            key padding mask for decoder, which non zero element has the same function said above.
        '''
        X = self.encoder_embedding(X)
        Y = self.decoder_embedding(Y)

        encoder_output = self.encoder.forward(X, key_padding_mask=X_key_padding_mask)
        decoder_output = self.decoder.forward(Y, encoder_output, Y_key_padding_mask)
        logits = self.classfier(decoder_output)
        return logits

    def predict(self, X):
        pass


if __name__ == "__main__":
    X = torch.randn(size=(2, 5, 512))
    Y = torch.randn(size=(2, 4, 512))
    Y_key_padding_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    X_key_padding_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])
    model = Transfomer(d_model=512, n_head=8, n_layers=6, target_sequence_length=4, vocab_size=30000)
    result = model.forward(X, Y, Y_key_padding_mask=Y_key_padding_mask, X_key_padding_mask=X_key_padding_mask)
    print(result.shape)
