import torch
from myTransformer.Transformer import Transformer

class model(Transformer):
	def __init__(self,
			d_model=512, 
			n_head=8, 
			n_layers=6, 
			target_sequence_length=24, 
			vocab_size: int=17851,
			device='cpu',
            encoder_num_embeddings: int=10012,
            decoder_num_embeddings: int=17851, 
			embedding_dim: int = 512, 
			ffnet_hidden_size: int = 2048,	
	) -> None:
		super().__init__(
			d_model, 
			n_head, 
			n_layers, 
			target_sequence_length, 
			vocab_size,
			device,
            encoder_num_embeddings,
            decoder_num_embeddings, 
			embedding_dim, 
			ffnet_hidden_size)

if __name__ == "__main__":
	model:Transformer = model(512,8,6,24,17851,'cpu',10012,17851)
	model.load_state_dict()