import torch

class TransformerEmbedding(torch.nn.Module):
	def __init__(self,num_embeddings:int,embedding_dim: int) -> None:
		super().__init__()
		self.embedding = torch.nn.Embedding(num_embeddings,embedding_dim)
	def __getitem__(self,)
