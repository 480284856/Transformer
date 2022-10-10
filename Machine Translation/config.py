import torch
from dataclasses import dataclass

@dataclass
class Config:
	'''
	vocab_size 需要根据num_example的变化而改变
	'''
	d_model: int = 512
	n_head: int = 8
	n_layers: int = 6

	target_sequence_length: int = 24
	num_steps: int = target_sequence_length

	vocab_size: int = 17851
	num_examples = None

	batch_size: int = 128
	epoch: int = 18

	src_language_vocab_size: int = 10012
	encoder_num_embeddings: int = src_language_vocab_size
	decoder_num_embeddings: int = vocab_size

	device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

	lr:float = 1e-4
	weight_decay:float = 1e-4
	model_state_dict_save_path: str = 'model'

	log_dir: str = 'train_log'
	log_dir_comment: str = 'training_1'

	padding_token: str = '<pad>'
	eos_token: str = '<eos>'
	generated_sequence_max_length: int = 24