"""
Base default handler to load torchscript or eager mode [state_dict] models
Also, provides handle method per torch serve custom model specification
"""
import re
import os
import sys
import dill
import torch



import abc
import logging
import os
import importlib.util
import time
import torch
from pkg_resources import packaging
from ts.utils.util import list_classes_from_module, load_label_mapping
from myTransformer.TransfomerTokenzer import truncate_pad, generate_key_padding_mask, load_data_nmt
if packaging.version.parse(torch.__version__) >= packaging.version.parse("1.8.1"):
	from torch.profiler import profile, record_function, ProfilerActivity
	PROFILER_AVAILABLE = True
else:
	PROFILER_AVAILABLE = False


logger = logging.getLogger(__name__)

ipex_enabled = False
if os.environ.get("TS_IPEX_ENABLE", "false") == "true":
	try:
		import intel_extension_for_pytorch as ipex
		ipex_enabled = True
	except ImportError as error:
		logger.warning("IPEX is enabled but intel-extension-for-pytorch is not installed. Proceeding without IPEX.")

class BaseHandler(abc.ABC):
	"""
	Base default handler to load torchscript or eager mode [state_dict] models
	Also, provides handle method per torch serve custom model specification
	"""

	def __init__(self):
		self.model = None
		self.mapping = None
		self.device = None
		self.initialized = False
		self.context = None
		self.manifest = None
		self.map_location = None
		self.explain = False
		self.target = 0
		self.profiler_args = {}

	def initialize(self, context):
		"""Initialize function loads the model.pt file and initialized the model object.
		First try to load torchscript else load eager mode state_dict based model.

		Args:
			context (context): It is a JSON Object containing information
			pertaining to the model artifacts parameters.

		Raises:
			RuntimeError: Raises the Runtime error when the model.py is missing

		"""
		properties = context.system_properties
		self.map_location = "cuda" if torch.cuda.is_available(
		) and properties.get("gpu_id") is not None else "cpu"
		self.device = torch.device(
			self.map_location + ":" + str(properties.get("gpu_id"))
			if torch.cuda.is_available() and properties.get("gpu_id") is not None
			else self.map_location
		)
		self.manifest = context.manifest

		model_dir = properties.get("model_dir")
		model_pt_path = None
		if "serializedFile" in self.manifest["model"]:
			serialized_file = self.manifest["model"]["serializedFile"]
			model_pt_path = os.path.join(model_dir, serialized_file)

		# model def file
		model_file = self.manifest["model"].get("modelFile", "")

		if model_file:
			logger.debug("Loading eager model")
			self.model = self._load_pickled_model(
				model_dir, model_file, model_pt_path)
			self.model.to(self.device)
		else:
			logger.debug("Loading torchscript model")
			if not os.path.isfile(model_pt_path):
				raise RuntimeError("Missing the model.pt file")

			self.model = self._load_torchscript_model(model_pt_path)

		self.model.eval()
		if ipex_enabled:
			self.model = self.model.to(memory_format=torch.channels_last)
			self.model = ipex.optimize(self.model)

		logger.debug('Model file %s loaded successfully', model_pt_path)

		# Load class mapping for classifiers
		mapping_file_path = os.path.join(model_dir, "index_to_name.json")
		self.mapping = load_label_mapping(mapping_file_path)

		_, self.src_vocab, self.tgt_vocab = load_data_nmt(1, 24, None)
			
		self.initialized = True
		print("*******************initialize***************************")
	def _load_torchscript_model(self, model_pt_path):
		"""Loads the PyTorch model and returns the NN model object.

		Args:
			model_pt_path (str): denotes the path of the model file.

		Returns:
			(NN Model Object) : Loads the model object.
		"""
		return torch.jit.load(model_pt_path, map_location=self.device)

	def _load_pickled_model(self, model_dir, model_file, model_pt_path):
		"""
		Loads the pickle file from the given model path.

		Args:
			model_dir (str): Points to the location of the model artefacts.
			model_file (.py): the file which contains the model class.
			model_pt_path (str): points to the location of the model pickle file.

		Raises:
			RuntimeError: It raises this error when the model.py file is missing.
			ValueError: Raises value error when there is more than one class in the label,
						since the mapping supports only one label per class.

		Returns:
			serialized model file: Returns the pickled pytorch model file
		"""
		model_def_path = os.path.join(model_dir, model_file)
		if not os.path.isfile(model_def_path):
			raise RuntimeError("Missing the model.py file")

		module = importlib.import_module(model_file.split(".")[0])
		model_class_definitions = list_classes_from_module(module)
		if len(model_class_definitions) != 1:
			raise ValueError(
				"Expected only one class as model definition. {}".format(
					model_class_definitions
				)
			)

		model_class = model_class_definitions[0]
		model = model_class()
		if model_pt_path:
			state_dict = torch.load(model_pt_path, map_location=self.device)
			model.load_state_dict(state_dict)

		for layer in model.decoder.TransformerDecoder:  # 预测的时候不用遮掩
			layer.masked_multi_head_attention_layer.masked_matrix = None
		return model

	def predict_prepare(self, X: str, src_vocab, tgt_vocab):
		'''
		translate string sequence into numbers of sequence

		Parameter:
			X: sequence to be translated into french, such as 'I love you'.

		Return:
			encoder_input_ids,decoder_input_ids
		'''
		X += ' <eos>'
		src_tokens = X.lower().split(' ')
		src_tokens = [src_vocab[e] for e in src_tokens]
		X_key_padding_mask = generate_key_padding_mask(1, 24, torch.tensor([len(src_tokens)])).to(
			self.device)
		encoder_input_ids = torch.tensor(
			[truncate_pad(src_tokens, 24, src_vocab['<pad>'], src_vocab)])  # bs, num_step

		decoder_input_ids = torch.tensor([[tgt_vocab['<bos>']]], dtype=torch.long)
		return encoder_input_ids, X_key_padding_mask, decoder_input_ids

	def preprocess(self, request: str):
		'''
		处理数据，转化为模型可以接受的形式
		----
		Parameter(s):
			request: str
				输入的文本
		'''
		line = request[0]
		preprocessed_data = line.get("data") or line.get("body")
		if isinstance(preprocessed_data, (bytes, bytearray)):
			preprocessed_data = preprocessed_data.decode('utf-8')

		
		encoder_input_ids, X_key_padding_mask, decoder_input_ids = self.predict_prepare(preprocessed_data, self.src_vocab, self.tgt_vocab)
		print("*******************preprocess***************************")
		return (encoder_input_ids, X_key_padding_mask, decoder_input_ids)

	def postprocess(self, generated_sequence):
		return [generated_sequence]

	def inference(self, data, *args, **kwargs):
		with torch.no_grad():
			self.model.eval()
			encoder_input_ids, X_key_padding_mask, decoder_input_ids = data

			encoder_embeddings = self.model.encoder_embedding(encoder_input_ids)
			memory = self.model.encoder.forward(encoder_embeddings, key_padding_mask=X_key_padding_mask)
			generated_sequence = []
			eos_token_id = self.tgt_vocab['<eos>']
			for _ in range(24):
				decoder_embeddings = self.model.decoder_embedding(decoder_input_ids)
				decoder_logits: torch.Tensor = \
				self.model.decoder.forward(decoder_embeddings, memory, memory_key_padding_mask=X_key_padding_mask)[0][-1]
				decoder_logits = self.model.classfier(decoder_logits)
				best_token = decoder_logits.argmax().view(1, 1)
				decoder_input_ids = torch.concat([decoder_input_ids, best_token], dim=-1)
				if best_token.item() == eos_token_id:
					break
				generated_sequence.append(best_token.item())
		print("*******************inference***************************")
		return ' '.join(self.tgt_vocab.to_tokens(generated_sequence))

	def handle(self, data, context):
		"""Entry point for default handler. It takes the data from the input request and returns
		the predicted outcome for the input.

		Args:
			data (list): The input data that needs to be made a prediction request on.
			context (Context): It is a JSON Object containing information pertaining to
							the model artefacts parameters.

		Returns:
			list : Returns a list of dictionary with the predicted response.
		"""

		# It can be used for pre or post processing if needed as additional request
		# information is available in context
		start_time = time.time()

		self.context = context
		metrics = self.context.metrics

		is_profiler_enabled = os.environ.get("ENABLE_TORCH_PROFILER", None)
		if is_profiler_enabled:
			if PROFILER_AVAILABLE:
				output, _ = self._infer_with_profiler(data=data)
			else:
				raise RuntimeError("Profiler is enabled but current version of torch does not support."
								"Install torch>=1.8.1 to use profiler.")
		else:
			if self._is_describe():
				output = [self.describe_handle()]
			else:
				data_preprocess = self.preprocess(data)

				if not self._is_explain():
					output = self.inference(data_preprocess)
					output = self.postprocess(output)
				else:
					output = self.explain_handle(data_preprocess, data)

		stop_time = time.time()
		metrics.add_time('HandlerTime', round(
			(stop_time - start_time) * 1000, 2), None, 'ms')
		return output

	def _infer_with_profiler(self, data):
		"""Custom method to generate pytorch profiler traces for preprocess/inference/postprocess

		Args:
			data (list): The input data that needs to be made a prediction request on.

		Returns:
			output : Returns a list of dictionary with the predicted response.
			prof: pytorch profiler object
		"""
		# Setting the default profiler arguments to profile cpu, gpu usage and record shapes
		# User can override this argument based on the requirement
		if not self.profiler_args:
			self.profiler_args["activities"] = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
			self.profiler_args["record_shapes"] = True

		if "on_trace_ready" not in self.profiler_args:
			result_path = "/tmp/pytorch_profiler"
			dir_name = ""
			try:
				model_name = self.manifest["model"]["modelName"]
				dir_name = model_name
			except KeyError:
				logging.debug("Model name not found in config")

			result_path = os.path.join(result_path, dir_name)
			self.profiler_args["on_trace_ready"] = torch.profiler.tensorboard_trace_handler(result_path)
			logger.info("Saving chrome trace to : ", result_path) # pylint: disable=logging-too-many-args

		with profile(**self.profiler_args) as prof:
			with record_function("preprocess"):
				data_preprocess = self.preprocess(data)
			if not self._is_explain():
				with record_function("inference"):
					output = self.inference(data_preprocess)
				with record_function("postprocess"):
					output = self.postprocess(output)
			else:
				with record_function("explain"):
					output = self.explain_handle(data_preprocess, data)

		logger.info(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
		return output, prof


	def explain_handle(self, data_preprocess, raw_data):
		"""Captum explanations handler

		Args:
			data_preprocess (Torch Tensor): Preprocessed data to be used for captum
			raw_data (list): The unprocessed data to get target from the request

		Returns:
			dict : A dictionary response with the explanations response.
		"""
		output_explain = None
		inputs = None
		target = 0

		logger.info("Calculating Explanations")
		row = raw_data[0]
		if isinstance(row, dict):
			logger.info("Getting data and target")
			inputs = row.get("data") or row.get("body")
			target = row.get("target")
			if not target:
				target = 0

		output_explain = self.get_insights(data_preprocess, inputs, target)
		return output_explain

	def _is_explain(self):
		if self.context and self.context.get_request_header(0, "explain"):
			if self.context.get_request_header(0, "explain") == "True":
				self.explain = True
				return True
		return False

	def _is_describe(self):
		if self.context and self.context.get_request_header(0, "describe"):
			if self.context.get_request_header(0, "describe") == "True":
				return True
		return False

	def describe_handle(self):
		"""Customized describe handler

		Returns:
			dict : A dictionary response.
		"""
		# pylint: disable=unnecessary-pass
		pass
		# pylint: enable=unnecessary-pass
