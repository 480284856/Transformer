import sys
import torch

sys.path.append('X:\Code\Source code of Transformer')
from config import Config
from Transformer import Transformer
from TransfomerTokenzer import truncate_pad,generate_key_padding_mask,load_data_nmt

config = Config()
config.device = 'cpu'

def predict_prepare(X: str, src_vocab, tgt_vocab):
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
    X_key_padding_mask = generate_key_padding_mask(1, config.num_steps, torch.tensor([len(src_tokens)])).to(config.device)
    encoder_input_ids = torch.tensor([truncate_pad(src_tokens, config.num_steps, src_vocab[config.padding_token], src_vocab)])  # bs, num_step

    decoder_input_ids = torch.tensor([[tgt_vocab['<bos>']]], dtype=torch.long)
    return encoder_input_ids, X_key_padding_mask, decoder_input_ids


def predict(model: Transformer, X: str, src_vocab, tgt_vocab):
    model.eval()

    encoder_input_ids, X_key_padding_mask, decoder_input_ids = predict_prepare(X, src_vocab, tgt_vocab)
    encoder_embeddings = model.encoder_embedding(encoder_input_ids)
    memory = model.encoder.forward(encoder_embeddings, key_padding_mask=X_key_padding_mask)

    generated_sequence = []
    eos_token_id = tgt_vocab[config.eos_token]
    for _ in range(config.generated_sequence_max_length):
        decoder_embeddings = model.decoder_embedding(decoder_input_ids)
        decoder_logits: torch.Tensor = model.decoder.forward(decoder_embeddings,memory, memory_key_padding_mask=X_key_padding_mask)[0][-1]
        decoder_logits = model.classfier(decoder_logits)
        best_token = decoder_logits.argmax().view(1,1)
        decoder_input_ids = torch.concat([decoder_input_ids,best_token],dim=-1)

        if best_token.item() == eos_token_id:
            break
        generated_sequence.append(best_token.item())
    return ' '.join(tgt_vocab.to_tokens(generated_sequence))


if __name__ == "__main__":
    inputs = 'I love you '
    model = Transformer(
        config.d_model,
        config.n_head,
        config.n_layers,
        config.target_sequence_length,
        config.vocab_size,
        config.device,
        config.encoder_num_embeddings,
        config.decoder_num_embeddings)
    model.load_state_dict(torch.load(config.model_state_dict_save_path+'/transformer_epoch16_loss_0.07695',map_location='cpu'))
    for layer in model.decoder.TransformerDecoder:  # 预测的时候不用遮掩
        layer.masked_multi_head_attention_layer.masked_matrix = None

    _, src_vocab, tgt_vocab = load_data_nmt(config.batch_size, config.num_steps, config.num_examples)

    translate_result: str = predict(model, inputs, src_vocab, tgt_vocab)
    print(translate_result)
