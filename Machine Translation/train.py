import sys
import torch
import d2l.torch as d2l

sys.path.append('X:\Code\Source code of Transformer')

from config import Config
from tqdm.auto import tqdm
from Transformer import Transformer
from torch.nn.functional import cross_entropy
from torch.utils.tensorboard import SummaryWriter
from Cell.criterion_func import MaskedSoftmaxCELoss
from TransfomerTokenzer import load_data_nmt, generate_key_padding_mask


def train():
    config = Config()
    writer = SummaryWriter(log_dir=config.log_dir, comment=config.log_dir_comment)
    model = Transformer(
        config.d_model,
        config.n_head,
        config.n_layers,
        config.target_sequence_length,
        config.vocab_size,
        config.device,
        config.encoder_num_embeddings,
        config.decoder_num_embeddings)
    # model = torch.jit.script(model, example_inputs=[(torch.randn(size=(2, 2, 512)), torch.randn(size=(2, 2, 512)),
    #                                                  torch.ones(size=(2, 2)), torch.ones(size=(2, 2)))])
    model.to(config.device)
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    train_dl, src_vocab, tgt_vocab = load_data_nmt(config.batch_size, config.num_steps,
                                                   num_examples=config.num_examples)
    # criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
    criterion = MaskedSoftmaxCELoss()

    train_dl = tqdm(train_dl, leave=True)
    global_step = 0
    for epoch in range(config.epoch):
        train_dl.set_description(f'Epoch:{epoch}')
        losses = 0
        len_dl = len(train_dl)
        for batch in train_dl:
            X, X_realtkns, Y, Y_realtkns = [value.to(config.device) for value in batch]
            X_key_padding_mask = generate_key_padding_mask(X.shape[0], config.num_steps, X_realtkns).to(config.device)
            Y_key_padding_mask = generate_key_padding_mask(Y.shape[0], config.num_steps, Y_realtkns).to(config.device)
            bos_token = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0]).reshape(-1, 1).to(config.device)
            decoder_input = torch.concat([bos_token, Y[:, :-1]], dim=1)
            logits = model.forward(X, decoder_input, X_key_padding_mask, Y_key_padding_mask, X_key_padding_mask)
            loss = criterion(logits, Y, Y_realtkns)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_dl.set_postfix(loss=loss.item())
            writer.add_scalar('Loss_myfunc_v2', loss.item(), global_step)
            global_step += 1
            losses += loss.item()
        if epoch >= 3:
            torch.save(model.state_dict(),config.model_state_dict_save_path+f'/transformer_epoch{epoch}_loss_{round(losses/len_dl,5)}')



if __name__ == "__main__":
    train()
