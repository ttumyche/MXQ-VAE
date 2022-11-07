import sys
sys.path.append('..')
import pickle
import numpy as np

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class Txt_1D_CNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.ignore_idx = self.tokenizer.vocab['[PAD]']  # 0
        self.mask_token = self.tokenizer.vocab['[MASK]']
        self.len_vocab = len(self.tokenizer.vocab)
        if args.bert_embedding_load:
            self.bert = BertModel.from_pretrained(self.args.bert_model_size)
        else:
            args.config.hidden_size = args.txt_word_embed_hidden_dim
            self.bert = BertModel(args.config)

        self.softmax = nn.LogSoftmax(dim=-1)
        channels = args.txt_word_embed_hidden_dim
        num_resnet_blocks = 0
        has_resblocks = num_resnet_blocks > 0

        codebook_dim = args.cb_dim
        enc_chans = np.linspace(channels, channels * 4, args.txt_num_layer).astype(int)
        dec_chans = list(reversed(enc_chans[1:-1]))

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, channels * 4, *dec_chans]
        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            stride = 1
            padding = 0
            if args.token_length == 80:
                filter_size = [16, 16]
            elif args.token_length == 68:
                filter_size = [14, 14]
            elif args.token_length == 64:
                filter_size = [15, 15]
            enc_layers.append(nn.Sequential(nn.Conv1d(enc_in, enc_out, filter_size[0], stride=stride, padding=padding), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose1d(dec_in, dec_out, filter_size[1], stride=stride, padding=padding), nn.ReLU()))

        enc_last_hdim = codebook_dim

        enc_layers.append(nn.Conv1d(enc_chans[-1], enc_last_hdim, 1))
        dec_layers.append(nn.Conv1d(dec_chans[-1], channels, 1))
        dec_layers.append(nn.Conv1d(channels, self.len_vocab, 1))  # to vocab size

        self.txt_1d_encoder = nn.Sequential(*enc_layers)
        self.txt_1d_decoder = nn.Sequential(*dec_layers)

    def forward(self, original_input_ids, input_ids, attn_mask, z):
        if z is None:
            embedded_txt = self.bert.embeddings.word_embeddings(input_ids).transpose(1, 2).contiguous()
            proj_embeds = self.txt_1d_encoder(embedded_txt)  # [bsz, #cb, reduced_txt_len]
            return proj_embeds

        else:
            dec_output = self.txt_1d_decoder(z.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            decoded_ids = torch.argmax(self.softmax(dec_output), dim=-1)
            decoded_txt = self.tokenizer.batch_decode(decoded_ids)

            if self.args.rand_drop in ['input_masking_8_patch_loss_whole'] and self.args.drop_ratio != 0.0:
                acc = (decoded_ids[original_input_ids != self.ignore_idx] == original_input_ids[original_input_ids != self.ignore_idx]).float().mean()
                loss = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)(dec_output.view(-1, self.len_vocab), original_input_ids.view(-1))
            else:
                acc = (decoded_ids[input_ids != self.ignore_idx] == input_ids[input_ids != self.ignore_idx]).float().mean()
                loss = nn.CrossEntropyLoss(ignore_index=self.ignore_idx)(dec_output.view(-1, self.len_vocab), input_ids.view(-1))

            return loss, acc, decoded_txt, decoded_txt, dec_output
