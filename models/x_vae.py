import os
import math
import copy
import random
import matplotlib
matplotlib.use('Agg')
from einops import rearrange

import torch
import torchvision
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import BertModel, BertConfig, GPT2Config
from .gpt2 import GPT2LMHeadModel
from transformers.optimization import get_cosine_schedule_with_warmup, get_constant_schedule

from utils import scheduler
from .img_nn import Img_2D_CNN
from .tst import Txt_1D_CNN

norm = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

def denormalize(x, mean=norm[0], std=norm[1]):
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)

    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

class MODEL(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.config = BertConfig.from_pretrained(args.bert_model_size)
        if args.step2 == 'none':
            self.model = X_BERT_VQVAE(args)

        elif args.step2 in ['transformer', 'transformer_generation']:
            self.model = Transformer_Prior(args)

    def forward(self, x):
        return None

    def training_step(self, batch, batch_idx):
        train_loader = self.train_dataloader()
        it = len(train_loader) * self.current_epoch + batch_idx

        self.beta_scheduler = scheduler('cosine_warmup', self.args.beta_start, self.args.beta_end,
                                        self.args.epochs, len(train_loader),
                                        warmup_epochs=self.args.beta_warmup_epoch,
                                        start_warmup_value=self.args.beta_start_warmup_value)
        self.args.beta = self.beta_scheduler[it]
        self.log('beta', self.args.beta, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False)

        ori_img, img, original_input_ids, input_ids, attn_masks, caption = batch
        losses, _ = self.model('train', 'none', batch_idx, img, original_input_ids, input_ids, attn_masks, caption,
                               return_logits=False, return_loss=True, return_recons=True)

        for k, v in losses.items():
            if k in ['bits_per_dim', 'z_idx', 'z_idx_i', 'z_idx_t', 'z_idx_share', 'logits_per_img', 'logits_per_txt']:
                continue
            k = 'tr_' + k
            self.log(k, v, on_step=True, on_epoch=False, logger=True, prog_bar=False, sync_dist=False)
        return losses['total_loss']

    def configure_optimizers(self):
        train_loader = self.train_dataloader()
        optimizer = AdamW(self.parameters(), lr=self.args.base_lr, betas=(self.args.beta1, self.args.beta2), eps=self.args.eps, weight_decay=self.args.weight_decay)
        if self.args.scheduler == 'cosine_warmup':
            lr_scheduler = {
                'scheduler':
                    get_cosine_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=self.args.warmup_epochs * len(train_loader),
                    num_training_steps=self.args.epochs * len(train_loader)),
                'interval': 'step',
            }
        elif self.args.scheduler == 'constant':
            lr_scheduler = {
                'scheduler': get_constant_schedule(optimizer=optimizer),
                'interval': 'step',
            }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class X_BERT_VQVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.img_enc_dec = Img_2D_CNN(args)
        self.txt_enc_dec = Txt_1D_CNN(args)

        # BERT
        custom_bert_tiny_config = copy.deepcopy(args.config)
        custom_bert_tiny_config.num_hidden_layers = args.x_bert_vqvae_bert_layer
        custom_bert_tiny_config.hidden_size = args.cb_dim
        self.BERT = BertModel(custom_bert_tiny_config)

        if args.img_size == 128:
            self.z_img = 16
            z_img = 16

            h_dim = 128
        elif args.img_size == 64:
            self.z_img = 8
            z_img = 8

        elif args.token_length == 80:
            self.z_txt = 20
            z_txt = 20
        elif args.token_length == 68:
            self.z_txt = 16
            z_txt = 16
        elif args.token_lengt == 64:
            self.z_txt = 8
            z_txt = 8

            h_dim = 32

        self.Layer_img = nn.Linear(z_img ** 2 + z_txt, z_img ** 2)
        self.Layer_txt = nn.Sequential(
            nn.Linear(z_img ** 2 + z_txt, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_txt)
        )

        self.avgpool_1d_img = nn.AdaptiveAvgPool1d(1)
        self.avgpool_1d_txt = nn.AdaptiveAvgPool1d(1)

        self.l1_dist = nn.PairwiseDistance(p=1)
        self.mse_dist = nn.PairwiseDistance(p=2)

        self.num_codebook = args.num_codebook
        self.codebook = nn.Embedding(args.num_codebook, args.cb_dim)
        self.codebook.weight.data.uniform_(-1.0 / self.num_codebook, 1.0 / self.num_codebook)

        self.cross_entropy = nn.CrossEntropyLoss()

    def get_extended_attn_mask(self, bsz, h, w, t, device):
        extended_attn_mask = torch.ones(h * w + t, h * w + t, dtype=torch.long, device=device)
        extended_attn_masks = extended_attn_mask.repeat(bsz, 1, 1)
        return (1.0 - extended_attn_masks.unsqueeze(1)) * - 10000.0

    def quantize(self, step, z):
        bsz, t, dim = z.shape
        z = rearrange(z, 'b t d -> (b t) d')

        d = z.pow(2).sum(1, keepdim=True) + \
            self.codebook.weight.pow(2).sum(1) + \
            - 2 * z @ self.codebook.weight.t()

        min_encoding_idx = torch.argmin(d, dim=1)
        z_q = self.codebook(min_encoding_idx).view(z.shape)

        b_min_idx = rearrange(min_encoding_idx, '(b t) -> b t', t=t)

        encodings = torch.zeros(min_encoding_idx.shape[0], self.args.num_codebook, device=z.device)
        encodings.scatter_(1, min_encoding_idx.unsqueeze(1), 1)

        loss_vq = F.mse_loss(z_q, z.detach())
        loss_commit = F.mse_loss(z, z_q.detach())

        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, '(b t) d -> b t d', b=bsz)

        return loss_vq, loss_commit, z_q, b_min_idx

    def forward(self, step, logger, batch_idx, img, original_input_ids, input_ids, attn_masks, caption, return_logits=False, return_loss=False, return_recons=False):
        if step not in ['test'] and self.args.rand_drop in ['input_masking_8_patch_loss_whole']:
            _, _, h, w = img.size()

            patch_size = int(self.args.rand_drop.split('_')[2])
            rearange_img = copy.deepcopy(rearrange(img, 'b d (h p1) (w p2) -> b (h w) (p1 p2 d)', p1=patch_size, p2=patch_size))
            bsz, seq_len, dim = rearange_img.size()
            rand = torch.randint((seq_len - 1), (bsz, int(seq_len * self.args.drop_ratio)))
            rearange_img[torch.arange(bsz).unsqueeze(1), rand, :] = 0.
            img_rand_drop = rearrange(rearange_img, 'b (h w) (p1 p2 d) -> b d (h p1) (w p2)',
                                      p1=patch_size, p2=patch_size, h=h//patch_size, w=w//patch_size)
            logits_i = self.img_enc_dec(img_rand_drop, None)

        else:
            logits_i = self.img_enc_dec(img, None)

        bsz, dim, h, w = logits_i.shape
        rearange_i = rearrange(logits_i, 'b d h w -> b (h w) d')

        logits_t = self.txt_enc_dec(original_input_ids, input_ids, attn_masks, None)  # bsz, dim, len
        _, _, t = logits_t.shape
        rearange_t = rearrange(logits_t, 'b d t -> b t d')

        concat_enc_input = torch.cat([rearange_i, rearange_t], dim=1)
        extended_attn_mask = self.get_extended_attn_mask(bsz, h, w, t, img.device)
        concat_i_t = self.BERT.encoder(concat_enc_input, attention_mask=extended_attn_mask, output_attentions=True).last_hidden_state

        loss_vq, loss_commit, z_q, b_min_idx = self.quantize(step, concat_i_t)

        z_q_i = rearrange(self.Layer_img(z_q.transpose(1, 2).contiguous()).transpose(1, 2).contiguous(), 'b (h w) d -> b d h w', h=h)
        z_q_t = self.Layer_txt(z_q.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        recon_i = self.img_enc_dec(img, z_q_i)
        recon_loss_i_mse = F.mse_loss(recon_i, img)

        re_img = rearrange(img, 'b c h w -> (b h w) c')
        re_recon_img = rearrange(recon_i, 'b c h w -> (b h w) c')

        mean_l1 = self.l1_dist(re_img, re_recon_img).mean()
        mean_l2 = self.mse_dist(re_img, re_recon_img).mean()
        recon_loss_t, acc_t, decoded_txt, recon_t, dec_output = self.txt_enc_dec(original_input_ids, input_ids, attn_masks, z_q_t)

        total_loss = recon_loss_i_mse + recon_loss_t + loss_vq + self.args.beta * loss_commit

        losses = {'matching_acc': acc_t.mean(),
                  'img_l1': mean_l1.mean(), 'img_l2': mean_l2.mean(),
                  'recon_loss_i': recon_loss_i_mse.mean(),
                  'recon_loss_t': recon_loss_t.mean(),
                  'embedding_loss': loss_vq.mean(),
                  'commit_loss': loss_commit.mean(),
                  'commit_loss_beta': loss_commit.mean() * self.args.beta,
                  'z_idx_share': b_min_idx,
                  'total_loss': total_loss.mean()}

        denorm_img = denormalize(recon_i.detach())
        decoded_txt = list()
        for seq_id in recon_t.tolist():
            decoded_txt.append(' '.join([self.idx2vocab[w] for w in seq_id]))
        recons = {'recon_i': denorm_img, 'ori_recon_i': recon_i, 'tf_decoded_txt': decoded_txt, 'recon_t': decoded_txt}

        return losses, recons

class Transformer_Prior(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.img_size == 128:
            self.z_img = 16
        elif args.img_size == 64:
            self.z_img = 8

        if args.token_length == 80:
            self.z_txt = 20
        elif args.token_length == 68:
            self.z_txt = 16
        elif args.token_length == 64:
            self.z_txt = 8

        self.model = X_BERT_VQVAE(args)
        if self.args.model_load_vq is not None:
            ckpt = torch.load(args.model_load_vq)
            new_k = list(map(lambda i: '.'.join(i.split('.')[1:]), ckpt['state_dict'].keys()))
            new_ckpt = dict(zip(new_k, ckpt['state_dict'].values()))
            self.model.load_state_dict(new_ckpt)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self.sos_tok = args.num_codebook
        self.cb_w_sos = nn.Embedding(args.num_codebook + 1, args.cb_dim)
        self.cb_w_sos.weight.data[:-1, ] = self.model.codebook.weight.data
        self.step2_cb_w_sos = nn.Embedding(1 + args.num_codebook, args.step2_cb_dim)

        config = GPT2Config.from_pretrained('gpt2')
        config.add_cross_attention = False
        config.n_layer = args.decoder_num_layers
        config.n_head = args.decoder_num_attn_heads
        config.vocab_size = args.num_codebook + 1

        config.n_embd = args.step2_cb_dim

        config.n_positions = self.z_img ** 2 + self.z_txt + 1
        config.n_ctx = self.z_img ** 2 + self.z_txt + 1

        self.prior = GPT2LMHeadModel(config, args, None, self.step2_cb_w_sos, None)

        self.softmax = nn.LogSoftmax(dim=-1)
        self.ce_loss = nn.CrossEntropyLoss()

        # self.args.step2 = 'transformer_generation'
        # self.args.top_p = [1.0]
        # self.args.top_k = [10]
        # self.args.temperature = [1.0]

    def forward(self, step, logger, batch_idx, img, original_input_ids, input_ids, attn_masks, caption, return_logits=False, return_loss=False, return_recons=False):
        if self.args.step2 in ['transformer_generation']:
            print(f'top_p_{self.args.top_p}_top_k_{self.args.top_k}_temp_{self.args.temperature}')

            top_p_candi = self.args.top_p
            top_k_candi = self.args.top_k
            temp_candi = self.args.temperature
            for top_p in top_p_candi:
                for top_k in top_k_candi:
                    for temp in temp_candi:
                        self.gen_samples = 0
                        gen_samples = 10000

                        self.folder_name = self.args.now + '_' + self.args.dataset
                        while self.gen_samples != gen_samples:
                            with torch.no_grad():
                                bsz = 1
                                h = self.z_img
                                bos_tok = torch.zeros(bsz, 1, dtype=torch.long, device=self.prior.device) + self.sos_tok
                                prompt = bos_tok

                                max_len = self.z_img ** 2 + self.z_txt + 1

                                q_z_idx = self.prior.generate(prompt, max_length=max_len, do_sample=True, temperature=temp, top_k=top_k, top_p=top_p)[:, 1:]
                                z_q = self.cb_w_sos(q_z_idx.flatten())
                                z_q = rearrange(z_q, '(b t) d -> b t d', b=bsz)

                                z_q_i = rearrange(self.model.Layer_img(z_q.transpose(1, 2).contiguous()).transpose(1, 2).contiguous(), 'b (h w) d -> b d h w', h=h)
                                z_q_t = self.model.Layer_txt(z_q.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

                                recon_i = self.model.img_enc_dec(img, z_q_i)
                                _, acc_t, decoded_txt, recon_t, bits_per_dim, _ = self.model.txt_enc_dec(original_input_ids, input_ids, attn_masks, z_q_t)
                                generated_txt = recon_t[0]

                                sampling_method = f'p_{top_p}_k_{top_k}_t_{temp}'
                                generated_img = denormalize(recon_i.detach())[0]
                                self.gen_samples += 1
                                save_path = os.path.join('gen_samples', self.folder_name, sampling_method, 'txt')
                                save_path_img = os.path.join('gen_samples', self.folder_name, sampling_method, 'img')
                                os.makedirs(save_path, exist_ok=True)
                                os.makedirs(save_path_img, exist_ok=True)
                                with open(save_path + str(self.gen_samples) + '.txt', 'w') as f:
                                    f.write(generated_txt)
                                torchvision.utils.save_image(generated_img, save_path_img + str(self.gen_samples) + '.png')

        else:
            with torch.no_grad():
                logits_i = self.model.img_enc_dec(img, None)
                logits_t = self.model.txt_enc_dec(original_input_ids, input_ids, attn_masks, None)

                bsz, dim, h, w = logits_i.shape

                rearange_i = rearrange(logits_i, 'b d h w -> b (h w) d')
                _, _, t = logits_t.shape
                rearange_t = rearrange(logits_t, 'b d t -> b t d')

                concat_i_t = self.model.BERT.encoder(torch.cat([rearange_i, rearange_t], dim=1), output_attentions=True).last_hidden_state
                _, _, z_q, b_min_idx = self.model.quantize(step, concat_i_t)

            b_min_idx_ = torch.cat([torch.zeros(bsz, 1, dtype=torch.long, device=z_q.device) + self.sos_tok, b_min_idx], 1)
            dec_output = self.prior(input_ids=b_min_idx_, labels=b_min_idx_.detach(), output_attentions=True)
            loss = dec_output.loss

            prediction_scores = dec_output.logits[:, :-1, :].contiguous()  # bsz, seq_len, vocab size
            pred_idx = torch.argmax(self.softmax(prediction_scores), dim=-1)
            acc = (pred_idx == b_min_idx).float().mean().item() * 100

            losses = {'total_loss': loss.mean(), 'step2_txt_matching_acc': acc}
            return losses, None