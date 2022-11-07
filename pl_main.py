import os
import sys
import argparse
from datetime import datetime

import pytorch_lightning as pl

from utils import set_seed
from models import MODEL
from dataset import DataModule

def train(args):
    set_seed(args.seed)
    pl.seed_everything(seed=args.seed, workers=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.save_path + str(args.dataset) + '_' + str(args.now),
                                                       filename='{epoch:02d}-{eval_total_loss: .2f}',
                                                       monitor='tr_total_loss', mode='min', verbose=True,
                                                       save_top_k=int(args.epochs / args.model_save_interval),
                                                       every_n_epochs=args.model_save_interval)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')

    data = DataModule(args=args)
    model = MODEL(args=args)


    trainer = pl.Trainer(gpus=-1, accelerator='ddp',
                         max_epochs=args.epochs, val_check_interval=1.0,
                         terminate_on_nan=True,
                         checkpoint_callback=True, resume_from_checkpoint=args.model_load,
                         callbacks=[checkpoint_callback, lr_monitor],
                         num_sanity_val_steps=0, log_every_n_steps=1, flush_logs_every_n_steps=10)
    trainer.fit(model, data)

    if model.global_rank != 0:
        sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_or_eval", type=bool, default=True, help='train(True), eval(False)')
    parser.add_argument("--eval_model_path", type=str, default=None, help='model path to eval')

    parser.add_argument("--step2", type=str, default='none', choices=['none', 'transformer', 'transformer_generation'])
    parser.add_argument("--rand_drop", type=str, default="none", choices=['none', 'input_masking_8_patch_loss_whole'])
    parser.add_argument("--drop_ratio", type=float, default=0.3)

    parser.add_argument("--img_num_residual_layer", type=int, default=6)
    parser.add_argument("--txt_num_layer", type=int, default=5)

    parser.add_argument("--x_bert_vqvae_bert_layer", type=int, default=4)
    parser.add_argument("--attn_temp", type=float, default=1.0)

    parser.add_argument("--bert_embedding_load", type=bool, default=False, help='load(T), scratch(F)')
    parser.add_argument("--bert_model_size", type=str, default='google/bert_uncased_L-2_H-128_A-2',
                        choices=['bert-base-uncased',  # base
                                 'google/bert_uncased_L-4_H-512_A-8',  # small
                                 'google/bert_uncased_L-4_H-256_A-4',  # mini
                                 'google/bert_uncased_L-2_H-128_A-2'])  # tiny
    parser.add_argument("--txt_word_embed_hidden_dim", type=int, default=128)

    parser.add_argument("--dataset", type=str, default="mnist", choices=["coco", "cub", "mnist", "flower"])
    parser.add_argument("--num_codebook", type=int, default=256)
    parser.add_argument("--cb_dim", type=int, default=128)
    parser.add_argument("--beta_start", type=float, default=0.25)
    parser.add_argument("--beta_end", type=float, default=0.25)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--token_length", type=int, default=64)

    parser.add_argument("--scheduler", type=str, default='cosine_warmup', choices=['cosine_warmup'])

    parser.add_argument("--subset_ratio", type=float, default=1.0, help='subset of train data')
    parser.add_argument("--valid_subset_ratio", type=float, default=1.0,)

    parser.add_argument("--model_load", type=str, default=None, help='restart')
    parser.add_argument("--model_load_vq", type=str, default=None, help='vq model path for step2')

    parser.add_argument("--decoder_num_layers", type=int, default=8)
    parser.add_argument("--decoder_num_attn_heads", type=int, default=8)
    parser.add_argument("--step2_cb_dim", type=int, default=512)

    parser.add_argument("--top_k", type=list, default=[10])
    parser.add_argument("--top_p", type=list, default=[1.0])
    parser.add_argument("--temperature", type=list, default=[1.0])
    parser.add_argument("--num_return_sequence", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=None, help='number of epochs')
    parser.add_argument("--base_lr", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=0)

    parser.add_argument("--train_bsz", type=int, default=None, help="number of batch size")
    parser.add_argument("--eval_bsz", type=int, default=None, help="number of batch size")

    parser.add_argument("--beta_warmup_epoch", type=float, default=0)
    parser.add_argument("--beta_start_warmup_value", type=float, default=0)

    parser.add_argument("--num_workers", type=int, default=50, help="num of workers")
    parser.add_argument("--eval_during_training", type=bool, default=True, help='')

    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    parser.add_argument("--model_save_interval", type=int, default=2, help='')
    # coco dset
    parser.add_argument("--coco_tr_img_path", type=str, default="path/to/coco/img", help="coco train image path")
    parser.add_argument("--coco_tr_ann_path", type=str, default="path/to/coco/ann", help="coco train annotation path")

    parser.add_argument("--base_folder", type=str, default="path/to/dataset", help="CUB dataset folder")

    parser.add_argument("--cub_train_dset", type=str,  default="path/to/cub",  help="CUB tr dset file")
    parser.add_argument("--cub_valid_dset", type=str,  default="path/to/cub", help="CUB val dset file")

    now = datetime.now()
    now = now.strftime("%Y%m%d_%H%M%S")
    parser.add_argument("--now", type=str, default=now)
    parser.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args()

    output_path = args.save_path + str(args.dataset) + '_' + str(args.now)
    os.makedirs(output_path, exist_ok=True)
    os.chmod(args.save_path, 0o777)

    train(args)
