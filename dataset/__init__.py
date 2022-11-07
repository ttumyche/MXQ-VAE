from typing import Optional

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from augmentations import train_aug
from .mm_dataset import COCO_Dataset, CUB_Dataset, MNIST_Dataset, Flower_Dataset


def get_dataset(args, dataset, img_dir, ann_dir, transform, train, test_dset_same):
    if dataset == 'coco':
        dataset = COCO_Dataset(args, img_dir, ann_dir, transform)
    elif dataset in ['cub']:
        dataset = CUB_Dataset(args, args.base_folder, transform, train)
    elif dataset == 'mnist':
        dataset = MNIST_Dataset(args, args.MNIST_dset, transform, train, test_dset_same)
    elif dataset in ['flower']:
        dataset = Flower_Dataset(args, transform, train)
    return dataset

def collate(batch):
    ori_imgs = []
    trans_imgs = []
    original_input_ids = []
    input_ids = []
    attn_masks = []
    captions = []

    for sample in batch:
        ori_imgs.append(sample[0])
        trans_imgs.append(sample[1])
        original_input_ids.append(sample[2])
        input_ids.append(sample[3])
        attn_masks.append(sample[4])
        captions.append(sample[5])

    return ori_imgs, torch.stack(trans_imgs, 0), torch.stack(original_input_ids, 0), torch.stack(input_ids, 0), torch.stack(attn_masks, 0), captions

class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            self.train_dataset = get_dataset(args=self.args, dataset=self.args.dataset, img_dir=self.args.coco_tr_img_path,
                                             ann_dir=self.args.coco_tr_ann_path, transform=train_aug(self.args), train=True, test_dset_same=True)
            self.valid_dataset = get_dataset(args=self.args, dataset=self.args.dataset, img_dir=self.args.coco_val_img_path,
                                             ann_dir=self.args.coco_val_ann_path, transform=train_aug(self.args),
                                             train=False, test_dset_same=True)
        if stage in (None, 'test'):
            self.test_dataset = get_dataset(args=self.args, dataset=self.args.dataset, img_dir=self.args.coco_val_img_path,
                                             ann_dir=self.args.coco_val_ann_path, transform=train_aug(self.args),
                                             train=False, test_dset_same=True)

    def train_dataloader(self):
        train_num_subset = list(range(0, int(len(self.train_dataset) * self.args.subset_ratio)))

        train_sub_datasets = torch.utils.data.Subset(self.train_dataset, train_num_subset)
        train_loader = DataLoader(train_sub_datasets, batch_size=self.args.train_bsz, pin_memory=True, num_workers=self.args.num_workers,
                                  collate_fn=collate, shuffle=True, drop_last=True)
        return train_loader

    def val_dataloader(self):
        valid_num_subset = list(range(0, int(len(self.valid_dataset) * self.args.valid_subset_ratio)))
        valid_sub_datasets = torch.utils.data.Subset(self.valid_dataset, valid_num_subset)

        valid_loader = DataLoader(valid_sub_datasets, batch_size=self.args.eval_bsz, pin_memory=True,
                                  num_workers=self.args.num_workers, collate_fn=collate, shuffle=False,
                                  drop_last=True)

        return valid_loader

    def test_dataloader(self):
        test_num_subset = list(range(0, int(len(self.test_dataset) * self.args.valid_subset_ratio)))
        test_sub_datasets = torch.utils.data.Subset(self.test_dataset, test_num_subset)

        test_loader = DataLoader(test_sub_datasets, batch_size=self.args.eval_bsz, pin_memory=True,
                                 num_workers=self.args.num_workers, collate_fn=collate, shuffle=False, drop_last=True)
        return test_loader
