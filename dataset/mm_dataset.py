import os
import re
import json
import copy
import random
from glob import glob
from PIL import Image

from pycocotools.coco import COCO
from torch.utils.data import Dataset

from transformers import BertTokenizer, CLIPTokenizer

class COCO_Dataset(Dataset):
    def __init__(self, args, img_dir, ann_dir, transform):
        self.args = args
        self.transform = transform
        self.img_dir = img_dir
        self.coco = COCO(ann_dir)
        self.ids = list(self.coco.anns.keys())
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ann_ids = self.ids[idx]
        caption = self.coco.anns[ann_ids]['caption']
        img_id = self.coco.anns[ann_ids]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        ori_img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')

        s = min(ori_img.size)
        r = self.args.img_size / s
        s = (round(r * ori_img.size[1]), round(r * ori_img.size[0]))
        resized_img = ori_img.resize(s)
        image, trans_ori_img = self.transform(resized_img)

        tokens = self.tokenizer(caption, add_special_tokens=True, return_token_type_ids=False, padding='max_length', truncation=True, max_length=self.args.token_length, return_tensors='pt')
        input_ids = tokens['input_ids'].squeeze()
        original_input_ids = copy.deepcopy(input_ids)
        attn_masks = tokens['attention_mask'].squeeze()
        if self.args.train_or_eval and self.args.rand_drop in ['input_masking_8_patch_loss_whole']:
            only_tokens = self.tokenizer(caption, add_special_tokens=True, return_token_type_ids=False,
                                         truncation=True, max_length=self.args.token_length, return_tensors='pt')['input_ids'].squeeze()
            for i in random.sample(range(len(only_tokens)), int(len(only_tokens) * self.args.drop_ratio)):
                only_tokens[i] = self.tokenizer.vocab['[MASK]']
            input_ids[:len(only_tokens)] = only_tokens
        return trans_ori_img, image, original_input_ids, input_ids, attn_masks, caption


class CUB_Dataset(Dataset):
    def __init__(self, args, base_folder, transform, train=True):
        self.args = args
        self.train = train
        self.transform = transform
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.base_folder = base_folder
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if args.dataset == 'cub':
            if train:
                    data = args.cub_train_dset
            else:
                data = args.cub_valid_dset
            self.data = [json.loads(l) for l in open(data)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_filename = self.data[idx]['img_id']
        caption = self.data[idx]['caption']

        img_filename = img_filename.replace('.jpg', '.png')
        img_path = os.path.join(self.base_folder, img_filename)

        ori_img = Image.open(img_path).convert('RGB')

        image, trans_ori_img = self.transform(ori_img)

        tokens = self.tokenizer(caption, add_special_tokens=True, return_token_type_ids=False, padding='max_length',
                                truncation=True, max_length=self.args.token_length,
                                return_tensors='pt')
        input_ids = tokens['input_ids'].squeeze()
        original_input_ids = copy.deepcopy(input_ids)
        attn_masks = tokens['attention_mask'].squeeze()
        if self.args.train_or_eval and self.args.rand_drop in ['input_masking_8_patch_loss_whole']:
            only_tokens = self.tokenizer(caption, add_special_tokens=True, return_token_type_ids=False, truncation=True, return_tensors='pt')['input_ids'].squeeze()
            for i in random.sample(range(len(only_tokens)), int(len(only_tokens) * self.args.drop_ratio)):
                only_tokens[i] = self.tokenizer.vocab['[MASK]']
            input_ids[:len(only_tokens)] = only_tokens
        return trans_ori_img, image, original_input_ids, input_ids, attn_masks, caption

class MNIST_Dataset(Dataset):
    def __init__(self, args, base_folder, transform, train=True, test_dset_same=True):
        self.args = args
        self.train = train
        self.transform = transform

        self.base_folder = base_folder
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.special = re.compile(r'[^ A-Za-z0-9가-힣+]')
        self.color = ['white', 'red', 'green', 'blue']
        self.digit = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        if train:
            self.img_data = glob(self.base_folder + '/mnist_train_img/*.png')
            self.txt_data = glob(self.base_folder + '/mnist_train_text/*.txt')
        else:
            self.img_data = glob(self.base_folder + '/mnist_test_img/*.png')
            self.txt_data = glob(self.base_folder + '/mnist_test_text/*.txt')

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_filename = self.img_data[idx]
        txt_filename = self.txt_data[idx]

        ori_img = Image.open(img_filename).convert('RGB')
        image, trans_ori_img = self.transform(ori_img)
        caption = open(txt_filename, 'r').readline()

        tokens = self.tokenizer(caption, add_special_tokens=True, return_token_type_ids=False,
                                padding='max_length', truncation=True, max_length=self.args.token_length, return_tensors='pt')
        input_ids = tokens['input_ids'].squeeze()
        original_input_ids = copy.deepcopy(input_ids)
        attn_masks = tokens['attention_mask'].squeeze()
        if self.args.train_or_eval and self.args.rand_drop in ['input_masking_8_patch_loss_whole']:
            only_tokens = self.tokenizer(caption, add_special_tokens=True, return_token_type_ids=False, truncation=True, return_tensors='pt')['input_ids'].squeeze()
            for i in random.sample(range(len(only_tokens)), int(len(only_tokens) * self.args.drop_ratio)):
                only_tokens[i] = self.tokenizer.vocab['[MASK]']
            input_ids[:len(only_tokens)] = only_tokens
        return trans_ori_img, image, original_input_ids, input_ids, attn_masks, caption

class Flower_Dataset(Dataset):
    def __init__(self, args, transform, train=True):
        self.args = args
        self.train = train
        self.transform = transform

        self.img_path = '/home/data_storage/oxfordflower102/jpg'
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        if args.dataset == 'flower':
            if train:
                self.data = [json.loads(l) for l in open('/home/data_storage/oxfordflower102/json_dset/flower_train.json')]
            else:
                self.data = [json.loads(l) for l in open('/home/data_storage/oxfordflower102/json_dset/flowert_valid.json')]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_filename = self.data[idx]['img_id']
        caption = self.data[idx]['caption']
        label = self.data[idx]['label']

        ori_img = Image.open(os.path.join(self.img_path, img_filename)).convert('RGB')
        image, trans_ori_img = self.transform(ori_img)
        tokens = self.tokenizer(caption, add_special_tokens=True, return_token_type_ids=False,
                                padding='max_length', truncation=True, max_length=self.args.token_length, return_tensors='pt')

        input_ids = tokens['input_ids'].squeeze()
        original_input_ids = copy.deepcopy(input_ids)
        attn_masks = tokens['attention_mask'].squeeze()
        if self.args.train_or_eval and self.args.rand_drop in ['input_masking_8_patch_loss_whole']:
            only_tokens = self.tokenizer(caption, add_special_tokens=True, return_token_type_ids=False,
                                         truncation=True, max_length=self.args.token_length, return_tensors='pt')['input_ids'].squeeze()
            for i in random.sample(range(len(only_tokens)), int(len(only_tokens) * self.args.drop_ratio)):
                only_tokens[i] = self.tokenizer.vocab['[MASK]']
            input_ids[:len(only_tokens)] = only_tokens

        return trans_ori_img, image, original_input_ids, input_ids, attn_masks, caption
