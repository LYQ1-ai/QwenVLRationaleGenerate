import ast
import itertools
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import pandas as pd

import Util

label_dict = {
    "real": 0,
    "fake": 1,
    "other": 2,
    0: 'real',
    1: 'fake',
    2: 'other',
}




class ImageTextPairDataset(Dataset):

    def __init__(self,dataframe):
        """
        dataframe = {
            'id':,
            'image_url':,
            "text":,
            'label':,
            "publish_date":,
            'image_id':
        }
        """
        self.df = dataframe

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        item = self.df.loc[idx]
        return {
            'id': item.id,
            'image_url': item.image_url,
            "text":item.text,
            'label':item.label,
            "publish_date":item.publish_date,
            'image_id':item.image_id
        }


class FakeNewsTextRationaleFewShotDataset(Dataset):

    def __init__(self,dataframe):
        """
        dataframe = {
            'text':str,
            'rationale':str,
            'label': real or fake,
        }
        """
        self.df = dataframe

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()

class InfiniteBalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.label_to_indices = defaultdict(list)

        for idx, item in enumerate(self.dataset):
            label = item['label']
            self.label_to_indices[label].append(idx)

        self.real_shot_nums = batch_size // 2
        self.fake_shot_nums = batch_size - self.real_shot_nums

        # 确保每个标签至少有 batch_size/2 个样本
        if len(self.label_to_indices['real']) < self.real_shot_nums:
            raise ValueError(f"标签 real 的样本数量不足 {self.real_shot_nums}")

        if len(self.label_to_indices['fake']) < self.fake_shot_nums:
            raise ValueError(f"标签 fake 的样本数量不足 {self.fake_shot_nums}")



    def __iter__(self):
        while True:  # Infinite loop to provide infinite batches
            real_perm = list(self.label_to_indices['real'])
            fake_perm = list(self.label_to_indices['fake'])

            random.shuffle(real_perm)
            random.shuffle(fake_perm)

            # Create iterators for each class that will yield indices indefinitely
            real_cycle = iter(itertools.cycle(real_perm))
            fake_cycle = iter(itertools.cycle(fake_perm))

            for _ in range(len(self.dataset) // self.batch_size):
                batch = [next(real_cycle) for _ in range(self.real_shot_nums)] + [next(fake_cycle) for _ in range(self.fake_shot_nums)]
                yield batch

    def __len__(self):
        # Since the sampler is infinite, we cannot provide a meaningful length.
        # However, this method can return the number of batches in one epoch for guidance.
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size







def load_text_few_shot_data(few_shot_dir,num_few_shot,language,rationale_name):
    few_shot_file_path = f'{few_shot_dir}/{language}_{rationale_name}_shot.csv'
    dataset = FakeNewsTextRationaleFewShotDataset(
            pd.read_csv(few_shot_file_path)
        )
    return DataLoader(
        dataset=dataset,
        sampler=InfiniteBalancedBatchSampler(dataset, num_few_shot)
    )


def load_en_image_text_pair_goss(root_path,batch_size = 1):
    data_dir = root_path
    file_path = f'{data_dir}/gossipcop.csv'
    df = pd.read_csv(file_path)
    df['image_id'] = df['id']
    df['image_url'] = df['id'].map(lambda x : f'file://{root_path}/images/{x}_top_img.png')
    dataset = ImageTextPairDataset(df)
    return DataLoader(dataset, batch_size,False,num_workers=4)


def get_twitter_image_url_dict(root_path):
    image_dir = f'{root_path}/images'
    return {
       file.split('.')[0] : f'file://{image_dir}/{file}' for file in os.listdir(image_dir)
    }

def load_twitter_data(root_path,batch_size = 1):
    data_dir = root_path
    file_path = f'{data_dir}/twitter.csv'
    df = pd.read_csv(file_path)
    image_id2url_dict = get_twitter_image_url_dict(root_path)

    df = pd.DataFrame({
        'id':df['post_id'],
        'text':df['post_text'],
        'label':df['label'],
        'publish_date':df['timestamp'],
        'image_id':df['image_id'],
        'image_url': df['image_id'].map(lambda x : image_id2url_dict.get(x)),
    })

    dataset = ImageTextPairDataset(df)
    return DataLoader(dataset, batch_size,False,num_workers=4)

def load_weibo_data(root_path,batch_size=1):
    """
    {
        'id':,
        'image_url':,
        "text":,
        'label':,
        "publish_date":,
        'image_id':
    }
    """
    def image_file_name_list2image_url(image_file_name_list):
        return f'file://{root_path}/{ast.literal_eval(image_file_name_list)[0]}'

    def get_image_id(image_file_name_list):
        return ast.literal_eval(image_file_name_list)[0].split('/')[-1].split('.')[0]

    data_dir = root_path
    file_path = f'{data_dir}/weibo.csv'
    df = pd.read_csv(file_path)
    df['text'] = df.apply(lambda x : f'{x["text"]} --发布来源：{x["release_source"]}', axis=1)
    df['image_url'] = df['available_image_paths'].apply(image_file_name_list2image_url)
    df['publish_date'] = np.nan
    df['image_id'] = df['available_image_paths'].apply(get_image_id)
    dataset = ImageTextPairDataset(df)
    return DataLoader(dataset, batch_size,False,num_workers=4)



