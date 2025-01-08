import ast
import itertools
import os
import random
from collections import defaultdict
from typing import Iterable, Iterator, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import pandas as pd

import Util

label_str2int_dict = {
    "real": 0,
    "fake": 1,
    "other": 2,
    '真':0,
    '假':1,
    '其他':2
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
        self.df['id'] = self.df['id'].apply(str)
        self.df['image_id'] = self.df['image_id'].apply(str)

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

    def __getitems__(self,indices):
        return self.df.iloc[indices].to_dict(orient='records')



class InfiniteBalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.label_to_indices = defaultdict(list)

        for idx, item in enumerate(self.dataset):
            label = label_str2int_dict[item['label']]
            self.label_to_indices[label].append(idx)

        self.real_shot_nums = batch_size // 2
        self.fake_shot_nums = batch_size - self.real_shot_nums


        if len(self.label_to_indices[0]) < self.real_shot_nums:
            raise ValueError('real 样本不足')
        if len(self.label_to_indices[1]) < self.fake_shot_nums:
            raise ValueError('fake 样本不足')



    def __iter__(self):
        while True:  # Infinite loop to provide infinite batches
            real_perm = list(self.label_to_indices[0])
            fake_perm = list(self.label_to_indices[1])

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



class FewShotDataLoader(Iterator):
    def __init__(self, dataset, batch_sampler):
        """
        :param dataset: 实现了 __getitem__ 和 __len__ 方法的数据集对象。
        :param batch_sampler: 提供索引批次的采样器对象。
        """
        self.dataset = dataset
        self.batch_sampler_iter = iter(batch_sampler)

    def __next__(self) -> List[Dict[str, Any]]:
        """
        返回下一个批次的数据。
        """
        try:
            batch_indices = next(self.batch_sampler_iter)
            # 使用 __getitems__ 方法来获取批量数据
            batch_data = self.dataset.__getitems__(batch_indices)
            return batch_data
        except StopIteration:
            raise StopIteration("No more batches available.")






def load_text_few_shot_data(few_shot_dir,num_few_shot,language,rationale_name):
    few_shot_file_path = f'{few_shot_dir}/{language}_{rationale_name}_shot.csv'
    dataset = FakeNewsTextRationaleFewShotDataset(
            pd.read_csv(few_shot_file_path)
        )
    return FewShotDataLoader(dataset,InfiniteBalancedBatchSampler(dataset, num_few_shot))



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
    [{
        'id':,
        'image_url':,
        "text":,
        'label':,
        "publish_date":,
        'image_id':
    }]
    """
    def image_file_name_list2image_url(image_file_name_list):
        return f'file://{root_path}/{ast.literal_eval(image_file_name_list)[0]}'

    def get_image_id(image_file_name_list):
        return ast.literal_eval(image_file_name_list)[0].split('/')[-1].split('.')[0]

    def collect_fn(batch):
        return list(batch)

    data_dir = root_path
    file_path = f'{data_dir}/weibo.csv'
    df = pd.read_csv(file_path)
    df['text'] = df.apply(lambda x : f'{x["text"]} --发布来源：{x["release_source"]}', axis=1)
    df['image_url'] = df['available_image_paths'].apply(image_file_name_list2image_url)
    df['publish_date'] = np.nan
    df['image_id'] = df['available_image_paths'].apply(get_image_id)
    dataset = ImageTextPairDataset(df)
    return DataLoader(dataset, batch_size,False,num_workers=4,collate_fn=collect_fn)


def load_data(dataset,root_path,batch_size=1):
    if dataset == 'gossipcop':
        return load_en_image_text_pair_goss(root_path,batch_size),'en'
    elif dataset == 'twitter':
        return load_twitter_data(root_path,batch_size),'en'
    elif dataset == 'weibo':
        return load_weibo_data(root_path,batch_size),'zh'
