import ast
import itertools
import os
import random
from collections import defaultdict
from pprint import pprint
from typing import Iterator, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import pandas as pd

label_str2int_dict = {
    "real": 0,
    "fake": 1,
    "other": 2,
    '真':0,
    '假':1,
    '其他':2
}




class FakeNewsDataset(Dataset):

    def __init__(self,dataframe):
        """
        dataframe: {
            source_id:str,
            image_path:str,
            text:str,
            label:str
        }
        """
        self.df = dataframe.copy()
        self.df['source_id'] = self.df['source_id'].astype(str)
        self.df['label'] = self.df['label'].astype(str)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        item = self.df.loc[idx]
        return {
            'source_id': item.source_id,
            'image_path': item.image_path,
            "text": item.text,
            'label': item.label
        }


def load_weibo_data(root_path):
    """
    Dataset :{
        source_id:str,
        image_path:str,
        text:str,
        label:str
    }
    lang: zh
    """
    def image_file_name_list2image_path(image_file_name_list):
        return f'{root_path}/{ast.literal_eval(image_file_name_list)[0]}'

    data_dir = root_path
    file_path = f'{data_dir}/weibo.csv'
    df = pd.read_csv(file_path,encoding='utf-8').rename(columns={'id': 'source_id'})
    df['text'] = df.apply(lambda x : f'{x["text"]} --发布来源：{x["release_source"]}', axis=1)
    df['image_path'] = df['available_image_paths'].apply(image_file_name_list2image_path)
    dataset = FakeNewsDataset(df[['source_id', 'image_path', 'text', 'label']])
    return dataset,'zh'

#class ImageTextPairDataset(Dataset):
#
#    def __init__(self,dataframe):
#        """
#        dataframe = {
#            'id':,
#            'image_url':,
#            "text":,
#            'label':,
#            "publish_date":,
#            'image_id':
#        }
#        """
#        self.df = dataframe
#        self.df['id'] = self.df['id'].apply(str)
#        self.df['image_id'] = self.df['image_id'].apply(str)
#
#    def __len__(self):
#        return self.df.shape[0]
#
#    def __getitem__(self, idx):
#        item = self.df.loc[idx]
#        return {
#            'id': item.id,
#            'image_url': item.image_url,
#            "text":item.text,
#            'label':item.label,
#            "publish_date":item.publish_date,
#            'image_id':item.image_id
#        }
#
#
#
#
#class FakeNewsRationaleFewShotDataset(Dataset):
#
#    def __init__(self,dataframe):
#        """
#        dataframe = {
#            'text':str,
#            'rationale':str,
#            'label': real or fake,
#        }
#        """
#        self.df = dataframe
#
#    def __len__(self):
#        return self.df.shape[0]
#
#    def __getitem__(self, idx):
#        return self.df.iloc[idx].to_dict()
#
#    def __getitems__(self,indices):
#        return self.df.iloc[indices].to_dict(orient='records')
#
#
#class InfiniteBatchSampler(Sampler):
#    def __init__(self, dataset, batch_size, drop_last=False):
#        super().__init__()
#        self.dataset = dataset
#        self.batch_size = batch_size
#        self.drop_last = drop_last  # 新增是否丢弃最后不完整批次
#
#    def __iter__(self):
#        while True:
#            # 生成随机索引
#            indices = torch.randperm(len(self.dataset)).tolist()
#
#            # 计算有效批次数
#            batch_count = len(indices) // self.batch_size
#            if not self.drop_last and (len(indices) % self.batch_size != 0):
#                batch_count += 1
#
#            # 生成完整批次
#            for i in range(batch_count):
#                start = i * self.batch_size
#                end = start + self.batch_size
#                batch = indices[start:end]
#                if len(batch) > 0:  # 确保最后不完整批次的有效性
#                    yield batch
#
#    def __len__(self):
#        if self.drop_last:
#            return len(self.dataset) // self.batch_size
#        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
#
#
#class InfiniteBalancedBatchSampler(Sampler):
#    def __init__(self, dataset, batch_size):
#        self.dataset = dataset
#        self.batch_size = batch_size
#        self.label_to_indices = defaultdict(list)
#
#        for idx, item in enumerate(self.dataset):
#            label = label_str2int_dict[item['label']]
#            self.label_to_indices[label].append(idx)
#
#        self.real_shot_nums = batch_size // 2
#        self.fake_shot_nums = batch_size - self.real_shot_nums
#
#
#        if len(self.label_to_indices[0]) < self.real_shot_nums:
#            raise ValueError('real 样本不足')
#        if len(self.label_to_indices[1]) < self.fake_shot_nums:
#            raise ValueError('fake 样本不足')
#
#
#
#    def __iter__(self):
#        while True:  # Infinite loop to provide infinite batches
#            real_perm = list(self.label_to_indices[0])
#            fake_perm = list(self.label_to_indices[1])
#
#            random.shuffle(real_perm)
#            random.shuffle(fake_perm)
#
#            # Create iterators for each class that will yield indices indefinitely
#            real_cycle = iter(itertools.cycle(real_perm))
#            fake_cycle = iter(itertools.cycle(fake_perm))
#
#            for _ in range(len(self.dataset) // self.batch_size):
#                batch = [next(real_cycle) for _ in range(self.real_shot_nums)] + [next(fake_cycle) for _ in range(self.fake_shot_nums)]
#                yield batch
#
#    def __len__(self):
#        # Since the sampler is infinite, we cannot provide a meaningful length.
#        # However, this method can return the number of batches in one epoch for guidance.
#        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
#
#
#
#class FewShotDataLoader(Iterator):
#    def __init__(self, dataset, batch_sampler):
#        """
#        :param dataset: 实现了 __getitem__ 和 __len__ 方法的数据集对象。
#        :param batch_sampler: 提供索引批次的采样器对象。
#        """
#        self.dataset = dataset
#        self.batch_sampler_iter = iter(batch_sampler)
#
#    def __next__(self) -> List[Dict[str, Any]]:
#        """
#        返回下一个批次的数据。
#        """
#        try:
#            batch_indices = next(self.batch_sampler_iter)
#            # 使用 __getitems__ 方法来获取批量数据
#            batch_data = self.dataset.__getitems__(batch_indices)
#            return batch_data
#        except StopIteration:
#            raise StopIteration("No more batches available.")
#
#
#def default_collect_fn(batch):
#    return list(batch)
#
#
#
#def load_text_few_shot_data(few_shot_dir,num_few_shot,language,rationale_name):
#    few_shot_file_path = f'{few_shot_dir}/{language}_{rationale_name}_shot.csv'
#    dataset = FakeNewsRationaleFewShotDataset(
#            pd.read_csv(few_shot_file_path)
#        )
#    return FewShotDataLoader(dataset,InfiniteBalancedBatchSampler(dataset, num_few_shot))
#
#def load_vl_few_shot_data(few_shot_dir,num_few_shot,language,rationale_name):
#    few_shot_file_path = f'{few_shot_dir}/{language}_{rationale_name}_shot.csv'
#    df = pd.read_csv(few_shot_file_path)
#    df['image_path'] = df['image_path'].apply(lambda x: f'{few_shot_dir}/{x}')
#    dataset = FakeNewsRationaleFewShotDataset(df)
#    return FewShotDataLoader(dataset, InfiniteBatchSampler(dataset, num_few_shot))
#
#
#
#def load_en_image_text_pair_goss(root_path,batch_size,collect_fn):
#    data_dir = root_path
#    file_path = f'{data_dir}/gossipcop.csv'
#    df = pd.read_csv(file_path)
#    df['image_id'] = df['id']
#    df['image_url'] = df['id'].map(lambda x : f'file://{root_path}/images/{x}_top_img.png')
#    dataset = ImageTextPairDataset(df)
#    return DataLoader(dataset, batch_size,False,num_workers=1,collate_fn=collect_fn)
#
#
#def get_twitter_image_url_dict(root_path):
#    image_dir = f'{root_path}/images'
#    return {
#       file.split('.')[0] : f'file://{image_dir}/{file}' for file in os.listdir(image_dir)
#    }
#
#def load_twitter_data(root_path,batch_size,collect_fn):
#    data_dir = root_path
#    file_path = f'{data_dir}/twitter.csv'
#    df = pd.read_csv(file_path)
#    image_id2url_dict = get_twitter_image_url_dict(root_path)
#
#    df = pd.DataFrame({
#        'id':df['post_id'],
#        'text':df['post_text'],
#        'label':df['label'],
#        'publish_date':df['timestamp'],
#        'image_id':df['image_id'],
#        'image_url': df['image_id'].map(lambda x : image_id2url_dict.get(x)),
#    })
#
#    dataset = ImageTextPairDataset(df)
#    return DataLoader(dataset, batch_size,False,num_workers=1,collate_fn=collect_fn)
#
#def load_weibo_en_data(root_path,batch_size,collect_fn):
#    file_path = os.path.join(root_path, 'weibo_en.csv')
#    df = pd.read_csv(file_path)
#    dataset = ImageTextPairDataset(df)
#    return DataLoader(dataset, batch_size,False,num_workers=1,collate_fn=collect_fn)



def load_few_shot_data(root_path,language,rationale_name,nums_few_shot):
    """
    return : {
        text,image_path,rationale,label
    }
    """
    file_path = os.path.join(root_path, f'{language}_{rationale_name}_shot.csv')
    return streaming_pandas_batch_iterator(file_path,nums_few_shot)


def streaming_pandas_batch_iterator(file_path, batch_size=100):
    """
    返回字典数组的流式批次迭代器

    参数：
    file_path: CSV文件路径
    batch_size: 每个批次的行数
    skip_header: 是否跳过表头

    返回：
    迭代器生成包含字典的列表，每个字典对应一行数据
    """
    base_dir = os.path.abspath(os.path.dirname(file_path))

    reader = pd.read_csv(file_path,
                         chunksize=batch_size,
                         header=0)

    processed_chunks = []
    for chunk in reader:
        # 校验列存在性
        if 'image_path' not in chunk.columns:
            raise KeyError("CSV文件中不存在 image_path 列")

        # 转换路径并转为字典列表
        chunk['image_path'] = chunk['image_path'].apply(
            lambda x: os.path.abspath(os.path.join(base_dir, x))
        )

        # 转换为字典数组（orient='records'）
        dict_batch = chunk.to_dict(orient='records')
        processed_chunks.append(dict_batch)

    return itertools.cycle(processed_chunks)


def default_collect_fn(batch):
    return list(batch)

def load_data(dataset_name,root_path,batch_size):
    data_load_func_dict = {
        'weibo':load_weibo_data
        # TODO  More Data loader
    }
    dataset,lang = data_load_func_dict[dataset_name](root_path)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=4,
                      pin_memory=True,
                      collate_fn=default_collect_fn),lang


# def load_data(dataset,root_path,batch_size=1,collect_fn=default_collect_fn):
#     if dataset == 'gossipcop':
#         return load_en_image_text_pair_goss(root_path,batch_size,collect_fn),'en'
#     elif dataset == 'twitter':
#         return load_twitter_data(root_path,batch_size,collect_fn),'en'
#     elif dataset == 'weibo':
#         return load_weibo_data(root_path,batch_size,collect_fn),'zh'
#     elif dataset == 'weibo_en':
#         return load_weibo_en_data(root_path,batch_size,collect_fn),'en'

