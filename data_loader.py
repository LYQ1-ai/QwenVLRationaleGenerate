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

def load_gossipcop_data(root_path):
    """
        Dataset :{
            source_id:str,
            image_path:str,
            text:str,
            label:str
        }
        lang: en
    """
    df = pd.read_csv(os.path.join(root_path, 'gossipcop.csv')).rename(columns={'id': 'source_id'})
    df['image_path'] = df['source_id'].apply(lambda x : os.path.join(root_path, 'images', f'{x}_top_img.png'))
    return FakeNewsDataset(df[['source_id','image_path','text','label']]),'en'


def load_twitter_data(root_path):
    def get_image_id2image_path_dict():
        return {file_name.split('.')[0]:os.path.join(root_path, 'images', file_name)  for file_name in os.listdir(os.path.join(root_path, 'images')) }

    df = pd.read_csv(os.path.join(root_path, 'twitter.csv')).rename(columns={'post_id': 'source_id','post_text':'text'})
    image_id2image_path_dict = get_image_id2image_path_dict()
    df['image_path'] = df['image_id'].apply(image_id2image_path_dict.get)
    return FakeNewsDataset(df[['source_id','image_path','text','label']]),'en'
    

def load_weibo_en_data(root_path):
    file_path = os.path.join(root_path, 'weibo_en.csv')
    df = pd.read_csv(file_path).rename(columns={'id': 'source_id'})
    dataset = FakeNewsDataset(df)
    return dataset,'en'



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
        'weibo':load_weibo_data,
        'gossipcop':load_gossipcop_data,
        'twitter':load_twitter_data,
        'weibo_en':load_weibo_en_data,
    }
    dataset,lang = data_load_func_dict[dataset_name](root_path)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=4,
                      pin_memory=True,
                      collate_fn=default_collect_fn),lang
