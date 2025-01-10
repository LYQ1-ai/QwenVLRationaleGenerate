import json
import re
from pprint import pprint

import pandas as pd
import torch
import yaml
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


import os
from tqdm import tqdm

import Util
import data_loader
import pickle

import argparse

import model
from Util import TextMessageUtil
from data_loader import load_en_image_text_pair_goss, load_twitter_data
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'


def filter_illegal_data(data):
    def is_valid(value):
        return all([
            value is not None,
            value.get('auth') is not None,
            value.get('reason') is not None
        ])
    return {k:v for k, v in data.items() if is_valid(v)}



def filter_input_batch(batch,exist_ids_set):
    """
    batch : [
        {
            'id':,
            'image_url':,
            "text":,
            'label':,
            "publish_date":,
            'image_id':
        }
    ]
    """
    return [item['id'] for item in batch if item['id'] not in exist_ids_set],[item['text'] for item in batch if item['id'] not in exist_ids_set]



def generate_LLM_Text_Rationale(data, model,few_shot_iter,cache_file,msg_util:TextMessageUtil):
    """
    return dict{
        'id':{
            authenticity:str
            reason:str
        }
    }
    """
    # 检查缓存是否存在并加载
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            ans = pickle.load(f)
    else:
        ans = {}

    ans = filter_illegal_data(ans)

    for batch in tqdm(data):

        batch_ids,batch_texts = filter_input_batch(batch,ans.keys())
        if len(batch_ids)==0:
            continue
        few_shot = next(few_shot_iter) if few_shot_iter is not None else None
        msg = msg_util.generate_text_messages(batch_texts,few_shot)
        outs = model.chat(msg)
        print(outs)
        dict_outs = [ msg_util.valid_output(out) for out in outs]
        pprint(dict_outs)
        ans.update(dict(zip(batch_ids,dict_outs)))
        # 定期保存缓存
        if len(ans.keys()) % 100 == 0:
            Util.save_cache(cache_file, ans)

    # 最后一次保存缓存
    Util.save_cache(cache_file, ans)

    return ans



def write_LLM_Rationale(data,save_file_path):
    """
    :param data: dict{
        'id':{
            authenticity:str
            reason:str
        }
    }
    """
    df = pd.DataFrame(data).from_dict(data,orient='index').reset_index()
    df = df.rename(columns={'index':'source_id'})
    df.to_csv(save_file_path, index=False)






if __name__ == '__main__':
    config_file_path = '/home/lyq/PycharmProjects/QwenVLRationaleGenerate/config/generatTextRationale_config.yaml'
    config = yaml.load(open(config_file_path),Loader=yaml.FullLoader)
    Qwen = model.VLLMQwen(config['qwen_path'],**config['QwenConfig'])
    data_iter,lang = data_loader.load_data(config['dataset'],config['root_path'],batch_size=config['batch_size'])
    few_shot_iter = None
    if config['few_shot']['enable']:
        few_shot_iter = data_loader.load_text_few_shot_data(few_shot_dir=config['few_shot']['few_shot_dir'],num_few_shot=config['few_shot']['num_few_shot'],language=lang,rationale_name=config['rationale_name'])

    message_util = TextMessageUtil(
        lang=lang,
        rationale_type=config['rationale_name'],
        few_shot=config['few_shot']['enable']
    )

    cache_file_path = f'cache/{config["dataset"]}/{config["rationale_name"]}.pkl'
    data = generate_LLM_Text_Rationale(data_iter,Qwen,few_shot_iter,cache_file_path,message_util)
    save_file_path = f'{config["root_path"]}/{config["rationale_name"]}.csv'
    write_LLM_Rationale(data,save_file_path)





















