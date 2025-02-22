

import argparse
import asyncio
import os
import pickle
import re
from pprint import pprint

import pandas as pd
import yaml
from tqdm import tqdm

import DeepSeekPredict
import Util
import data_loader
import model
from DeepSeekPredict import filter_illegal_data, filter_input_batch



config_file_path = 'config/deepSeek671Predict.yaml'

parser = argparse.ArgumentParser()
parser.add_argument('--config_file_path', type=str, default=config_file_path)

config_file_path = parser.parse_args().config_file_path
config = yaml.load(open(config_file_path), Loader=yaml.FullLoader)

import json
def messages_builder_example(content):
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": content}]
    return messages



def write_batch_file_request(messages,custom_id,model_name,out_file_path):
    body = {"model": model_name, "messages": messages}
    request = {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body}
    with open(out_file_path, "a") as fout:
        fout.write(json.dumps(request, separators=(',', ':'), ensure_ascii=False) + "\n", )



def gen_batch_file(model_name,msg_Util,data_iter,cache_file,out_file_path):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            ans = pickle.load(f)
    else:
        ans = {}


    ans = filter_illegal_data(ans)
    for batch in tqdm(data_iter):
        batch_ids,batch_texts = filter_input_batch(batch,ans.keys())
        if len(batch_ids)==0:
            continue
        input_messages = msg_Util.wrapper_message(batch_texts)
        pprint(input_messages)
        assert len(input_messages)==len(batch_ids)
        for i in range(len(batch_ids)):
            write_batch_file_request(input_messages[i],batch_ids[i],model_name,out_file_path)




if __name__ == '__main__':
    model_name = config['ModelConfig']['bootConfig']['model_name']
    data_iter,lang = data_loader.load_data(config['dataset'],config['root_path'],batch_size=config['batch_size'])
    cache_file_path = f'cache/{config["dataset"]}/{config["ModelConfig"]["cache_file_path"]}'
    out_file_path = f'{config["dataset"]}_batch_input.jsonl'
    msg_Util = DeepSeekPredict.DeepSeekPredictUtil2(lang)
    # ans = predict(deepseek,msg_Util,data_iter,cache_file_path)
    gen_batch_file(model_name,msg_Util,data_iter,cache_file_path,out_file_path)





