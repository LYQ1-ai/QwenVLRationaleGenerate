import os
import pickle
import re
from pprint import pprint

import pandas as pd
import yaml
from tqdm import tqdm

import Util
import data_loader
from Util import TextMessageUtil
from model import VLLMQwenVL, VLLMQwen

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

config = yaml.load(open('/home/lyq/PycharmProjects/QwenVLRationaleGenerate/config/summaryNewsText_config.yaml', 'r'),Loader=yaml.FullLoader)

def extract_summary(text):
    pattern = r'<text>(.*?)</text>'
    matches = re.findall(pattern, text, re.DOTALL)
    return str.join(' ', matches)

class SummaryMessageUtil:
    def __init__(self,lang,max_len):
        if lang == 'en':
            system_prompt = Util.en_summary_system_prompt
        elif lang == 'zh':
            system_prompt = Util.zh_summary_system_prompt
        else:
            raise Exception

        system_prompt = system_prompt.format(max_len=max_len)
        input_prompt = Util.input_prompt
        self.msgUtil = TextMessageUtil(system_prompt, input_prompt)

    def summaryNewsText(self,texts):
        return self.msgUtil.generate_text_message(texts)


def filter_illegal_data(data):
    if len(data['id'])==0:
        return data
    df = pd.DataFrame.from_dict(data)
    df = df.dropna(subset=['summary'])
    return df.to_dict('list')


def filter_exist_id(batch, exist_id):
    # 将 exist_id 转换为集合以加速查找操作
    exist_id_set = set(exist_id)

    # 提取 batch 中的 id 和 summary 并确保它们长度相同
    batch_ids = batch.get('id', [])
    texts = batch.get('text', [])
    assert len(batch_ids) == len(texts), "The 'id' and 'summary' lists must have the same length."

    # 过滤出不在 exist_id_set 中的 id 及其对应的 summary
    filtered_data = [
        (id_, t)
        for id_, t in zip(batch_ids, texts)
        if id_ not in exist_id_set
    ]

    result_ids, result_texts = zip(*filtered_data) if filtered_data else ([], [])

    return {
        'id': list(result_ids),
        'text': list(result_texts)
    }


def summarizeNewsText(data_iter, model,lang,cache_file):
    msgUtil = SummaryMessageUtil(lang,config['QwenConfig']['generateConfig']['max_tokens'])
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            result = pickle.load(f)
    else:
        result = {
            'id': [],
            'summary': [],
        }

    result = filter_illegal_data(result)
    exist_id = set(result['id'])

    for batch in tqdm(data_iter):
        batch = filter_exist_id(batch,exist_id)
        texts = batch['text']
        messages = msgUtil.summaryNewsText(texts)
        outputs = model.chat(messages,**config['QwenConfig']['generateConfig'])
        pprint(outputs)
        result['id'].extend(batch['id'])
        result['summary'].extend(outputs)
        Util.save_cache(cache_file,result)


    return result


if __name__ == "__main__":
    model = VLLMQwen(config['qwen_path'], **config['QwenConfig']['bootConfig'])
    data_iter,lang = data_loader.load_data(config['dataset'],config['root_path'],config['batch_size'],collect_fn=None)
    cache_file = f'cache/{config["dataset"]}/summary.pkl'
    result = summarizeNewsText(data_iter, model,lang,cache_file)
    pd.DataFrame(result).to_csv(f'{config["root_path"]}/{config["dataset"]}_summary.csv',index=False)


