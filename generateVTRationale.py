import json
import os
import pickle

import pandas as pd
import requests
import torch
import yaml


import argparse

from av.container import output
from tqdm import tqdm

import Util
import model
from generateTextRationale import validate_model_output, Qwen2VL, dataloader_func_dict




CACHE_DIR = '/home/lyq/PycharmProjects/llamaRationaleGenerate/cache'





parser = argparse.ArgumentParser()
parser.add_argument('--config_file_path', type=str, default='config/generateVTRationale_config.yaml')
args = parser.parse_args()

config = yaml.load(open(args.config_file_path),Loader=yaml.FullLoader)

prompt_ITC_No_few_shot_template = """
The text enclosed in the <text></text> tags is a news summary, and the given image is the cover of that news article.
Please analyze the authenticity of this news article step by step from the perspective of whether there are contradictions between the image and the textual description.
Output in the following format:
- authenticity: a single word: fake or real
- reason: The basis for judging the authenticity of the news from the perspective of whether there are contradictions between the image and the textual description.
news text: <text>{news_text}</text>
"""


prompt_emo = """
You are a news veracity analyst. The text in the following text using <text></text> tags is a summary of a news article, and the picture given is the cover of the news.
Combining the picture and the text, please analyze the authenticity of the news article from the perspective of sentiment analysis (e.g. whether there is over-emotion, whether there is a deliberate attempt to lead or stimulate the reader's emotions, whether the emotions expressed in the picture are consistent with the textual description, etc.) step-by-step and give the basis for your judgment in English.
Output in the following format:
- Authenticity: one word: fake or real
- Reason: The basis for judging the authenticity of news from the perspective of sentiment analysis.
News text: <text>{news_text}</text
"""

prompt_multi = """
You are a news veracity analyzer. The text tagged with <text></text> below is a summary of a news story, and the given image is the cover of the news.
Combining the picture and the text, analyze the authenticity of the news article step-by-step from different perspectives and give the basis for your judgment in English.

You may choose to analyze it from the following angles:
(1) General social knowledge: whether the news source is reliable and logical
(2) Textual description: whether the description is deliberately derogatory or biased.
(3) Emotional tendency: whether the news content is overly emotional or deliberately guides the reader's emotions and provokes conflicts
(4) Graphic consistency: whether the picture and the text are consistent

Output in the following format:
- Authenticity: one word: Fake or Real
- Reason: The basis for judging the authenticity of the news.
News text: <text>{news_text}</text
"""


rationale_prompt_dict = {
    'itc': prompt_ITC_No_few_shot_template,
    'emo': prompt_emo,
    'multi':prompt_multi
}

model_type_mapping = {
    'remote':Util.RemoteQwenVL,
    'local':Util.QwenVL
}


class MessageUtil:



    def __init__(self,rationale_name):
        self.prompt = rationale_prompt_dict[rationale_name]


    def parser_output(self,output):
        return validate_model_output(output)

    def parser_batch_output(self,outputs):
        return [
            self.parser_output(item) for item in outputs
        ]

def validate_model_output(output):
    try:
       text = output
       res = {}
       auth,reason = text.split('\n',maxsplit=1)
       if 'fake' in auth.lower():
           res['authenticity'] = 'fake'
       elif 'real' in auth.lower():
           res['authenticity'] = 'real'
       elif 'other' in auth.lower():
           res['authenticity'] = 'other'
       if 'reason:' in reason:
           res['reason'] = reason.split('reason:',maxsplit=1)[1]
       elif 'Reason:' in reason:
           res['reason'] = reason.split('Reason:',maxsplit=1)[1]
       else:
           res['reason'] = None
       return res
    except Exception as e:
        return {}


def preprocess_input(batch,exits_id):
    result = {
        'id':[],
        'text':[],
        'image_path':[]
    }
    for i in range(len(batch['id'])):
        if batch['id'][i] in exits_id:
            continue
        result['id'].append(batch['id'][i])
        result['text'].append(batch['text'][i])
        result['image_path'].append(batch['image_url'][i][len('file://'):])

    return result



def filter_illegal_data(cache):
    if cache:
        df = pd.DataFrame.from_dict(cache, orient='index')
        df.dropna(subset=['authenticity','reason'], inplace=True)
        return df.to_dict(orient='index')
    else:
        return cache





def generate_batch_vl_Rationale(data, model,rationale_name,dataset):

    msg_util = MessageUtil(rationale_name)

    cache_file = f'{CACHE_DIR}/{dataset}/{rationale_name}.pkl'
    # 检查缓存是否存在并加载
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            ans = pickle.load(f)
    else:
        ans = {}

    ans = filter_illegal_data(ans)

    for batch in tqdm(data):
        if isinstance(batch['id'], torch.Tensor):
            batch['id'] = batch['id'].tolist()

        inputs = preprocess_input(batch, ans.keys())
        input_ids = inputs['id']
        if len(input_ids) == 0:
            continue
        try:
            texts = [msg_util.prompt.format(news_text=t) for t in inputs['text']]
            outputs = model.batch_inference_v2(texts,inputs['image_path'])
            print(outputs)
            outputs = msg_util.parser_batch_output(outputs)
            ans.update(zip(input_ids, outputs))

            # 定期保存缓存
            if len(ans) % 100 == 0:
                with open(cache_file, 'wb') as f:
                    pickle.dump(ans, f)
        except Exception as e:
            print(e)


    # 最后一次保存缓存
    with open(cache_file, 'wb') as f:
        pickle.dump(ans, f)

    return ans


def write_vl_rationales(result,save_file):
    """
    :param result: {
        "id1": {
            'authenticity' : str,
            'reason' : str,
        },
        "id2": {
            'authenticity' : str,
            'reason' : str,
        }
    }
    :return:
    """
    df = pd.DataFrame.from_dict(result, orient='index').reset_index()
    df.rename(columns={'index':'source_id'}, inplace=True)
    df.to_csv(save_file, index=False)






if __name__ == '__main__':
    model = model_type_mapping[config['model_type']](model_dir = config['qwen_path'])
    dataset = config['dataset']
    print(f'generate : {dataset}')
    rationale_name = config['rationale_name']
    data = dataloader_func_dict[dataset](root_path=config['root_path'],batch_size=1)
    result = generate_batch_vl_Rationale(data,model,rationale_name,dataset)
    save_file = f'{config["root_path"]}/{rationale_name}.csv'
    write_vl_rationales(result,save_file)