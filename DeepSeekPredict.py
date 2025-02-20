import argparse
import asyncio
import os
import pickle
import re
from pprint import pprint

import pandas as pd
import yaml
from tqdm import tqdm

import Util
import data_loader
import model

predict_system_zh_prompt_template = """你是一名新闻真实性分析员。下面给出的文本是一篇新闻报道，请逐步分析以下新闻的真实性,你只能回答 '真' 或 '假',以下是一个示例：
"""
predict_system_en_prompt_template = """You are a news veracity analyser. The text given below is a news report.Analyse the truthfulness of the following news item step by step.You can only answer 'Real' or 'Fake'.Here is an example:
"""



config_file_path = 'config/deepSeekPredict.yaml'

parser = argparse.ArgumentParser()
parser.add_argument('--config_file_path', type=str, default=config_file_path)

config_file_path = parser.parse_args().config_file_path
config = yaml.load(open(config_file_path), Loader=yaml.FullLoader)


def filter_illegal_data(data):
    """
    data :[
        {id,pred}
    ]
    """
    def is_valid(value):
        return value==0 or value==1
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


class DeepSeekPredictMessageUtil:
    def __init__(self,lang):
        self.lang = lang
        system_prompt = predict_system_zh_prompt_template if lang == 'zh' else predict_system_en_prompt_template
        self.msg_Util = Util.TextMessageUtil(system_prompt=system_prompt, input_prompt=Util.input_prompt)

    def wrapper_message(self, texts):
        messages = self.msg_Util.generate_text_message(texts)

        res = []
        if self.lang == 'en':
            few_shot = [
                {
                    'role': 'user',
                    'content': 'The people of Boston ask your help identifying these men associated with the backpack bombs at the Boston Marathon http://t.co/u5fX9Gksbf'
                }
                ,
                {
                    'role': 'assistant',
                    'content': 'fake'
                }
            ]
        elif self.lang == 'zh':
            few_shot = [
                {
                    'role': 'user',
                    'content': '每次都是17人死亡，你忽悠谁呢？[发怒]#天津开发区发生爆炸##天津塘沽大爆炸#'
                }
                ,
                {
                    'role': 'assistant',
                    'content': '假'
                }
            ]
        else:
            raise NotImplementedError

        system_msg = messages[0]
        for i in range(1, len(messages)):
            request_messages = [system_msg]
            request_messages.extend(few_shot)
            request_messages.append(messages[i])
            res.append(request_messages)

        return res

    @staticmethod
    def remove_think_tags(text):
        """
        删除字符串中 <think> 和 </think> 标签及其包裹的内容。

        参数:
            text (str): 输入的字符串。

        返回:
            str: 删除了 <think> 标签及其内容的字符串。
        """
        # 使用正则表达式匹配 <think> 和 </think> 标签及其内容
        pattern = r'<think>.*?</think>'
        result = re.sub(pattern, '', text, flags=re.DOTALL)
        return result

    def process_output(self,outs):

        def parse_pred(text):
            """
           从text中找出以下字典中的key，label_str2int_dict = {
               "real": 0,
               "fake": 1,
               "other": 2,
               '真':0,
               '假':1,
               '其他':2
           }。text中只能包含label_str2int_dict中的一个键，如果包含多个或者不包含则返回None，否则返回包含的唯一的Key
           """
            matches = []
            for key in data_loader.label_str2int_dict:
                if key in text:
                    matches.append(key)
                    if len(matches) > 1:  # 发现多个匹配时立即终止遍历
                        break
            return data_loader.label_str2int_dict[matches[0]] if len(matches) == 1 else None



        def filter_output(text,lang):
            if text is None:
                return None
            text = DeepSeekPredictMessageUtil.remove_think_tags(text)
            if lang=='en':
                text=text.lower()
            text = text.strip()
            text = text.split('\n')[-1]
            return parse_pred(text)

        predicts =  [ filter_output(out,self.lang) for out in outs]
        return predicts





async def predict(model,msg_Util,data_iter,cache_file):
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
        outs = await model.batch_inference(input_messages,**config['ModelConfig']['generateConfig'])
        pprint(outs)
        predicts = msg_Util.process_output(outs)
        pprint(predicts)
        ans.update(dict(zip(batch_ids, predicts)))
        Util.save_cache(cache_file, ans)

    Util.save_cache(cache_file, ans)

    return ans


def write_LLM_predict(data,save_file_path):
    """
    :param data: dict{
        'id':{
            authenticity:str
            reason:str
        }
    }
    """
    df = pd.DataFrame(list(data.items()), columns=['source_id', 'pred'])
    df.to_csv(save_file_path, index=False)



if __name__ == '__main__':
    deepseek = model.AsyncRemoteDeepSeek(**config['ModelConfig']['bootConfig'])
    data_iter,lang = data_loader.load_data(config['dataset'],config['root_path'],batch_size=config['batch_size'])
    cache_file_path = f'cache/{config["dataset"]}/{config["ModelConfig"]["cache_file_path"]}'
    save_file_path = f'{config["root_path"]}/{config["ModelConfig"]["save_file_path"]}'
    msg_Util = DeepSeekPredictMessageUtil(lang)
    # ans = predict(deepseek,msg_Util,data_iter,cache_file_path)
    ans = asyncio.run(predict(deepseek,msg_Util,data_iter,cache_file_path))
    write_LLM_predict(ans,save_file_path)





