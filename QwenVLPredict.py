import argparse
import asyncio
import pickle
import re
from pprint import pprint

import pandas as pd
import yaml
from tqdm import tqdm

import Util
import data_loader
import genRationaleUtil
import model
from genRationaleUtil import wrapper_remote_msg_image_content, wrapper_local_msg_image_content

predict_zh_input_template = """<待校验新闻>
{news_content}"""
predict_en_input_template = """<News to Verify>
{news_content}"""


predict_system_vl_zh_prompt_template = """你是一名新闻真实性分析员,下面给出的文本是一篇新闻报道,给出的图片为该新闻的封面,请逐步分析以下新闻的真实性,按照以下规则回答[仅返回XML标签]：

<规则>
1. 输出必须为独立XML行
2. 禁止任何附加文本
3. 严格二选一：<label>真</label> 或 <label>假</label>
</规则>

<违规示例>
❌ 经核查该新闻为<label>假</label>
❌ 真实性结论：<label>真</label>

<合规示例>
<label>真</label>
"""


predict_system_vl_en_prompt_template = """You are a news truthfulness analyst, the text given below is a news report, the picture given is the cover of the news, please analyze the truthfulness of the following news step by step, answer according to the following rules [return XML tags only]:

<Rules> 
1. Output must be a standalone XML line 
2. No additional text allowed 
3. Strict binary choice: `<label>real</label>` or `<label>fake</label>` 
</Rules> 

<Invalid Examples> 
❌ After verification: `<label>fake</label>` 
❌ Conclusion: `<label>real</label>` 

<Valid Example> `<label>real</label>` 

"""


config_file_path = 'config/qwenVLPredict.yaml'

parser = argparse.ArgumentParser()
parser.add_argument('--config_file_path', type=str, default=config_file_path)

config_file_path = parser.parse_args().config_file_path
config = yaml.load(open(config_file_path), Loader=yaml.FullLoader)


def filter_batch_predict_out(batch_ids,predicts):

    def is_valid(predict):
        if predict is None:
            return False
        return int(predict) in [0,1,2]
    ans = dict(zip(batch_ids, predicts))
    return { k:ans[k] for k in ans.keys() if is_valid(ans[k])}


class QwenVLPredictMessageUtil:

    def __init__(self, lang,image_url_type='remote'):
        self.lang = lang
        self.system_prompt = predict_system_vl_zh_prompt_template if lang == 'zh' else predict_system_vl_en_prompt_template
        self.input_template = predict_zh_input_template if lang == 'zh' else predict_en_input_template
        self.image_content_wrapper_func = wrapper_remote_msg_image_content if image_url_type=='remote' else wrapper_local_msg_image_content

    def wrapper_message(self, texts,image_path_list):
        assert len(texts) == len(image_path_list)
        return [self.wrapper_message0(t,i) for t,i in zip(texts, image_path_list)]

    def wrapper_message0(self, text, image_path):
        if self.lang == 'en':
            few_shot = [
                {
                    'role': 'user',
                    'content': 'The people of Boston ask your help identifying these men associated with the backpack bombs at the Boston Marathon http://t.co/u5fX9Gksbf'
                }
                ,
                {
                    'role': 'assistant',
                    'content': '<label>fake</label>'
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
                    'content': '<label>假</label>'
                }
            ]
        else:
            raise NotImplementedError


        text_input_content = {
            'type':'text',
            'text':self.input_template.format(news_content=text)
        }
        image_input_content = self.image_content_wrapper_func(image_path)
        res = [
            {'role': 'system', 'content': self.system_prompt},
        ]
        res.extend(few_shot)
        res.append({
            'role':'user',
            'content':[text_input_content,image_input_content]
        })
        return res



    @staticmethod
    def extract_label_content(text):
        """
        提取文本中第一个<label>标签的内容

        参数:
        text (str): 包含XML标签的输入文本

        返回:
        str/None: 第一个匹配的内容字符串，无匹配时返回None

        示例:
        extract_first_label('前缀<label>真</label> 后缀')
        '真'
        extract_first_label('无标签文本') is None
        True
        """
        match = re.search(r'<label[^>]*>(.*?)</label>', text, flags=re.DOTALL)
        return match.group(1).strip() if match else None

    def process_output(self, outs):
        return [self.process_output0(o) for o in outs]

    def process_output0(self, out):
        if out is None:
            return None
        if self.lang == 'en':
            out = out.lower().strip()
        else:
            out = out.strip()
        pred = self.extract_label_content(out)
        return data_loader.label_str2int_dict.get(pred, None)


async def async_predict(model,msg_Util,data_iter,cache_file):
    cache_manager = genRationaleUtil.CacheManager(cache_file)

    ans = cache_manager.load_cache()
    printer = genRationaleUtil.CustomPrinter()

    for batch in tqdm(data_iter):
        batch = genRationaleUtil.filter_batch_input(batch, set(ans.keys()))
        if len(batch)==0:
            continue

        batch_ids,batch_texts ,batch_image_paths = ([item['source_id'] for item in batch],
                                                    [item['text'] for item in batch],
                                                    [item['image_path'] for item in batch])
        input_messages = msg_Util.wrapper_message(batch_texts,batch_image_paths)
        printer.pprint(input_messages)
        outs = await model.batch_inference(input_messages,**config['ModelConfig']['generateConfig'])
        printer.pprint(outs)
        predicts = msg_Util.process_output(outs)
        printer.pprint(predicts)
        ans.update(filter_batch_predict_out(batch_ids,predicts))
        cache_manager.save_cache(ans)

    cache_manager.save_cache(ans)

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
    image_url_type = 'remote' if 'RemoteConfig' in config['ModelConfig'] else 'local'
    if image_url_type == 'remote':
        pred_model = model.AsyncRemoteLLM(**config['ModelConfig']['RemoteConfig'])
    else:
        pred_model = model.VLLMQwenVL(**config['ModelConfig']['LocalConfig'])

    data_iter,lang = data_loader.load_data(config['dataset'],config['root_path'],batch_size=config['batch_size'])
    cache_file_path = f'cache/{config["dataset"]}/{config["ModelConfig"]["cache_file_path"]}'
    save_file_path = f'{config["root_path"]}/{config["ModelConfig"]["save_file_path"]}'
    #msg_Util = DeepSeekPredictMessageUtil(lang)
    msg_Util = QwenVLPredictMessageUtil(lang,image_url_type)
    # ans = predict(deepseek,msg_Util,data_iter,cache_file_path)
    ans = asyncio.run(async_predict(pred_model,msg_Util,data_iter,cache_file_path))
    write_LLM_predict(ans,save_file_path)



