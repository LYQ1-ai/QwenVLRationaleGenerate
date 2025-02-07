import os
import pickle
from pprint import pprint

import pandas as pd
import yaml


import argparse

from tqdm import tqdm

import Util
import data_loader
import model





CACHE_DIR = '/home/lyq/PycharmProjects/llamaRationaleGenerate/cache'





parser = argparse.ArgumentParser()
parser.add_argument('--config_file_path', type=str, default='config/generateVTRationale_config.yaml')
args = parser.parse_args()

config = yaml.load(open(args.config_file_path),Loader=yaml.FullLoader)


os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

class VLRationaleMessageUtil:



    def __init__(self,rationale_type,lang,max_tokens,few_shot=False):
        self.lang = lang
        if lang == 'en':
            self.system_prompt = Util.en_vl_system_prompt.format(rationale_type=Util.en_rationale_type_dict[rationale_type],max_tokens=max_tokens)
            if few_shot:
                pass # TODO 是否需要few_shot
            self.input_prompt = Util.en_input_prompt
        elif lang == 'zh':
            self.system_prompt = Util.zh_vl_system_prompt.format(rationale_type=Util.zh_rationale_type_dict[rationale_type],max_tokens=max_tokens)
            if few_shot:
                pass # TODO 是否需要few_shot
            self.input_prompt = Util.zh_input_prompt

        self.msgUtil = Util.VLMessageUtil(system_prompt=self.system_prompt)

    def valid_outputs(self,outputs):
        valid_func = Util.validate_model_zh_output if self.lang == 'zh' else Util.validate_model_en_output
        return [ valid_func(out) for out in outputs]

    def wrapper_message(self,texts,image_url_list,few_shot_data,image_url_type):
        input_prompts = [self.input_prompt.format(news_text=text) for text in texts]
        image_url_list = [Util.local_image_url2image_path(image_url) for image_url in image_url_list]
        return self.msgUtil.generate_vl_message(input_prompts, image_url_list,image_url_type=image_url_type)

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
    return generate id list,generate text list,generate image url list
    """
    return ([item['id'] for item in batch if item['id'] not in exist_ids_set],
            [item['text'] for item in batch if item['id'] not in exist_ids_set],
            [item['image_url'] for item in batch if item['id'] not in exist_ids_set]
            )

def generate_LLM_Text_Rationale(data, model, few_shot_iter, cache_file, msg_util:VLRationaleMessageUtil,image_url_type):
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

        batch_ids,batch_texts,batch_image_urls = filter_input_batch(batch,ans.keys())
        if len(batch_ids)==0:
            continue
        few_shot = next(few_shot_iter) if few_shot_iter is not None else None
        msg = msg_util.wrapper_message(batch_texts,batch_image_urls,few_shot,image_url_type=image_url_type)
        outs = model.chat(msg,**config["QwenConfig"]["generateConfig"])
        pprint(outs)
        dict_outs = msg_util.valid_outputs(outs)
        pprint(dict_outs)
        ans.update(dict(zip(batch_ids,dict_outs)))
        # 定期保存缓存
        Util.save_cache(cache_file, ans)

    # 最后一次保存缓存
    Util.save_cache(cache_file, ans)

    return ans

def write_VL_LLM_Rationale(data,save_file_path):
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
    Qwen = model.VLLMQwenVL(config['qwen_path'], **config['QwenConfig']["bootConfig"])
    data_iter, lang = data_loader.load_data(config['dataset'], config['root_path'], batch_size=config['batch_size'])
    few_shot_iter = None
    if config['few_shot']['enable']:
        few_shot_iter = data_loader.load_text_few_shot_data(few_shot_dir=config['few_shot']['few_shot_dir'],
                                                            num_few_shot=config['few_shot']['num_few_shot'],
                                                            language=lang, rationale_name=config['rationale_name'])
    message_util = VLRationaleMessageUtil(rationale_type=config['rationale_name'],
                                          lang=lang,
                                          max_tokens=config['QwenConfig']['generateConfig']['max_tokens'],
                                          few_shot=config['few_shot']['enable'])

    cache_file_path = f'cache/{config["dataset"]}/{config["rationale_name"]}.pkl'
    image_url_type = Qwen.image_url_type
    data = generate_LLM_Text_Rationale(data_iter, Qwen, few_shot_iter, cache_file_path, message_util,image_url_type)
    save_file_path = f'{config["root_path"]}/{config["rationale_name"]}.csv'
    write_VL_LLM_Rationale(data, save_file_path)