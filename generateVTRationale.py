import asyncio
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
        self.rationale_type = rationale_type
        self.lang = lang
        if lang == 'en':
            self.system_prompt = Util.en_vl_system_prompt.format(rationale_type=Util.en_rationale_type_dict[rationale_type],max_tokens=max_tokens)
            if few_shot:
                self.system_prompt += "Here are some examples:\n"
                self.few_shot_output_prompt = Util.en_text_few_shot_output_prompt
            self.input_prompt = Util.en_input_prompt
        elif lang == 'zh':
            self.system_prompt = Util.zh_vl_system_prompt.format(rationale_type=Util.zh_rationale_type_dict[rationale_type],max_tokens=max_tokens)
            if few_shot:
                self.system_prompt += "以下是一些示例：\n"
                self.few_shot_output_prompt = Util.zh_text_few_shot_output_prompt
            self.input_prompt = Util.zh_input_prompt

        self.msgUtil = Util.VLMessageUtil(system_prompt=self.system_prompt)

    def valid_outputs(self,outputs):
        valid_func = Util.validate_model_zh_output if self.lang == 'zh' else Util.validate_model_en_output
        return [ valid_func(out) for out in outputs]

    def wrapper_message(self,batch,few_shot_data,image_url_type):
        """
        :param texts:
        :param image_url_list: file path
        :param few_shot_data: list[Dict],has keys [text,image_path,rationale,label]
        """
        few_shot_msgs = None
        use_text = 'text' in Util.rationale_type2mode[self.rationale_type]
        use_image = 'image' in Util.rationale_type2mode[self.rationale_type]
        if few_shot_data:
            for shot in few_shot_data:
                if use_text:
                    shot['text'] = self.input_prompt.format(news_text=shot['text'])
                shot['rationale'] = self.few_shot_output_prompt.format(label=shot['label'],rationale=shot['rationale'])
            few_shot_msgs = Util.generateFewShotMessage(few_shot_data,image_url_type=image_url_type)


        batch_message = []
        for item in batch:
            msg = {}
            msg['id'] = item['id']
            if use_text:
                msg['text'] = self.input_prompt.format(news_text=item['text'])
            if use_image:
                msg['image_path'] = Util.local_image_url2image_path(item['image_url'])

            batch_message.append(msg)

        return self.msgUtil.generate_vl_message(batch_message,few_shot_msgs,image_url_type=image_url_type)



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
    return [item for item in batch if item['id'] not in exist_ids_set]

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

    generate_config = config['QwenConfig']['generateConfig']
    del generate_config['max_tokens']

    for batch in tqdm(data):

        batch_after_filter = filter_input_batch(batch,ans.keys())
        if len(batch_after_filter)==0:
            continue
        few_shot = next(few_shot_iter) if few_shot_iter is not None else None
        msgs,batch_ids = msg_util.wrapper_message(batch_after_filter,few_shot,image_url_type=image_url_type)
        pprint(f'input message: {msgs}')
        outs = model.chat(msgs,batch_ids=batch_ids,**generate_config)
        pprint(outs)
        dict_outs = msg_util.valid_outputs(outs)
        pprint(dict_outs)
        ans.update(dict(zip(batch_ids, dict_outs)))
        # 定期保存缓存
        Util.save_cache(cache_file, ans)

    # 最后一次保存缓存
    Util.save_cache(cache_file, ans)

    return ans


async def async_generate_LLM_Text_Rationale(data, model, few_shot_iter, cache_file, msg_util:VLRationaleMessageUtil,image_url_type):
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

    generate_config = config['QwenConfig']['generateConfig']
    del generate_config['max_tokens']

    for batch in tqdm(data):

        batch_after_filter = filter_input_batch(batch,ans.keys())
        if len(batch_after_filter)==0:
            continue
        few_shot = next(few_shot_iter) if few_shot_iter is not None else None
        pprint(f'input message: {batch_after_filter}')
        msgs,batch_ids = msg_util.wrapper_message(batch_after_filter,few_shot,image_url_type=image_url_type)
        outs = await model.batch_inference(msgs,batch_ids=batch_ids,**generate_config)
        pprint(outs)
        dict_outs = msg_util.valid_outputs(outs)
        pprint(dict_outs)
        ans.update(dict(zip(batch_ids, dict_outs)))
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
    if 'LocalConfig' in config['QwenConfig']:
        Qwen = model.VLLMQwenVL(config['qwen_path'], **config['QwenConfig']["LocalConfig"])
    else:
        Qwen = model.AsyncRemoteLLM(**config['QwenConfig']["RemoteConfig"])

    data_iter, lang = data_loader.load_data(config['dataset'], config['root_path'], batch_size=config['batch_size'])
    few_shot_iter = None
    if config['few_shot']['enable']:
        few_shot_iter = data_loader.load_vl_few_shot_data(few_shot_dir=config['few_shot']['few_shot_dir'],
                                                            num_few_shot=config['few_shot']['num_few_shot'],
                                                            language=lang, rationale_name=config['rationale_name'])
    message_util = VLRationaleMessageUtil(rationale_type=config['rationale_name'],
                                          lang=lang,
                                          max_tokens=config['QwenConfig']['generateConfig']['max_tokens'],
                                          few_shot=config['few_shot']['enable'])

    cache_file_path = f'cache/{config["dataset"]}/{config["rationale_name"]}.pkl'
    image_url_type = Qwen.image_url_type
    if image_url_type == 'local':
        data = generate_LLM_Text_Rationale(data_iter, Qwen, few_shot_iter, cache_file_path, message_util,image_url_type)
    else:
        data = asyncio.run(async_generate_LLM_Text_Rationale(data_iter, Qwen, few_shot_iter, cache_file_path, message_util,image_url_type))
    save_file_path = f'{config["root_path"]}/{config["rationale_name"]}.csv'
    write_VL_LLM_Rationale(data, save_file_path)