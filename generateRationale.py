import argparse
import asyncio
import os
import pprint

import pandas as pd
import yaml
from tqdm import tqdm

import data_loader
import genRationaleUtil
import model

CACHE_DIR = '/home/lyq/PycharmProjects/QwenVLRationaleGenerate/cache'





parser = argparse.ArgumentParser()
parser.add_argument('--config_file_path', type=str, default='config/generateRationale_config.yaml')
args = parser.parse_args()
config = yaml.load(open(args.config_file_path),Loader=yaml.FullLoader)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"




async def async_generate_rationale(data_iter, qwen, msg_Util,few_shot_data_iter=None):
    """
    data_iter : [{
        source_id:str,
        image_path:str,
        text:str,
        label:str
    }]
    few_shot_data_iter : [{
        text,image_path,rationale,label
    }]
    """
    cache_file_path = os.path.join(CACHE_DIR, config['dataset'],f"{config['rationale_type']}_cache.pkl")
    cache_manager = genRationaleUtil.CacheManager(cache_file_path)

    res = cache_manager.load_cache()
    printer = genRationaleUtil.CustomPrinter()

    for batch in tqdm(data_iter):
        batch = genRationaleUtil.filter_batch_input(batch,set(res.keys()))
        batch_ids = [item['source_id'] for item in batch]
        if len(batch_ids) ==0:
            continue
        few_shot_data = next(few_shot_data_iter) if few_shot_data_iter else None
        messages = [msg_Util.wrapper_message(item,few_shot_data) for item in batch]
        print(f'=================================================input message=================================================\n')
        printer.pprint(messages)
        
        outs = await qwen(messages,**config['ModelConfig']['generateConfig'])
        print(f'=================================================raw out=================================================\n')
        printer.pprint(outs)
        dict_outs = [msg_Util.valid_output(out) for out in outs]
        print(
            f'=================================================dict out=================================================\n')
        printer.pprint(dict_outs)
        batch_outs = genRationaleUtil.filter_batch_out(dict(zip(batch_ids, dict_outs)),set(res.keys()))
        res.update(batch_outs)
        cache_manager.save_cache(res)

    return res


def generate_rationale(data_iter, qwen, msg_Util, few_shot_data_iter=None):
    """
    data_iter : [{
        source_id:str,
        image_path:str,
        text:str,
        label:str
    }]
    few_shot_data_iter : [{
        text,image_path,rationale,label
    }]
    """
    cache_file_path = os.path.join(CACHE_DIR, config['dataset'], f"{config['rationale_type']}_cache.pkl")
    cache_manager = genRationaleUtil.CacheManager(cache_file_path)

    res = cache_manager.load_cache()

    for batch in tqdm(data_iter):
        batch = genRationaleUtil.filter_batch_input(batch, set(res.keys()))
        batch_ids = [item['source_id'] for item in batch]
        if len(batch_ids) == 0:
            continue
        few_shot_data = next(few_shot_data_iter) if few_shot_data_iter else None
        messages = [msg_Util.wrapper_message(item, few_shot_data) for item in batch]
        print(
            f'=================================================input message=================================================\n')
        pprint.pprint(messages)

        outs = qwen(messages, **config['ModelConfig']['generateConfig'])
        print(
            f'=================================================raw out=================================================\n')
        pprint.pprint(outs)
        dict_outs = [msg_Util.valid_output(out) for out in outs]
        print(
            f'=================================================dict out=================================================\n')
        pprint.pprint(dict_outs)
        batch_outs = genRationaleUtil.filter_batch_out(dict(zip(batch_ids, dict_outs)), set(res.keys()))
        res.update(batch_outs)
        cache_manager.save_cache(res)

    return res


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
    model_type = 'local' if 'LocalConfig' in config['ModelConfig'] else 'remote'
    if model_type=='local':
        qwen = model.VLLMQwenVL(**config['ModelConfig']["LocalConfig"])
    else:
        qwen = model.AsyncRemoteLLM(**config['ModelConfig']["RemoteConfig"])
    data_iter, lang = data_loader.load_data(config['dataset'], config['root_path'], batch_size=config['batch_size'])
    use_few_shot = config['few_shot']['enable']
    rationale_type = config['rationale_type']
    few_shot_data_iter = None
    if use_few_shot:
        few_shot_data_iter = data_loader.load_few_shot_data(config['few_shot']['root_path'], lang,rationale_type, config['few_shot']['nums_few_shot'])
    msg_Util = genRationaleUtil.get_messageUtil(rationale_type=rationale_type, lang=lang,
                                                use_few_shot=use_few_shot,
                                                max_tokens=config['rationale_max_length'],
                                                image_url_type=qwen.image_url_type)


    if model_type=='remote':
        res = asyncio.run(async_generate_rationale(data_iter, qwen, msg_Util,few_shot_data_iter=few_shot_data_iter))
    else:
        res = generate_rationale(data_iter, qwen, msg_Util,few_shot_data_iter=few_shot_data_iter)
    save_file_path = os.path.join(config['root_path'],f'{config["rationale_type"]}.csv')
    write_VL_LLM_Rationale(res,save_file_path)


