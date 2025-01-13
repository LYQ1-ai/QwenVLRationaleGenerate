import os
from pprint import pprint

import pandas as pd
import yaml
from tqdm import tqdm
import pickle

import Util
import data_loader
from model import VLLMQwen, VLLMQwenVL

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

class CaptionMessageUtil:

    msgUtil = None

    def __init__(self, lang):
        if lang == 'zh':
            self.system_prompt = Util.zh_caption_prompt
            self.input_prompt = Util.zh_input_prompt
        elif lang == 'en':
            self.system_prompt = Util.en_caption_prompt
            self.input_prompt = Util.en_input_prompt

        self.msgUtil = Util.VLMessageUtil(self.system_prompt, self.input_prompt)

    def generateCaptionMsg(self, texts, images_url):
        return self.msgUtil.generate_vl_message(texts, images_url)



def generate_image_caption(data_iter,model,lang):

    msgUtil = CaptionMessageUtil(lang)
    result = {
        'id':[],
        'caption':[],
    }
    for batch in tqdm(data_iter):
        texts = batch['text']
        images_url = [Util.local_image_url2image_path(image_url) for image_url in batch['image_url']]
        messages = msgUtil.generateCaptionMsg(texts, images_url)
        outputs = model.chat(messages)
        pprint(outputs)
        result['id'].extend(batch['id'])
        result['caption'].extend(outputs)


    return result


if __name__ == "__main__":
    config = yaml.load(open('/home/lyq/PycharmProjects/QwenVLRationaleGenerate/config/generatCaption_config.yaml', 'r'),Loader=yaml.FullLoader)
    model = VLLMQwenVL(config['qwen_path'],**config['QwenConfig'])

    data_iter,lang = data_loader.load_data(config['dataset'],config['root_path'],config['batch_size'],collect_fn=None)
    result = generate_image_caption(data_iter, model,lang)
    pd.DataFrame(result).to_csv(f'{config["root_path"]}/gossipcop_qwen_image_caption.csv',index=False)

