import copy
import itertools
import pickle
import re
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
import base64
import json

import requests

from openai import OpenAI
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor




prompt_mode = {
    'td': {'text'},
    'cs': {'text','image'},
}



def image_path2image_url(image_path):
    with open(image_path, "rb") as image_file:
        base64_url = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_url}"


def generate_remote_qwen_msg(text,image_path):
    """
    :param text: prompt text
    :param image_path: local image path
    :return: dict
    """
    # text_msg = self.prompt.format(news_text=text)
    return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    'text':text
                },
                {
                    "type": "image_url",
                    'image_url': {
                        'url': image_path2image_url(image_path)
                    }
                }
            ]
    }


def generate_msg(text,image_path):
    """
    :param text: prompt text
    :param image_path: local image path
    :return: messages: [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https | file://",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    """
    #text_msg = self.prompt.format(news_text=text)
    return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    'text':text
                },
                {
                    "type": "image",
                    'image': f'file://{image_path}'
                }
            ]
    }


zh_system_prompt = """您是一名新闻真实性分析员。下面使用 <text></text> 标记的文字是一篇新闻报道的摘要。
        请从{rationale_type}的角度逐步分析该新闻文章的真实性，并用中文给出判断依据。
        按以下格式输出：
        - 真实性：一个词：真 或 假
        - 原因： 从{rationale_type}的角度判断新闻真伪的依据。"""

zh_input_prompt = """输入：<text>{news_text}</text>
输出: """

zh_text_few_shot_prompt = """输入：<text>{text}</text>
输出: 
- 真实性：{label}
- 原因：{rationale}
"""

zh_rationale_type_dict = {
    'td':"文字描述",
    'cs':"社会常识"
}


def validate_model_zh_output(output):
    try:
        auth,reason = output.split("\n",maxsplit=1)
        auth = auth.split('：',maxsplit=1)[-1]
        if '真' in auth:
            auth = '真'
        elif '假' in auth:
            auth = '假'
        else:
            auth = '其他'

        reason = reason.split('：',maxsplit=1)[1]
        return {
            'auth':auth,
            'reason':reason
        }

    except Exception as e:
        print(e)
        return {}




def validate_model_en_output(output):
    try:
       text = output
       res = {}
       auth,reason = text.split('\n',maxsplit=1)
       if '假' in auth:
           res['authenticity'] = '假'
       elif '真' in auth:
           res['authenticity'] = '真'
       elif '其他' in auth:
           res['authenticity'] = '其他'
       if 'reason:' in reason:
           res['reason'] = reason.split('reason:',maxsplit=1)[1]
       elif 'Reason:' in reason:
           res['reason'] = reason.split('Reason:',maxsplit=1)[1]
       else:
           res['reason'] = None
       return res
    except Exception as e:
        return {}



class TextMessageUtil:

    def __init__(self,lang,rationale_type,few_shot=True):
        self.lang = lang
        if lang == 'zh':
            self.system_prompt = zh_system_prompt
            self.system_prompt = self.system_prompt.format(rationale_type=zh_rationale_type_dict[rationale_type])
            self.input_prompt = zh_input_prompt
            self.few_shot = few_shot
            if few_shot:
                self.system_prompt += "以下是一些示例：\n"
                self.few_shot_template = zh_text_few_shot_prompt

    def generate_text_messages(self,text,few_shot_data=None):
        """
        :param text: news text
        :param rationale_type: rationale_type td or cs
        :param few_shot_data: few_shot_data = [{
            text : str,
            rationale: str,
            label: real or fake for en ,真 或者 假 for zh
        }]
        """
        input_prompt = self.input_prompt.format(news_text=text)
        system_prompt = self.system_prompt
        if few_shot_data and self.few_shot:
            for shot in few_shot_data:
                system_prompt += self.few_shot_template.format(**shot)

        return [
            {'role':'system','content':system_prompt},
            {'role':'user','content':input_prompt}
        ]

    def valid_output(self,out):
        if self.lang == 'zh':
            return validate_model_zh_output(out)
        else:
            return validate_model_en_output(out)


def save_cache(cache_file, data):
    """Helper function to save the cache."""
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)