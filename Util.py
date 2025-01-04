import itertools
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

