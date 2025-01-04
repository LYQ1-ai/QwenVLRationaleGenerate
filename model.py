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

from Util import generate_remote_qwen_msg, generate_msg
from data_loader import label_dict


class RemoteQwenVL:


    def __init__(self,model_dir='/home/lyq/Model/Qwen2-VL-72B-Instruct-GPTQ-Int4'):
        self.model_dir = model_dir
        self.url = "http://localhost:8000/v1"
        self.client = OpenAI(
            base_url=self.url,
            api_key="token-abc123",
        )

    def chat(self,messages):
        """
        :param messages: [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        'text':"Describe the following pictures"
                    },
                    {
                        "type": "image_url",
                        'image_url': {
                            'url':f"data:image/jpeg;base64,{encode_image('/home/lyq/DataSet/FakeNews/gossipcop/images/gossipcop-541230_top_img.png')}"
                        }
                    }
                ]
            },
        ]
        :return: str
        """
        return self.client.chat.completions.create(
            model=self.model_dir,
            messages=messages
        ).choices[0].message.content


    def batch_inference(self,messages):
        return [
            self.chat([msg])
            for msg in messages
        ]

    def batch_inference_v2(self,texts,image_paths):
        batch_size = len(texts)
        assert batch_size == len(image_paths)
        messages = [generate_remote_qwen_msg(texts[i],image_paths[i]) for i in range(batch_size)]
        return self.batch_inference(messages)





class QwenVL:
    def __init__(self, model_dir):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)

    def chat(self,messages,max_len=512):
        """
        :param messages: [
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
        :return: str
        """
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to('cuda')
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_len)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]

    def batch_inference(self,messages,max_len=512):
        return [
            self.chat([msg],max_len=max_len) for msg in messages
        ]

    def batch_inference_v2(self,texts,image_paths):
        batch_size = len(texts)
        assert batch_size == len(image_paths)
        messages = [generate_msg(texts[i], image_paths[i]) for i in range(batch_size)]
        return self.batch_inference(messages)








class LLMPredictDataset(Dataset):

    def __init__(self,df):
        self.df = df

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        result = {
                'text': row['text'].tolist(),
                'label': row['label'].apply(lambda x: label_dict[x]).tolist(),
            }
        if 'rationale' in self.df.columns:
            result['rationale'] = row['rationale'].tolist()
        return result


