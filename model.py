import itertools
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
import base64
import json

import requests

from openai import OpenAI
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from vllm import LLM, SamplingParams

from Util import generate_remote_qwen_msg, generate_msg



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

class Qwen:

    def __init__(self, model_dir):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

    def chat(self,messages,**kwargs):
        """
            [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        """
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        max_new_tokens = kwargs.get("max_new_tokens", 256)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

class RemoteQwen:

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.client = OpenAI(
            base_url=f"http://localhost:8000/v1",
            api_key="token-abc123",
        )

    def chat(self, messages, **kwargs):
        # 设置默认参数值
        temperature = kwargs.get('temperature', 0.7)  # 默认温度值
        top_p = kwargs.get('top_p', 0.8)              # 默认核采样概率
        max_tokens = kwargs.get('max_tokens', 256)     # 默认最大生成标记数
        #extra_body = kwargs.get('extra_body',None)

        return self.client.chat.completions.create(
            model=self.model_dir,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            #extra_body=extra_body # {"guided_regex": "\w+@\w+\.com\n", "stop": ["\n"]}
        ).choices[0].message.content


class VLLMQwen:
    def __init__(self, model_dir,**kwargs):
        tensor_parallel_size = kwargs.get('tensor_parallel_size', 2)
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.8)
        repetition_penalty = kwargs.get('repetition_penalty', 1.05)
        max_tokens = kwargs.get('max_tokens', 512)
        self.llm = LLM(model_dir,
                       tensor_parallel_size = tensor_parallel_size,
                       gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.8),
                       trust_remote_code=True)
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, max_tokens=max_tokens)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
    def chat(self,messages,**kwargs):
        system_prompt = [msg for msg in messages if msg['role']=='system'][0]
        user_prompts = [msg for msg in messages if msg['role']=='user']
        input_prompts = [ [system_prompt,prompt] for prompt in user_prompts]
        input_ids = [self.tokenizer.apply_chat_template(
            input_prompt,
            tokenize=False,
            add_generation_prompt=True
        ) for input_prompt in input_prompts ]
        outputs = self.llm.generate(input_ids, self.sampling_params)
        return [ output.outputs[0].text for output in outputs]








