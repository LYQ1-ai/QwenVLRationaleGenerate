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

label_dict = {
    'fake':0,
    'real':1,
    'other':2,
    0:'fake',
    1:'real',
    2:'other',
    0.0:'fake',
    1.0:'real',
    2.0:'other'
}


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
        messages = [ generate_remote_qwen_msg(texts[i],image_paths[i]) for i in range(batch_size)]
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


class BalancedSampler(Sampler):
    def __init__(self, labels, batch_size=10):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.fake_indices = np.where(self.labels == 0)[0]
        self.real_indices = np.where(self.labels == 1)[0]

    def __iter__(self):
        num_batches = len(self.labels) // self.batch_size
        for _ in range(num_batches):
            # 随机抽取5个fake和5个real
            real_nums = self.batch_size // 2
            fake_nums = self.batch_size - real_nums
            fake_batch = np.random.choice(self.fake_indices, real_nums, replace=False)
            real_batch = np.random.choice(self.real_indices, fake_nums, replace=False)
            batch = np.concatenate((fake_batch, real_batch))
            np.random.shuffle(batch)
            yield batch  # 返回该批次的索引

    def __len__(self):
        return len(self.labels) // self.batch_size


class LLM:

    few_shot_prompt = """
The text encompassed by the tags <text></text> is a title of the news.
Please make a judgment on the authenticity of the news.
the output should contain only one word: real or fake.
Several examples are provided below.
"""
    few_example = """<text>{news_text}</text>  
            {label}"""



    @abstractmethod
    def chat(self,prompt,history):
        raise NotImplementedError


    def text2Msg(self,text):
        return {
            'role':'user',
            'content':text
        }

    def predict(self,news,few_shot_examples:dict[str,list[str]])->int:
        news_msg = self.few_example.format(news_text=news,label=' ')
        history = [self.text2Msg(self.few_shot_prompt)]
        few_shot_nums = len(few_shot_examples['label'])
        for i in range(few_shot_nums):
            text = few_shot_examples['text'][i]
            label = few_shot_examples['label'][i]
            history.append(self.text2Msg(self.few_example.format(news_text=text,label=label)))


        max_try = 10
        for i in range(max_try):
            response = self.chat(news_msg, history)
            print(response)
            if response in label_dict.keys():
                return label_dict[response]
        return -1


def get_few_shot(df,shot_nums=5):
    few_shot_df = df
    llm_ds = LLMPredictDataset(few_shot_df)
    sampler = BalancedSampler(df['label'],shot_nums)
    return itertools.cycle(iter(DataLoader(llm_ds,sampler=sampler)))




def generate_llm_predict(model,pred_df,few_shot_df, shot_nums=5):
    few_shot_iter = get_few_shot(few_shot_df,shot_nums)
    result = pred_df
    for i in tqdm(range(len(result))):
        line = result.loc[i,'line']
        print(f"predict line :{line}")
        news = result.loc[i,'text']
        pred = model.predict(news,next(few_shot_iter))
        result.loc[i,'llm_pred'] = pred
        result.loc[i,'llm_acc'] = int(pred == result.loc[i,'label'])

    result = result[result['llm_pred'] != -1]
    return result


def calculate_acc(df):
    print(f"sum data {df.shape[0]}")
    for rationale_name in prompt_mode.keys():
        legal_data_df = df[(df[f'{rationale_name}_pred']!=-1) & (df[f'{rationale_name}_rationale'] is not None)]
        print(f"{rationale_name} : acc {(legal_data_df[f'{rationale_name}_acc'] == 1).sum() / legal_data_df.shape[0]} , "
              f" acc_real {((legal_data_df[f'{rationale_name}_acc'] == 1) & (legal_data_df['label'] == 1)).sum() / (legal_data_df['label'] == 1).sum()} ,"
              f" acc_fake {((legal_data_df[f'{rationale_name}_acc'] == 1) & (legal_data_df['label'] == 0)).sum() / (legal_data_df['label'] == 0).sum()} ,")


