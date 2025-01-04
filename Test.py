from pprint import pprint

import pandas as pd
from openai import OpenAI

import Util
import data_loader
from model import RemoteQwen
# 以下是一些示例：
messages = [
    {
        'role': 'user',
        'content': [
            {
                'type':'text',
                'text':"hello"
            }
        ]
    }
]




pprint(messages)


model_dir = '/media/lyq/d/Model/Qwen2.5-72B-Instruct-GPTQ-Int8'
ip = 'localhost'
port = 8000
#model = RemoteQwen(model_dir)

#print(model.chat(messages))
out = """- 真实性：假
- 原因：1. 来源：该信息来源于微博，这是一个社交媒体平台，其中的信息未经核实，可能存在不准确或误导性内容。
2. 价格差异：信息中提到的美国食用油价格（1.3美元/桶）与中国的市场价格（40-60元/桶）存在巨大差异，这在实际中不太可能。
3. 货币汇率：信息中提到的1.3美元折合人民币8.58元，但未考虑到实际汇率波动，且未提供具体的汇率信息。
4. 产品描述：信息中提到的“绿色纯天然的，不是转基因的”这一描述没有提供任何证据或来源，这可能是为了吸引读者的注意。
5. 语言：信息中使用了“震惊”、“想都不敢想”等情感化语言，这可能是为了引起读者的共鸣和转发，但并不符合新闻报道的专业性和客观性。"""
print(Util.validate_model_zh_output(out))