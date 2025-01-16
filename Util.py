import base64
import os
import pickle

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

zh_system_prompt = """您是一名新闻真实性分析员。下面使用 <text></text> 标记的文字是一篇新闻报道的摘要。
        请从{rationale_type}的角度逐步分析该新闻文章的真实性，并用中文给出判断依据。
        按以下格式输出：
        - 真实性：一个词：真 或 假
        - 原因： 从{rationale_type}的角度判断新闻真伪的依据。"""

en_system_prompt = """You are a news veracity analyst. The text below, tagged with <text></text>, is a summary of a news article.
        Please analyze the authenticity of the news article step by step from the perspective of {rationale_type} and give the basis for your judgment in English.
        Output in the following format:
        - Authenticity: One word: Real or Fake
        - Reason: The basis for judging the authenticity of a news article from the perspective of {rationale_type}.
"""




zh_input_prompt = """输入：<text>{news_text}</text>
输出: """

en_input_prompt = """Input: <text>{news_text}</text>
Output: """

zh_text_few_shot_prompt = """输入：<text>{text}</text>
输出: 
- 真实性：{label}
- 原因：{rationale}
"""

en_text_few_shot_prompt = """Input: <text>{text}</text>
Output: 
- Authenticity: {label}
- Reason: {rationale}
"""

zh_rationale_type_dict = {
    'td':"文字描述",
    'cs':"社会常识"
}
en_rationale_type_dict = {
    'td':"Textual description",
    'cs':"Common sense"
}

zh_caption_prompt = """你是一个新闻图像描述助手，下面使用<text></text>括起来的是一篇新闻文章，给出的图片是该新闻的封面，请结合新闻文本描述该图像\n"""
en_caption_prompt = """You are a news image description assistant. Below is a news article enclosed in <text></text> and the given image is the cover of the news, describe the image in relation to the news text\n"""

en_summary_system_prompt = """You are a news summarization assistant. The following user input is a news clip. Summarize the news clip in one paragraph, limiting it to {max_len} words, and focusing on the source, time, place, people, and events of the news:
"""
zh_summary_system_prompt = """你是一个新闻总结助手，下面用户输入的为一篇新闻片段，用一段话对该新闻片段进行概括，字数限制在{max_len}个字以内，并且侧重于新闻的来源，时间，地点，人物，事件:
"""

input_prompt = """{news_text}"""

def image_path2image_url(image_path):
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read())
    encoded_image_text = encoded_image.decode("utf-8")
    return f"data:image;base64,{encoded_image_text}"


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
            'authenticity':auth,
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
       if 'fake' in auth.lower():
           res['authenticity'] = 'fake'
       elif 'real' in auth.lower():
           res['authenticity'] = 'real'
       else:
           res['authenticity'] = 'other'
       if 'Reason:' in reason:
           res['reason'] = reason.split('Reason:',maxsplit=1)[1]
       elif 'reason:' in reason:
           res['reason'] = reason.split('reason:',maxsplit=1)[1]
       else:
           res['reason'] = None
       return res
    except Exception as e:
        return {}



class VLMessageUtil:

    def __init__(self, system_prompt,input_prompt):
        self.system_prompt = system_prompt
        self.input_prompt = input_prompt

    def generate_vl_message(self, texts, images_path,image_url_type='local'):
        system_msg = {'role':'system','content':[
            {'type':'text','text':self.system_prompt},
        ]}

        assert len(texts) == len(images_path)
        messages = [ system_msg]
        for text,image_path in zip(texts,images_path):
            if image_url_type == 'remote':
                messages.append(generate_remote_qwen_msg(text,image_path))
            elif image_url_type == 'local':
                messages.append(generate_msg(text,image_path))

        return messages



class TextMessageUtil:

    def __init__(self, system_prompt,input_prompt):
        self.system_prompt = system_prompt
        self.input_prompt = input_prompt

    def generate_text_message(self, texts):
        input_prompts = [self.input_prompt.format(news_text=text) for text in texts]
        system_prompt = self.system_prompt
        messages = [
            {'role': 'system', 'content': system_prompt},
        ]
        messages.extend([{'role': 'user', 'content': input_prompt} for input_prompt in input_prompts])
        return messages



class RationaleMessageUtil:

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
        elif lang == 'en':
            self.system_prompt = en_system_prompt
            self.system_prompt = self.system_prompt.format(rationale_type=en_rationale_type_dict[rationale_type])
            self.input_prompt = en_input_prompt
            self.few_shot = few_shot
            if few_shot:
                self.system_prompt += "Here are some examples:\n"
                self.few_shot_template = en_text_few_shot_prompt



    def generate_text_messages(self,texts,few_shot_data=None):
        """
        :param texts: news text
        :param rationale_type: rationale_type td or cs
        :param few_shot_data: few_shot_data = [{
            text : str,
            rationale: str,
            label: real or fake for en ,真 或者 假 for zh
        }]
        """
        input_prompts = [self.input_prompt.format(news_text=text) for text in texts]
        system_prompt = self.system_prompt
        if few_shot_data and self.few_shot:
            for shot in few_shot_data:
                system_prompt += self.few_shot_template.format(**shot)

        messages =  [
            {'role':'system','content':system_prompt},
        ]
        messages.extend([{'role':'user','content':input_prompt} for input_prompt in input_prompts])
        return messages

    def valid_output(self,out):
        if self.lang == 'zh':
            return validate_model_zh_output(out)
        elif self.lang == 'en':
            return validate_model_en_output(out)


def save_cache(cache_file, data):
    """Helper function to save the cache."""
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)


def cal_rationale_metrics(y_pred, y_true):
    # 计算真实类和伪造类的各项指标
    recall_real = recall_score(y_true, y_pred, average=None, labels=[0])[0]
    recall_fake = recall_score(y_true, y_pred, average=None, labels=[1])[0]
    precision_real = precision_score(y_true, y_pred, average=None, labels=[0])[0]
    precision_fake = precision_score(y_true, y_pred, average=None, labels=[1])[0]
    f1_real = f1_score(y_true, y_pred, average=None, labels=[0])[0]
    f1_fake = f1_score(y_true, y_pred, average=None, labels=[1])[0]

    # 宏平均指标由真实类和伪造类的指标算术平均得出
    recall_macro = (recall_real + recall_fake) / 2
    precision_macro = (precision_real + precision_fake) / 2
    f1_macro = (f1_real + f1_fake) / 2

    return {
        'acc': accuracy_score(y_true, y_pred),
        'recall': recall_macro,
        'recall_real': recall_real,
        'recall_fake': recall_fake,
        'precision': precision_macro,
        'precision_real': precision_real,
        'precision_fake': precision_fake,
        'f1_macro': f1_macro,
        'f1_real': f1_real,
        'f1_fake': f1_fake
    }



def local_image_url2image_path(image_url):
    # 去掉 'file://' 的前缀
    if image_url.startswith("file://"):
        image_path = image_url[7:]  # 去掉前7个字符
    else:
        raise ValueError("URL 必须以 'file://' 开头")

    # 如果 image_path 是相对路径，转换为绝对路径
    image_path = os.path.abspath(image_path)
    return image_path

def read_image_from_url(image_url):

    image_path = local_image_url2image_path(image_url)
    # 读取图片
    try:
        image = Image.open(image_path)
        return image
    except FileNotFoundError:
        print(f"图片未找到: {image_path}")
        return None
    except Exception as e:
        print(f"读取图片时发生错误: {e}")
        return None

