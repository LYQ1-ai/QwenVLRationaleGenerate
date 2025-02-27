import base64
import mimetypes
import os
import pickle
from pprint import PrettyPrinter
from tempfile import NamedTemporaryFile

import data_loader

zh_rationale_type_dict = {
    'td':"文字描述",
    'cs':"社会常识",
    'img':'图片描述',
    'itc':'图文一致性'
}
en_rationale_type_dict = {
    'td':"Textual description",
    'cs':"Common sense",
    'img':'Image Description',
    'itc':'Text-Image Consistency'
}


zh_td_system_prompt = """您是一名新闻真实性分析员。下面使用 <text></text> 标记的文字是一篇新闻报道的摘要。
        请从文字描述的角度逐步分析该新闻文章的真实性，并用中文给出判断依据。
        按以下格式输出：
        - 真实性：一个词：真 或 假
        - 原因： 从文字描述的角度判断新闻真伪的依据。"""

en_td_system_prompt = """You are a news veracity analyst. The text below, tagged with <text></text>, is a summary of a news article.
        Please analyze the authenticity of the news article step by step from the perspective of Textual description and give the basis for your judgment in English.
        Output in the following format:
        - Authenticity: One word: Real or Fake
        - Reason: The basis for judging the authenticity of a news article from the perspective of Textual description.
"""

zh_vl_itc_system_prompt = """您是一名新闻真实性分析员。下面使用 <text></text> 标记的文字是一篇新闻报道的摘要,给出的图片为该新闻报道的封面。
        请结合文字与图片从图文一致性的角度逐步分析该新闻文章的真实性，并用中文给出判断依据，字数限制在{max_tokens}以内
        按以下格式输出：
        - 真实性：一个词：真 或 假
        - 原因： 从图文一致性的角度判断新闻真伪的依据。"""

en_vl_itc_system_prompt = """You are a news veracity analyzer. The text tagged with <text></text> is a summary of a news article, and the picture given is the cover of the news article.
        Please analyze the authenticity of the news article step by step from the perspective of Text-Image Consistency by combining the text and the picture, and give the basis for your judgment in English, and limit the word count to {max_tokens}.
        Output in the following format:
        - Authenticity: One word: Real or Fake
        - Reason: The basis for judging the authenticity of the news article from the perspective of Text-Image Consistency.
"""

zh_vl_img_system_prompt = """您是一名新闻真实性分析员。下面给出的图片为该新闻报道的封面。
        请从图片描述的角度逐步分析该新闻的真实性，并用中文给出判断依据，字数限制在{max_tokens}以内
        按以下格式输出：
        - 真实性：一个词：真 或 假
        - 原因： 从图片描述的角度判断新闻真伪的依据。"""

en_vl_img_system_prompt = """You are a news veracity analyst. The picture given below is the cover of a news report.
        Please analyze the authenticity of the news report from the perspective of the Image description, and give the basis for your judgment in English, with a word limit of {max_tokens}.
        Output in the following format:
        - Authenticity: One word: Real or Fake
        - Reason: The basis for judging the authenticity of the news from the perspective of the Image description.
"""

zh_input_prompt = """输入：<text>{news_text}</text>"""
zh_output_few_shot_prompt = """- 真实性：{label}
- 原因：{rationale}
"""


en_input_prompt = """Input: <text>{news_text}</text>"""
en_output_few_shot_prompt = """- Authenticity: {label}
- Reason: {rationale}
"""




zh_rationale_prompt_dict = {
    'td':zh_td_system_prompt,
    'img':zh_vl_img_system_prompt,
    'itc':zh_vl_itc_system_prompt
}

en_rationale_prompt_dict = {
    'td':en_td_system_prompt,
    'img':en_vl_img_system_prompt,
    'itc':en_vl_itc_system_prompt
}




def image_path2image_url(image_path):
    # 检测文件MIME类型
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith('image/'):
        raise ValueError(f"Unsupported image format: {image_path}")

    # 读取并编码图片
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")

    # 构造符合规范的Data URL
    return f"data:{mime_type};base64,{encoded_image}"




class  Prompt:

    def __init__(self, lang,rationale_type,use_few_shot,max_tokens):
        """
        lang : zh or en
        rationale_type : td  or itc or img
        use_few_shot : True or False
        """
        self.lang = lang
        if lang == 'zh':
            self.system_prompt = zh_rationale_prompt_dict[rationale_type].format(max_tokens=max_tokens)
            self.input_prompt = zh_input_prompt
            if use_few_shot:
                self.system_prompt += "以下是一些示例：\n"
                self.few_shot_output = zh_output_few_shot_prompt

        elif lang == 'en':
            self.system_prompt = en_rationale_prompt_dict[rationale_type].format(max_tokens=max_tokens)
            self.input_prompt = en_input_prompt
            if use_few_shot:
                self.system_prompt += "Here are some examples:\n"
                self.few_shot_output = en_output_few_shot_prompt




def wrapper_remote_msg_image_content(image_path):
    return {'type':'image_url','image_url':{
        'url':image_path2image_url(image_path)
    }}

def wrapper_local_msg_image_content(image_path):
    return {
        'type':'image',
        'image':f'file://{image_path}'
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


class CustomPrinter(PrettyPrinter):
    def _format(self, obj, *args):
        if isinstance(obj, str) and len(obj) > 10000:  # 自定义超长字符串处理
            return f'"{obj[:20]}...{obj[-20:]}" ({len(obj)} chars)', True
        return super()._format(obj, *args)




class IMGRationaleMessageUtil:

    def __init__(self, lang,use_few_shot,max_tokens,image_url_type):
        self.lang = lang
        self.prompt = Prompt(lang,'img',use_few_shot,max_tokens)
        self.image_url_wrapper_func = wrapper_remote_msg_image_content if image_url_type == 'remote' else wrapper_local_msg_image_content
        self.valid_output_func = validate_model_zh_output if lang == 'zh' else validate_model_en_output

    def valid_output(self, output):
        return self.valid_output_func(output)

    def wrapper_message0(self,input_image_path,few_shot_data):

        """
        包装单个msg
        input_image_path : abs image path
        few_shot_data : few shot data ,[
            {image_path,rationale,label}
        ]
        """
        msg = [
            {
                'role':'system',
                'content':self.prompt.system_prompt
            }
        ]
        if few_shot_data:
            for item in few_shot_data:
                msg.append({
                    'role':'user',
                    'content':[self.image_url_wrapper_func(item['image_path'])]
                })
                msg.append({
                    'role':'assistant',
                    'content':self.prompt.few_shot_output.format(rationale=item['rationale'],label=item['label'])
                })

        msg.append({
            'role': 'user',
            'content': [self.image_url_wrapper_func(input_image_path)]
        })

        return msg

    def wrapper_message(self,item,few_shot_data):
        return self.wrapper_message0(item['image_path'],few_shot_data)


class ITCRationaleMessageUtil:
    def __init__(self, lang, use_few_shot, max_tokens, image_url_type):
        self.lang = lang
        self.prompt = Prompt(lang, 'itc', use_few_shot, max_tokens)
        self.image_url_wrapper_func = wrapper_remote_msg_image_content if image_url_type == 'remote' else wrapper_local_msg_image_content
        self.valid_output_func = validate_model_zh_output if lang == 'zh' else validate_model_en_output

    def valid_output(self, output):
        return self.valid_output_func(output)

    def wrapper_message0(self,input_image_path,input_text,few_shot_data):

        """
        包装单个msg
        input_image_path : abs image path
        few_shot_data : few shot data ,[
            {image_path,text,rationale,label}
        ]
        """
        msg = [
            {
                'role':'system',
                'content':self.prompt.system_prompt
            }
        ]
        if few_shot_data:
            for item in few_shot_data:
                msg.append({
                    'role':'user',
                    'content':[
                        {'type':'text','text': self.prompt.input_prompt.format(news_text=item['text'])},
                        self.image_url_wrapper_func(item['image_path'])
                    ]
                })
                msg.append({
                    'role':'assistant',
                    'content':self.prompt.few_shot_output.format(rationale=item['rationale'],label=item['label'])
                })

        msg.append({
            'role': 'user',
            'content': [
                {'type': 'text', 'text': self.prompt.input_prompt.format(news_text=input_text)},
                self.image_url_wrapper_func(input_image_path)
            ]
        })

        return msg

    def wrapper_message(self,item,few_shot_data):
        return self.wrapper_message0(item['image_path'],item['text'],few_shot_data)




def get_messageUtil(rationale_type,lang,use_few_shot,max_tokens,image_url_type):
    if rationale_type == 'img':
        return IMGRationaleMessageUtil(lang,use_few_shot,max_tokens,image_url_type)
    elif rationale_type == 'itc':
        return ITCRationaleMessageUtil(lang,use_few_shot,max_tokens,image_url_type)
    else:
        # TODO 处理其他类型
        raise ValueError(f"Invalid rationale_type: {rationale_type}")


class CacheManager:
    def __init__(self, cache_file_path):
        self.cache_file_path = cache_file_path

    def load_cache(self):
        """
        加载缓存数据，返回字典格式
        1. 检查缓存文件是否存在
        2. 反序列化时处理可能的数据损坏异常
        3. 文件/目录自动创建（通过save_cache）
        """
        if not os.path.exists(self.cache_file_path):

            return {}

        try:
            with open(self.cache_file_path, 'rb') as f:
                return pickle.load(f)
        except (IOError, EOFError, pickle.UnpicklingError) as e:
            print(f"缓存加载失败: {str(e)}")
            return {}

    def save_cache(self, cache_data):
        """
        持久化缓存数据（优化版）
        1. 增加路径规范化处理
        2. 使用更安全的NamedTemporaryFile
        3. 细化错误类型捕获
        """
        try:
            # 获取规范化的目录路径
            dir_path = os.path.dirname(self.cache_file_path)

            # 创建目录（exist_ok=True表示目录存在时不报错）
            if dir_path:  # 防止根目录的情况
                os.makedirs(dir_path, exist_ok=True)

            # 使用NamedTemporaryFile更安全（自动处理清理）
            with NamedTemporaryFile(mode='wb', dir=dir_path, delete=False) as f:
                temp_path = f.name
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 原子替换操作
            os.replace(temp_path, self.cache_file_path)

        except (PermissionError, FileNotFoundError) as e:
            print(f"目录创建失败: {str(e)}")
            raise
        except (pickle.PicklingError, TypeError) as e:
            print(f"序列化失败: {str(e)}")
            raise
        except OSError as e:
            print(f"文件操作异常: {str(e)}")
            raise

def filter_batch_input(batch_input,exist_ids):
    return [item for item in batch_input if item['source_id'] not in exist_ids]


def filter_batch_out(batch_out,exist_ids):
    """
    batch_out = [
        source_id:{
            authenticity:str,
            reason:str,
        }
    ]
    """
    def valid_out(source_id:str,out_item:dict,exist_ids:set):
        return all([
            'authenticity' in out_item,
            'reason' in out_item,
            out_item['authenticity'] in data_loader.label_str2int_dict.keys(),
            out_item['reason'] is not None or len(out_item['reason']) > 0,
            source_id is not None,
            len(source_id) > 0,
            source_id not in exist_ids
        ])

    return {source_id:out for source_id,out in batch_out.items() if valid_out(source_id,out,exist_ids)}


