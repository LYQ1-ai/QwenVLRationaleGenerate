import asyncio

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info

from openai import AsyncOpenAI, APIError
import openai
from openai import OpenAI
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from vllm import LLM, SamplingParams

from Util import generate_remote_qwen_msg, generate_msg






class QwenVL:
    def __init__(self, model_dir):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.image_url_type = 'local'
        min_pixels = 256*28*28
        max_pixels = 1280*28*28
        self.processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)

    def chat(self,messages,max_len=512):
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
        self.image_url_type = 'local'
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
        self.image_url_type = 'remote'

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
        tensor_parallel_size = kwargs.get('tensor_parallel_size', 1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.llm = LLM(model_dir,
                       tensor_parallel_size=tensor_parallel_size,
                       gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.8),
                       trust_remote_code=True)
        self.image_url_type = 'local'

    def chat(self,messages,**kwargs):
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.8)
        repetition_penalty = kwargs.get('repetition_penalty', 1.05)
        max_tokens = kwargs.get('max_tokens', 512)
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p,
                                              repetition_penalty=repetition_penalty, max_tokens=max_tokens)

        system_prompt = [msg for msg in messages if msg['role']=='system'][0]
        user_prompts = [msg for msg in messages if msg['role']=='user']
        input_prompts = [ [system_prompt,prompt] for prompt in user_prompts]
        input_ids = [self.tokenizer.apply_chat_template(
            input_prompt,
            tokenize=False,
            add_generation_prompt=True
        ) for input_prompt in input_prompts ]
        outputs = self.llm.generate(input_ids, sampling_params)
        return [ output.outputs[0].text for output in outputs]



class VLLMQwenVL:

    def __init__(self, model_dir,**kwargs):
        tensor_parallel_size = kwargs.get('tensor_parallel_size', 2)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.llm = LLM(model_dir,
                       tensor_parallel_size=tensor_parallel_size,
                       gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.8),
                       trust_remote_code=True,
                       limit_mm_per_prompt={"image": 10})
        self.image_url_type = 'local'

    def chat(self,messages,**kwargs):
        temperature = kwargs.get('temperature', 0.7)
        top_p = kwargs.get('top_p', 0.8)
        repetition_penalty = kwargs.get('repetition_penalty', 1.05)
        max_tokens = kwargs.get('max_tokens', 512)
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p,
                                         repetition_penalty=repetition_penalty, max_tokens=max_tokens)
        input_messages = []
        for input_prompt in messages:
            mm_data = {}
            text = self.processor.apply_chat_template(input_prompt, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(input_prompt)
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            llm_inputs = {
                "prompt": text,
                "multi_modal_data": mm_data,
            }
            input_messages.append(llm_inputs)
        outputs = self.llm.generate(input_messages, sampling_params)
        return [output.outputs[0].text for output in outputs]

class RemoteDeepSeek:

    def __init__(self,url, model_name,api_key=None):
        self.client = OpenAI(
            base_url=url,
            api_key=api_key,
        )
        self.model_name = model_name
        self.image_url_type = 'remote'

    def chat(self, messages, **kwargs):
        # 设置默认参数值
        temperature = kwargs.get('temperature', 0.7)  # 默认温度值
        top_p = kwargs.get('top_p', 0.8)              # 默认核采样概率
        max_tokens = kwargs.get('max_tokens', 256)     # 默认最大生成标记数
        #extra_body = kwargs.get('extra_body',None)

        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            #extra_body=extra_body # {"guided_regex": "\w+@\w+\.com\n", "stop": ["\n"]}
        ).choices[0].message.content

    def batch_inference(self, messages, **kwargs):
        system_msg = [msg for msg in messages if msg['role']=='system'][0]
        responses = []
        for msg in messages:
            if msg['role']=='user':
                responses.append(self.chat([system_msg,msg],**kwargs))
        return responses




class AsyncRemoteDeepSeek:

    def __init__(self, url, model_name, api_key=None):
        self.client = AsyncOpenAI(  # 使用异步客户端
            base_url=url,
            api_key=api_key,
        )
        self.model_name = model_name
        self.image_url_type = 'remote'

    async def chat(self, messages, **kwargs):  # 改为异步方法
        # 设置默认参数值
        temperature = kwargs.get('temperature', None)
        top_p = kwargs.get('top_p', None)
        max_tokens = kwargs.get('max_tokens', None)

        max_retries = 10
        retries = 0

        while retries < max_retries:
            try:
                response = await self.client.chat.completions.create(  # 添加await
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                retries += 1
                if retries < max_retries:
                    await asyncio.sleep(2)  # 等待2秒后重试
                else:
                    print(f'error: {e}')
                    return None

    async def batch_inference(self, messages, **kwargs):  # 改为异步方法
        # 并行处理所有用户消息
        tasks = [self.chat(msg,**kwargs) for msg in messages]
        responses = await asyncio.gather(*tasks)  # 并行执行所有请求
        return responses





