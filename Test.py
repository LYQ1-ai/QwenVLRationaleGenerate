import multiprocessing
import os
from pprint import pprint


from model import RemoteQwen, VLLMQwen

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
if __name__ == '__main__':

    multiprocessing.set_start_method('spawn')
    # 以下是一些示例：
    prompt = "Tell me something about large language models."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]



    pprint(messages)


    model_dir = '/home/lyq/Model/Qwen2.5-72B-Instruct-GPTQ-Int8'

    model = VLLMQwen(model_dir)

    outputs  = model.chat(messages)

    print(outputs)
