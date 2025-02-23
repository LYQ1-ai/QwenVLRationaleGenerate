import json
import os
import pickle

import pandas as pd

import DeepSeekPredict
import Util
from DeepSeekPredict import filter_illegal_data

"""
{
"id": "batch_req_wnaDys", 
"custom_id": "request-2", 
"response": {"status_code": 200, "request_id": "req_c187b3", 
    "body": {"id": "chatcmpl-9758Iw", "object": "chat.completion", "created": 1711475054, "model": "qwen-max", "choices": [{"index": 0, "message": {"role": "assistant", "content": "2 + 2 equals 4."}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 24, "completion_tokens": 15, "total_tokens": 39}, "system_fingerprint": null}}, "error": null}
"""

def read_batch_out(file_path,valid_out_put):
    res = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            json_result = json.loads(line)
            response_content = json_result['response']['body']['choices'][0]['message']['content']
            res[json_result['custom_id']] = valid_out_put(response_content)

        return res


def write_out2cache_file(cache_file,data):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            ans = pickle.load(f)
    else:
        ans = {}

    ans = filter_illegal_data(ans)
    ans.update(data)
    Util.save_cache(cache_file, ans)


if __name__ == '__main__':
    cache_file_path = 'cache/weibo_en/deepseek671B_pred.pkl'
    output_file_path = 'weibo_en_output.jsonl'
    msg_Util = DeepSeekPredict.DeepSeekPredictUtil2('en')
    batch_out = read_batch_out(output_file_path,msg_Util.process_output0)
    write_out2cache_file(cache_file_path,batch_out)

