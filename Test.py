import yaml

import model


url = 'http://localhost:32768/v1'
api_key = 'no'
model_name = 'DeepSeek-R1-Q4_K_M'

deepseek = model.RemoteDeepSeek(url,model_name,api_key)

out = deepseek.chat(messages=[
    {
        "role": "user",
        "content": "hello"
    },
])
print(out)
