from openai import OpenAI

client = OpenAI(  # 使用异步客户端
            base_url='http://172.31.136.239:32774/v1/',
            api_key='no',
        )

print(client.chat.completions.create(messages=[{"role": "user", "content": "Hello!"}],
                                     model="DeepSeek-R1-Q4_K_M"))