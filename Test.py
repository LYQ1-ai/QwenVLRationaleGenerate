from openai import OpenAI

import mimetypes
import base64


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

client = OpenAI(  # 使用异步客户端
            base_url='http://localhost:8000/v1/',
            api_key='no',
        )



a = {
    'role': 'system', 'content': [{
    'type': 'text',
    'text': '您是一名新闻真实性分析员。下面使用 <text></text> 标记的文字是一篇新闻报道的摘要,给出的图片为该新闻报道的封面。\\n        请结合文字与图片从图片描述的角度逐步分析该新闻文章的真实性，并用中文给出判断依据，字数限制在256以内\\n        按以下格式输出：\\n        - 真实性：一个词：真 或 假\\n        - 原因： 从图片描述的角度判断新闻真伪的依据。'
    }]
}
response = client.chat.completions.create(
    model="/home/lyq/Model/Qwen2-VL-72B-Instruct-GPTQ-Int4",
    messages=[a],
    max_tokens=300
)
print(response)