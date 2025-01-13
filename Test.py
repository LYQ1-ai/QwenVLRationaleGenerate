import os

from modelscope import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM


if __name__ == '__main__':
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    model_dir = '/home/lyq/Model/Qwen2-VL-72B-Instruct-GPTQ-Int4'
    llm = LLM(model_dir,
                           tensor_parallel_size=4,
                           gpu_memory_utilization=0.8,
                           trust_remote_code=True)


    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "file:///home/lyq/DataSet/FakeNews/gossipcop/images/gossipcop-541230_top_img.png",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    processor = AutoProcessor.from_pretrained(model_dir)
    # Preparation for inference
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }
    outputs = llm.generate([llm_inputs])
    generated_text = outputs[0].outputs[0].text
    print(generated_text)