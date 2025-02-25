 vllm serve /home/lyq/Model/Qwen2-VL-72B-Instruct-GPTQ-Int4 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt image=10 \
    --trust-remote-code