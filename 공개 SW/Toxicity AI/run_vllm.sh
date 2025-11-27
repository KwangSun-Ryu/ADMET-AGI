#!/usr/bin/env bash

# VLLM 컨테이너 실행 예시 (경로/포트/GPU는 환경에 맞게 수정)
docker run -it --rm \
  --gpus all \
  -p 30002:8000 \
  -v "/mnt/e/Google Drive/External SSD/HealthCare/ADMET/코드/ADMET-AGI/Toxicity AI":/workspace:rw \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TP_SIZE=1 \
  -e MODEL_PATH=/workspace/25TOXMC_Blowfish_v1.0.9-AWQ \
  -e CHAT_TEMPLATE_PATH=/workspace/no_tool_chat_template_qwen3.jinja \
  -e GPU_MEMORY_UTILIZATION=0.9 \
  -e DTYPE=bfloat16 \
  vllm-25admet-vllm \
  --host=0.0.0.0 \
  --model=/workspace/25TOXMC_Blowfish_v1.0.9-AWQ \
  --dtype=bfloat16 \
  --chat-template=/workspace/no_tool_chat_template_qwen3.jinja \
  --gpu-memory-utilization=0.9 \
  --tensor-parallel-size=1 \
  --max-model-len=16384

# 컨테이너 내부에서 .env 변수 설정 후 평가 스크립트 실행 예시
# export BASE_URL=http://<host>:30002/v1/
# export GPT_API_KEY=<your_gpt_key>
# python3 mobile_eval_e.py
# python3 mmlu_toxic.py
# python3 chem_cot.py
