#!/bin/bash

export CUDA_VISIBLE_DEVICES=""

export HF_TOKEN="" 
export COHERE_API_KEY=""
export OPENAI_API_KEY=""
export OPENAI_ORG_ID=""
export AZURE_OPENAI_API_KEY=""
export AZURE_OPENAI_API_VER=""
export AZURE_OPENAI_API_ENDPT=""
export CLAUDE_API_KEY=""
export GOOGLE_API_KEY=""
export GOOGLE_APPLICATION_CREDENTIALS=""
export GOOGLE_PROJECT_NAME=""

# Define model keys
MODEL_KEYS=(
    "gpt-4-1106-preview"
    "gpt-3.5-turbo-1106"
    "aya-101"
    "gemini-pro"
    "claude-3-opus-20240229"
    "claude-3-sonnet-20240229"
    "claude-3-haiku-20240307"
    "Qwen1.5-72B-Chat"
    "Qwen1.5-14B-Chat"
    "Qwen1.5-32B-Chat"
    "text-bison-002"
    "c4ai-command-r-v01"
    "c4ai-command-r-plus"
    "aya-23"
    "SeaLLM-7B-v2.5"
    "Merak-7B-v4"
    "jais-13b-chat"
)

for model_key in "${MODEL_KEYS[@]}"; do
    python multiple_choice_evaluation.py --model "$model_key" \
        --model_cache_dir '.cache' \
        --mc_dir './mc_data' \
        --questions_file 'mc_questions_file.csv' \
        --response_file "${model_key}-mc_res.csv" \
        --temperature 0 \
        --top_p 1 \
        --gpt_azure 'True'
done