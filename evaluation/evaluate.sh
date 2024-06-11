#!/bin/bash

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
)

# Define countries and languages as an associative array
declare -A COUNTRY_LANG
COUNTRY_LANG["UK"]="English"
COUNTRY_LANG["US"]="English"
COUNTRY_LANG["South_Korea"]="Korean"
COUNTRY_LANG["Algeria"]="Arabic"
COUNTRY_LANG["China"]="Chinese"
COUNTRY_LANG["Indonesia"]="Indonesian"
COUNTRY_LANG["Spain"]="Spanish"
COUNTRY_LANG["Iran"]="Persian"
COUNTRY_LANG["Mexico"]="Spanish"
COUNTRY_LANG["Assam"]="Assamese"
COUNTRY_LANG["Greece"]="Greek"
COUNTRY_LANG["Ethiopia"]="Amharic"
COUNTRY_LANG["Northern_Nigeria"]="Hausa"
COUNTRY_LANG["Azerbaijan"]="Azerbaijani"
COUNTRY_LANG["North_Korea"]="Korean"
COUNTRY_LANG["West_Java"]="Sundanese"

# Prompt numbers
PROMPT_NUMBERS=("inst-4" "pers-3")

# Iterate over models, countries, languages, and prompts
for model_key in "${MODEL_KEYS[@]}"; do
    for country in "${!COUNTRY_LANG[@]}"; do
        language="${COUNTRY_LANG[$country]}"
        for prompt_no in "${PROMPT_NUMBERS[@]}"; do
            python evaluate.py --model "$model_key" \
                                --language "$language" \
                                --country "$country" \
                                --prompt_no "$prompt_no" \
                                --id_col ID \
                                --question_col Translation \
                                --response_col response \
                                --annotation_filename "${country}_data.json" \
                                --annotations_key "annotations" \
                                --evaluation_result_file "evaluation_results.csv"
            if [ "$language" != "English" ]; then
                python evaluate.py --model "$model_key" \
                                    --language "English" \
                                    --country "$country" \
                                    --prompt_no "$prompt_no" \
                                    --id_col ID \
                                    --question_col Translation \
                                    --response_col response \
                                    --annotation_filename "${country}_data.json"  \
                                    --annotations_key "annotations" \
                                    --evaluation_result_file "evaluation_results.csv"
            fi
        done
    done
done




