export CUDA_VISIBLE_DEVICES="4,5,6,7"

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

python multiple_choice_generation.py --id_col 'ID' \
    --question_col 'Question' \
    --question_dir '../data/questions' \
    --question_data_template '{country}_questions.csv' \
    --annotation_dir "../data/annotations" \
    --annotation_data_template '{country}_data.json' \
    --mc_dir  "./mc_data/" \
    --answer_choice_file "./unique_answer_choice.json" \
    --mc_questions_file "./mc_questions_file.csv" \
    --en_annotation_key 'en_answers'