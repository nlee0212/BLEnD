from evaluation_utils import *
from multiple_choice_generation import *

def get_model_mc_response(model_name,model_cache_dir,mc_dir,questions_file,response_file=None,temperature=1,top_p=0,gpt_azure=True):
    if response_file == None:
        response_file = f"{model_name}-mc_res.csv"
    
    questions_df = pd.read_csv(os.path.join(mc_dir,questions_file),encoding='utf-8')
    already = None
    if not os.path.exists(os.path.join(mc_dir,response_file)):
        write_csv_row(list(questions_df.columns)+['full_res','final_ans'],os.path.join(mc_dir,response_file))
    else:
        already = pd.read_csv(os.path.join(mc_dir,response_file),encoding='utf-8')
    
    tokenizer,model = get_tokenizer_model(model_name,MODEL_PATHS[model_name],model_cache_dir)

    pb = tqdm(questions_df.iterrows(),total=len(questions_df))
    right = 0
    for i,row in pb:
        
        qid = row['MCQID']
        pb.set_description(qid)
        
        if isinstance(already,pd.DataFrame):
            if qid in set(already['MCQID']):
                continue
        
        country = row['country']
        
        prompt = row['prompt']
        print(prompt)
        full_res = get_model_response(model_name,prompt,model,tokenizer,temperature,top_p,gpt_azure)
        print(full_res)
        json_res = get_json_str(full_res)
        
        if isinstance(json_res,dict) and 'answer_choice' in json_res:
            try:
                final_ans = re.findall(r'[A-Z]',str(json_res['answer_choice']))[0]
                if final_ans+'.' not in prompt:
                    for k,v in json.loads(row['choices']).items():
                        if v == json_res['answer_choice']:
                            final_ans = str(k)
                            break
                    else:
                        final_ans = full_res 
                
            except:
                for k,v in json.loads(row['choices']).items():
                    if v == json_res['answer_choice']:
                        final_ans = str(k)
                        break
                else:
                    final_ans = full_res
        else:
            try:
                final_ans = re.findall(r'[A-Z]',json_res)[0]
            except:
                final_ans = full_res
        
        write_csv_row(list(row)+[full_res,final_ans],os.path.join(mc_dir,response_file))
        if final_ans == row['answer_idx']:
            right += 1
        pb.set_postfix({'score':right/(i+1)})

def multiple_choice_score(model,mc_dir,mrf,mc_res_file,eval_res_file,wrong_country_ratio_file,country):
    
    df = pd.read_csv(os.path.join(mc_dir,mrf),encoding='utf-8')
    df = df[df['country'] == country]
    
    scores = []
    
    for i,row in tqdm(df.iterrows(),total=len(df)):
        if str(row['answer_idx']) == str(row['final_ans']):
            scores.append(1)
        else:
            scores.append(0)
            
        
    df['score'] = scores
    final_score = df['score'].mean()
    
    return final_score
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose your model(s) & language(s)')
    
    parser.add_argument('--model',type=str,
                        help='Provide the model you want to use. Check and choose from the key values of the MODEL_PATHS variable. If you want to test on multiple models, provide multiple model names with ", " between each (e.g., "gpt-4-0125-preview, aya-101").')
    parser.add_argument('--model_cache_dir',type=str,default='.cache',
                    help='Provide the directory saving model caches.')
    
    parser.add_argument('--mc_dir',type=str,default='./mc_data',
                        help='Provide the directory for the data files from the human annotators.')
    parser.add_argument('--questions_file',type=str,default='mc_questions_file.csv',
                        help='Provide the directory for the data files from the human annotators.')
    parser.add_argument('--response_file',type=str,default=None,
                        help='Provide the filename to save LLM responses.')
    
    parser.add_argument('--temperature',type=int,default=0,
                    help='Provide generation temperature for LLMs.')
    parser.add_argument('--top_p',type=float,default=1,
                    help='Provide generation top_p for LLMs.')
    
    parser.add_argument("--gpt_azure", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Whether you are using the AzureOpenAI for GPT-models' response generation.")
    
    args = parser.parse_args()
    
    get_model_mc_response(model_name=args.model,
                          model_cache_dir=args.model_cache_dir,
                          mc_dir=args.mc_dir,
                          questions_file=args.questions_file,
                          response_file=args.response_file,
                          temperature=args.temperature,
                          top_p=args.top_p,
                          gpt_azure=args.gpt_azure)