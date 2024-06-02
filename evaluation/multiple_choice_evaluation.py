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

def multiple_choice_score(model,mc_dir,mrf,mc_res_file,eval_res_file,wrong_country_ratio_file,country_list):
    
    with open(os.path.join(mc_dir,'dictionary.json'),'r') as f:
        dictionary = json.load(f)
        
    with open(os.path.join(mc_dir,'unique_answer_choice.json'),'r') as f:
        unique_answer_choice = json.load(f)
    
    print(mrf)
    df = pd.read_csv(os.path.join(mc_dir,mrf),encoding='utf-8')
    
    scores = []
    chosen_country = defaultdict(list)
    country_wrong = dict()
    chosen_country_cnt = dict()
    wrong_country_ratio = dict()
    
    
    for i,row in tqdm(df.iterrows(),total=len(df)):
        if str(row['answer_idx']) == str(row['final_ans']):
            scores.append(1)
        else:
            scores.append(0)
            country_wrong[row['country']] = country_wrong.get(row['country'],0)+1

            qid = row['ID']
            choices = json.loads(row['choices'])
            choice_countries = json.loads(row['choice_countries'])
            if row['final_ans'] in choices:
                if choice_countries[row['final_ans']] == 'dummy':
                    chosen_country[row['country']].append('dummy')
                else:
                    ans = choices[row['final_ans']]
                    for country in country_list:
                        flag = True
                        for annot in unique_answer_choice[qid]['annotations'][country]:
                            if annot in dictionary and ans in dictionary and annot in dictionary[ans] and ans in dictionary[annot] and dictionary[annot][ans]:
                                flag = False
                                break
                            
                            flag = no_common_word(ans,annot)
                            
                            if not flag:
                                dictionary[annot][ans] = True
                                dictionary[ans][annot] = True
                                break
                            
                        if not flag:
                            chosen_country[row['country']].append(country)
                        
                        else:
                            if another_similar_term(unique_answer_choice[qid]['question'],ans,annot,choice_countries[row['final_ans']],country):
                                dictionary[ans][annot] = True
                                dictionary[annot][ans] = True
                                chosen_country[row['country']].append(country)
                                
                            else:
                                dictionary[ans][annot] = False
                                dictionary[annot][ans] = False
    
                        with open(os.path.join(mc_dir,'dictionary.json'),'w') as f:
                            json.dump(dictionary,f,ensure_ascii=False)
        
        
    df['score'] = scores
    score_df = df.groupby(['ID','country']).mean('score')
    total_score_df = score_df.groupby(['country']).mean('score')
    
    for country in country_list:
        write_csv_row([model,country,'English',None,'MC',float(total_score_df[country]['score'])],eval_res_file)
        
    for c in country_list:
        print(c)
        print(Counter(chosen_country[c]))
        chosen_country_cnt[c] = dict(Counter(chosen_country[c]))
        
    for c in chosen_country:
        wrong_country_ratio[c] = {k:chosen_country_cnt[c][k]/country_wrong[c] for k in chosen_country[c]}

        print(c)
        print(wrong_country_ratio[c])
        
    with open(wrong_country_ratio_file,'w') as f:
        json.dump(wrong_country_ratio,f,ensure_ascii=False,indent=4)
    
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