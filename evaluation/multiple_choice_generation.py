from evaluation_utils import *

from itertools import combinations

def no_common_word(s1,s2):
    if is_float(s1) and is_float(s2):
        if float(s1)  == float(s2):
            return False
        else:
            return True
        
    if s1 in s2 or s2 in s1:
        return False
        
    def split_words(a):
        if '/' in a:
            a = a.split('/')
        else:
            a = [a]
        tmp_a_list = []
        for tmp_a in a:
            tmp_a_list += tmp_a.split()
        return tmp_a_list

    s1 = split_words(s1)
    s2 = split_words(s2)

    if set(s1) & set(s2):
        return False
    else:
        return True
    
def another_similar_term(question,answers,word,country1,country2):
    if isinstance(answers,str):
        answers = [answers]
    simple_flag = False
    all_floatortimeordate = True
    for c in answers:
        if is_float(c) and is_float(word):
            if float(c) == float(word):
                simple_flag = True
                break
        elif (is_date_format(c) and is_date_format(word)) or (is_time_format(c) and is_time_format(word)):
            if c in word or word in c:
                simple_flag=True
                break
        else:
            all_floatortimeordate = False
            
    if simple_flag:
        return True
    
    if all_floatortimeordate:
        return False

    prompt = """Determine if a 'target' word is the same in meaning(e.g., football & soccer or soccer & football) to at least one of the 'answer' words, or one is a subset to another(e.g., fruit & apple or apple & fruit). If so, the 'result' for 'target' word is 'O'. However, if the two simply falls into the same level of hierarchy, the 'result' is 'X' (banana & apple, rose & carnation). 
    
Note that the 'answer' list is from 'answer_country,' and the 'target' word is from 'target_country,' as written by a person.

Write down your reasoning first. Do not write any other JSON formatted object in your answer except for the result JSON object, formatted as {"result":"O"} or {"result":"X"}.

"""
    
    json_dict = {'answer':answers,'answer_country':country1,'target':word,'target_country':country2}
    json_str = json.dumps(json_dict)
    print(json_str)
    prompt += json_str
    prompt += '\n\nReasoning:'

    res = inference_azure(prompt,model_name=MODEL_PATHS['gpt-4-1106-preview'])
    res = res.replace('{result:','{"result":')
    print(res)
    json_res = get_json_str(res)
    if type(json_res) == dict  and 'result' in json_res:
        if json_res['result'] == 'O':
            return  True
        else:
            return False
    return True

def filter_mc_questions(original_questions_df,en_annotations,en_annotation_key,mc_dir):
    filtered_questions_df = original_questions_df.copy()
    
    for i,row in original_questions_df.iterrows():
        qid = row['ID']
        
        has_idk = False
        small_max_vote = False
        
        for country in en_annotations.keys():
            country_annotation = en_annotations[country]
            if qid in country_annotation:
                country_annotation_qid = country_annotation[qid]    
                if ('not-applicable' in country_annotation_qid['idks'] and country_annotation_qid['idks']['not-applicable']>0) or sum(country_annotation_qid['idks'].values()) > 2:
                    print('idks:',country_annotation_qid['idks'])
                    has_idk = True
                      
                
                elif country_annotation_qid['aggregated_answers'] and country_annotation_qid['aggregated_answers'][0][en_annotation_key] and country_annotation_qid['aggregated_answers'][0]['count'] < 2:
                    small_max_vote = True
        
            if has_idk or small_max_vote:
                filtered_questions_df = filtered_questions_df.drop(i)
                print(qid,country,has_idk,small_max_vote)
                break
    
    print('Leftover questions:',len(filtered_questions_df))
    filtered_questions_df.to_csv(os.path.join(mc_dir,'filtered_questions.csv'),index=False,encoding='utf-8')
    return filtered_questions_df

def generate_answer_choices(country_list,annotation_data_dir,annotation_data_template,question_dir,question_data_template,id_col,question_col,en_annotation_key,mc_dir,output_filename='unique_answer_choice.json'):
    country_unique_answer_choice = dict()
    
    if os.path.exists(os.path.join(mc_dir,output_filename)):
        with open(os.path.join(mc_dir,output_filename),'r') as f:
            country_unique_answer_choice = json.load(f)
            
    final_questions = get_questions(data_dir=question_dir,country=country_list[0],template=question_data_template)
    english_annotations = {country:get_annotations(data_dir=annotation_data_dir,country=country,template=annotation_data_template) for country in country_list}
    
    filtered_questions = filter_mc_questions(final_questions,english_annotations,en_annotation_key,mc_dir)
    same_dict = defaultdict(dict)
    if os.path.exists(os.path.join(mc_dir,'dictionary.json')):
        with open(os.path.join(mc_dir,'dictionary.json'),'r') as f:
            _same_dict = json.load(f)
            for k,v in _same_dict.items():
                same_dict[k] = v
        

    for i,row in tqdm(filtered_questions.iterrows(),total=len(filtered_questions)):
        qid = row[id_col]
        
        if qid in country_unique_answer_choice and country_list[-1] in country_unique_answer_choice[qid]['annotations']:
            continue
        
        print(row[question_col])

        each_qid_dict = dict()
        each_qid_dict['question'] = row[question_col]
        each_qid_dict['annotations'] = dict()
        for country in country_list:
            each_country_dict = dict()
            annotations = {data[en_annotation_key][0]:data['count'] for data in english_annotations[country][qid]['aggregated_answers'] if len(data[en_annotation_key]) > 0}
            print(annotations)
            blocked = set()
            if annotations:
                max_vote = max(list(annotations.values()))
                if 'HH:MM' in row[question_col]:
                    each_country_dict['answer'] = [k for k,v in annotations.items() if v == max_vote and is_time_format(k)]
                    tmp = {k:v for k,v in annotations.items() if is_time_format(k)}
                    annotations = tmp
                elif 'MM/DD' in row[question_col]:
                    each_country_dict['answer'] = [k for k,v in annotations.items() if v == max_vote and is_date_format(k)]
                    tmp = {k:v for k,v in annotations.items() if is_date_format(k)}
                    annotations = tmp
                elif 'Arabic' in row[question_col]:
                    each_country_dict['answer'] = [k for k,v in annotations.items() if v == max_vote and is_float(k)]
                    tmp = {k:v for k,v in annotations.items() if is_float(k)}
                    annotations = tmp
                else:
                    each_country_dict['answer'] = [k for k,v in annotations.items() if v == max_vote]
                choices = dict()
                
                other_countries_annotations = {
                    other_country: {data[en_annotation_key][0]:data['count'] for data in english_annotations[other_country][qid]['aggregated_answers'] if len(data[en_annotation_key]) > 0}
                    for other_country in country_list if other_country != country
                }
                print(other_countries_annotations)
                all_answer_choices = sorted(
                    [(vote_count, answer, other_country)
                     for other_country, other_annotations in other_countries_annotations.items()
                     for answer, vote_count in other_annotations.items()
                     if vote_count >= 2],
                    key=lambda x: x[0], reverse=True
                )
                
                print(all_answer_choices)

                for vote_count, answer, other_country in all_answer_choices:
                    
                    if other_country in choices:
                        continue
                    
                    if answer in blocked:
                        continue
                    
                    if 'HH:MM' in row[question_col] and not is_time_format(answer):
                        continue
                    elif 'MM/DD' in row[question_col] and not is_date_format(answer):
                        continue
                    elif 'Arabic' in row[question_col] and not is_float(answer):
                        continue
                    
                    flag = True
                    for candidate in annotations.keys():
                        
                        if candidate in same_dict and answer in same_dict[candidate] and same_dict[candidate][answer]:
                            flag = False
                            break
                        flag = no_common_word(answer,candidate)
                        
                        if not flag:
                            same_dict[candidate][answer] = not flag
                            same_dict[answer][candidate] = not flag
                            blocked.add(answer)

                            break
                    if flag:
                        final_flag = True
                        for k,c in choices.items():
                            if c in same_dict and answer in same_dict[c]:
                                if same_dict[c][answer]:
                                    blocked.add(answer)
                                    final_flag=False
                                    break
                                else:
                                    continue
                            
                            if not no_common_word(c,answer) or another_similar_term(each_qid_dict['question'],c,answer,k,other_country):
                                final_flag=False
                                same_dict[answer][c] = True
                                same_dict[c][answer] = True
                                blocked.add(answer)
                            else:
                                same_dict[answer][c] = False
                                same_dict[c][answer] = False
                                
                            if not final_flag:
                                break
                            
                        if final_flag:
                            all_checked = True
                            at_least_one = False
                            for candidate in annotations.keys():
                                if not (candidate in same_dict and answer in same_dict[candidate]):
                                    all_checked = False
                                if candidate in same_dict and answer in same_dict[candidate] and same_dict[candidate][answer]:
                                    at_least_one = True
                                    
                                if not all_checked or at_least_one:
                                    break

                            print('all_checked',all_checked)
                            print('at_least_one',at_least_one)
                            if answer in same_dict:
                                print(same_dict[answer])
                            
                            if at_least_one:
                                blocked.add(answer)
                                continue
                            elif all_checked or not another_similar_term(each_qid_dict['question'],list(annotations.keys()),answer,country,other_country):
                                choices[other_country] = answer
                                if not all_checked:
                                    for candidate in annotations.keys():
                                        same_dict[candidate][answer] = False
                                        same_dict[answer][candidate] = False
                                
                            else:
                                blocked.add(answer)
                each_country_dict['choices'] = choices
                
                with open(os.path.join(mc_dir,'dictionary.json'),'w') as f:
                    json.dump(same_dict,f,indent=4,ensure_ascii=False)
                
                
                each_qid_dict[country] = each_country_dict

            each_qid_dict['annotations'][country] = annotations
            print(each_qid_dict)
        country_unique_answer_choice[qid] = each_qid_dict
        
        with open(os.path.join(mc_dir,output_filename),'w') as f:
            json.dump(country_unique_answer_choice,f,indent=4,ensure_ascii=False)

    
            
def generate_prompt_mc(question,country,answers,choices,min_choice,dummy_choices):
    res = []
    
    for answer in answers:
        if country in ['US','UK']:
            prompt = question.replace('your country',f'the {country}')
        else:
            prompt = question.replace('your country',country.replace('_',' '))
        
        prompt += ' Without any explanation, choose only one from the given alphabet choices(e.g., A, B, C). Provide as JSON format: {"answer_choice":""}\n\n'
        
        for chosen_choices in combinations(choices.items(),min_choice):
            all_choices = sorted([(v,k) for k,v in chosen_choices]+[(answer,country)]+[(dummy,'dummy') for dummy in dummy_choices])
            all_choices_idx = dict()
            all_choices_country = dict()
            answer_idx = -1
            
            this_prompt = prompt
            for i,(a,a_country) in enumerate(all_choices):
                if a == answer:
                    answer_idx = chr(ord('A')+i)
                all_choices_idx[chr(ord('A')+i)] = a
                all_choices_country[chr(ord('A')+i)] = a_country
                this_prompt += f'{chr(ord("A")+i)}. {a}\n'
            this_prompt += '\nAnswer:'

            res.append((this_prompt,all_choices_idx,all_choices_country,answer_idx))
    return res

def get_dummy_choices(question,annotations,num):
    prompt = 'Provide '+str(num)+' dummy option(s) that makes sense to be the answer(s) of the given "question", and has to exist in real-life (non-fiction), but is totally different from the given "answers" without any explanation. Make sure that the options are different from each other, and cannot be an answer from any country. Provide as JSON format: {"dummy_options":[]}\n\n'
    json_str = json.dumps({'question':question,'answers':list(set([v for k in annotations for v in annotations[k] ]))},ensure_ascii=False, indent=4)
    prompt += json_str
    
    print(prompt)
    
    while True:
        res = inference_azure(prompt,temperature=1,top_p=1,model_name=MODEL_PATHS['gpt-4-1106-preview'])
        res = res.replace('{dummy_options:','{"dummy_options":')
        json_res = get_json_str(res)
        print(json_res)
        
        if type(json_res) == dict  and 'dummy_options' in json_res and type(json_res['dummy_options']) == list and len(json_res['dummy_options']) == num and len(set(json_res['dummy_options']))==num and len(set(json_res['dummy_options'])&set([v for k in annotations for v in annotations[k]])) == 0:
            return [s.lower() for s in json_res['dummy_options']]
    

def generate_multiple_choice(country_list,mc_dir,answer_choice_file,questions_file,generate_dummy=True):
    with open(os.path.join(mc_dir,answer_choice_file),'r') as f:
        answer_choices = json.load(f)
    
    if os.path.exists(os.path.join(mc_dir,questions_file)):
        os.remove(os.path.join(mc_dir,questions_file))
        
    write_csv_row(['MCQID','ID','country','prompt','choices','choice_countries','answer_idx'],os.path.join(mc_dir,questions_file))
    
    pb = tqdm(answer_choices,total=len(answer_choices))
    for qid in pb:
        pb.set_description(qid)
        question = answer_choices[qid]['question']
        cnt = 0
        
        # check the minimum number of answer_choices[qid][country]['choices']
        min_choice = min([len(answer_choices[qid][country]['choices']) for country in country_list])
        
        dummy_choices = []
        
        if min_choice < 3:
            if generate_dummy and 'dummy_choices' not in answer_choices[qid][country]:
                dummy_choices = get_dummy_choices(question,answer_choices[qid]['annotations'],3-min_choice)
                answer_choices[qid][country]['dummy_choices'] = dummy_choices
                with open(os.path.join(mc_dir,answer_choice_file),'w') as f:
                    json.dump(answer_choices,f,indent=4,ensure_ascii=False)
            elif 'dummy_choices' in answer_choices[qid][country]:
                dummy_choices = answer_choices[qid][country]['dummy_choices']
            else:
                print('ERROR: No dummy choices for',qid,'in',country,'and min_choice:',min_choice)
                continue
        
        for country in country_list:
            pb.set_postfix({'country':country})
            prompt_questions = generate_prompt_mc(question,country,answer_choices[qid][country]['answer'],answer_choices[qid][country]['choices'],min(min_choice,3),dummy_choices)
            if prompt_questions:
                for q,choices,choice_countries,answer_idx in prompt_questions:
                    write_csv_row([f'{qid}_{cnt}',qid,country,q,json.dumps(choices,indent=4,ensure_ascii=False),json.dumps(choice_countries,indent=4,ensure_ascii=False),answer_idx],os.path.join(mc_dir,questions_file))
                    cnt += 1
                   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose your model(s) & language(s)')

    parser.add_argument('--id_col',type=str,default='ID',
                        help='Provide the column name from the final question file name with question IDs.') 
    parser.add_argument('--question_col',type=str,default='Question',
                        help='Provide the column name from the final question file name with questions.')

    parser.add_argument('--question_dir',type=str,default='../data/questions/',
                        help='Provide the directory for the output files to be saved.')
    parser.add_argument('--question_data_template',type=str,default='{country}_questions.csv',
                        help='Provide the filename template of the question data file.')
    parser.add_argument('--annotation_dir',type=str,default='../data/annotations/',
                        help='Provide the directory for the data files from the human annotators.')
    parser.add_argument('--annotation_data_template',type=str,default='{country}_data.json',)
    parser.add_argument('--mc_dir',type=str,default='./mc_data',
                        help='Provide the directory for the data files from the human annotators.')
    parser.add_argument('--answer_choice_file',type=str,default='unique_answer_choice.json',
                        help='Provide the directory for the data files from the human annotators.')
    parser.add_argument('--mc_questions_file',type=str,default='mc_questions_file.csv',
                        help='Provide the directory for the data files from the human annotators.')
    parser.add_argument('--en_annotation_key',type=str,default='en_answers',
                        help='Provide the directory for the data files from the human annotators.')
    args = parser.parse_args()
    
    if not os.path.exists(args.mc_dir):
        os.mkdir(args.mc_dir)

    country_list = list(COUNTRY_LANG.keys())
    print(country_list)
    generate_answer_choices(country_list=country_list,
                            annotation_data_dir=args.annotation_dir,
                            annotation_data_template=args.annotation_data_template,
                            question_dir=args.question_dir,
                            question_data_template=args.question_data_template,
                            id_col=args.id_col,
                            question_col=args.question_col,
                            en_annotation_key=args.en_annotation_key,
                            mc_dir=args.mc_dir,
                            output_filename=args.answer_choice_file)
    
    generate_multiple_choice(country_list=country_list,
                             mc_dir=args.mc_dir,
                             answer_choice_file=args.answer_choice_file,
                             questions_file=args.mc_questions_file)