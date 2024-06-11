import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils import *

import matplotlib.pyplot as plt
import numpy as np

COUNTRY_ISO = {
    "UK": "GB", 
    "US": "US", 
    "South_Korea": "KR",
    "Algeria": "DZ",
    "China": "CN",
    "Indonesia": "ID",
    "Spain": "ES",
    "Iran": "IR",
    "Mexico":"MX",
    "Assam":"AS",
    "Greece":"GR",
    "Ethiopia":"ET",
    "Northern_Nigeria":"NG",
    "Azerbaijan":"AZ",
    "North_Korea":"KP",
    "West_Java":"JB"
}

LANG_CODE = {
    'English':'en',
    'Chinese':'zh',
    'Spanish':'es',
    'Indonesian':'id',
    'Greek':'el',
    'Sundanese':'su',
    'Azerbaijani':'az',
    'Korean':'ko',
    'Arabic':'ar',
    'Persian':'fa',
    'Assamese':'as',
    'Amharic':'am',
    'Hausa':'ha',
}

def get_questions(
    filename=None,
    data_dir=None,
    country=None,
    template='{country}_final_questions.csv'
):
    
    if filename == None:
        filename = template.replace('{country}',country.replace(' ','_'))
        
    if data_dir == None:
        assert 'ERROR: No data directory given'
        
    df = pd.read_csv(os.path.join(data_dir,filename),encoding='utf-8')
    
    return df

def get_annotations(
    filename=None,
    data_dir=None,
    country=None,
    template='{country}_data_aggregated.json'
    ):
    
    if filename == None:
        filename = template.replace('{country}',country.replace(' ','_'))
        
    if data_dir == None:
        assert 'ERROR: No data directory given'
    
    with open(os.path.join(data_dir,filename),'r') as f:
        country_data = json.load(f)
        
    return country_data

def get_model_response_file(
    filename=None,
    data_dir=None,
    model=None,
    country=None,
    language=None,
    prompt_no=None,
    template='{model}-{country}_{language}_{prompt_no}_result.csv'
    ):
    
    if filename == None:
        filename = template.replace('{model}',model).replace('{country}',country.replace(' ','_')).replace('{language}',language).replace('{prompt_no}',prompt_no)
        print(filename)
    if data_dir == None:
        assert 'ERROR: No data directory given' 
        
    model_res_df = pd.read_csv(os.path.join(data_dir,filename),encoding='utf-8')
    
    return model_res_df

def delete_prompt_from_answer(text,prompt):
    """
    The function `delete_prompt_from_answer` aims to remove 'Answer:' part from the LLM response if there is any.
    
    :param text: LLM response
    :return: LLM response with 'Answer:' part removed
    """
    
    # Regular expression to find a word followed by a colon, capturing the word before the last colon
    text =  text.replace(prompt,'').replace('：',':').replace('、',',').replace('，',',').replace('。','.').lower()
    prompt = prompt.replace('：',':').replace('、',',').replace('，',',').replace('。','.').lower()
    
    match = re.findall(r'^(\w+:)\s', text)
    extracted = ''
    for m in match:
        if len(m) > len(extracted) and m.replace(':','') in prompt:
            extracted = m
    
    if match:
        return text.replace(extracted,'').strip()  # Return the captured word
    else:
        return text.strip()  # Return an empty string if no pattern is found

def get_llm_response_by_id(res_df,qid,id_col,r_col):
    
    if qid not in set(res_df[id_col]):
        print(qid,'not in LLM response df')
        return None
    
    try:
        llm_response = res_df[res_df[id_col]==qid][r_col].values[-1]
        prompt = res_df[res_df[id_col]==qid]['prompt'].values[-1]

        llm_response = delete_prompt_from_answer(llm_response,prompt)
        llm_response = llm_response.strip('.').lower()

    except:
        print(res_df[res_df[id_col]==qid])
        llm_response = None
    return llm_response 

def get_nested_json_str(response):
    """Extract json object from LLM response

    Args:
        response (str): LLM response with JSON format included

    Returns:
        dict: Extracted json (dict) object
    """
    
    try:
        response = response.replace('\n','')
        if "{" not in response:
            print(response)
            return response
        
        response = response.replace('```json','').replace('`','').replace(',}','}')
        
        jsons = re.findall(r'{.+}',response)

        response = jsons[-1]
        json_object = json.loads(response)
    except:
        return response 


    return json_object

