import os
import re
import csv
import json
import time
import argparse
import requests
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from easydict import EasyDict
from collections import defaultdict, Counter
import pathlib
import textwrap
import os.path as osp
import math

import openai
from openai import AzureOpenAI,OpenAI
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, LlamaTokenizer, pipeline, AutoConfig, BitsAndBytesConfig
from transformers.generation.utils import GenerationConfig
from peft import PeftModel, PeftConfig
import torch
import anthropic
from typing import Union
import google.generativeai as genai
from google.generativeai.types import safety_types
from google.oauth2 import service_account
import vertexai
from vertexai.language_models import TextGenerationModel
import anthropic
from anthropic import HUMAN_PROMPT, AI_PROMPT
import cohere

MODEL_PATHS = {
    "gpt-3.5-turbo-0125":"gpt-3.5-turbo-0125",
    "gpt-4-0125-preview":"gpt-4-0125-preview",
    "gpt-4-1106-preview":"gpt-4-1106-preview",
    "aya-101":"CohereForAI/aya-101",
    "gemini-pro":"gemini-pro",
    "gemini-1.5-pro":"gemini-1.5-pro-latest",
    'Orion-14B-Chat':'OrionStarAI/Orion-14B-Chat',
    "claude-3-opus-20240229":'claude-3-opus-20240229',
    "claude-3-sonnet-20240229":'claude-3-sonnet-20240229',
    "claude-3-haiku-20240307":'claude-3-haiku-20240307',
    'Qwen1.5-72B-Chat':'Qwen/Qwen1.5-72B-Chat',
    'Qwen1.5-14B-Chat':'Qwen/Qwen1.5-14B-Chat' ,
    'Qwen1.5-32B-Chat':'Qwen/Qwen1.5-32B-Chat' ,
    'text-bison-002':'text-bison@002',
    'c4ai-command-r-v01':'CohereForAI/c4ai-command-r-v01',
    'c4ai-command-r-plus':'command-r-plus',
    'Mixtral-8x7B-Instruct-v0.1':'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'aya-23':'CohereForAI/aya-23-35B',
    'SeaLLM-7B-v2.5':'SeaLLMs/SeaLLM-7B-v2.5',
    'Merak-7B-v4':'Ichsan2895/Merak-7B-v4',
    'jais-13b-chat':'core42/jais-13b-chat',
}

COUNTRY_LANG = { 
    "UK": "English", 
    "US": "English", 
    "South_Korea": "Korean",
    "Algeria": "Arabic",
    "China": "Chinese",
    "Indonesia": "Indonesian",
    "Spain": "Spanish",
    "Iran": "Persian",
    "Mexico":"Spanish",
    "Assam":"Assamese",
    "Greece":"Greek",
    "Ethiopia":"Amharic",
    "Northern_Nigeria":"Hausa",
    "Azerbaijan":"Azerbaijani",
    "North_Korea":"Korean",
    "West_Java":"Sundanese"
}


def get_tokenizer_model(model_name,model_path,model_cache_dir):
    tokenizer,model = None,None
    
    if 'gpt' not in model_name and 'gemini' not in model_name and 'claude' not in model_name and 'bison' not in model_name and 'plus' not in model_name:
        if 'llama' in model_name.lower():
            tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False,token=os.getenv("HF_TOKEN"))
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", 
                                                                torch_dtype=torch.float16,
                                                                resume_download=True,
                                                                cache_dir=os.path.join(model_cache_dir,model_path),token=os.getenv("HF_TOKEN"))
        
        elif 'Orion' in model_name or 'polylm' in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True ,torch_dtype=torch.bfloat16,
                                                                resume_download=True,
                                                                cache_dir=os.path.join(model_cache_dir,model_path))
            
        elif 'c4ai' in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",
                                                                resume_download=True,
                                                                cache_dir=os.path.join(model_cache_dir,model_path))
        
        elif 'aya' in model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if '23' in model_name:
                model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",token=os.getenv("HF_TOKEN"),
                                                                resume_download=True,
                                                                cache_dir=os.path.join(model_cache_dir,model_path))
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path, device_map="auto",
                                                                    resume_download=True,
                                                                    cache_dir=os.path.join(model_cache_dir,model_path))
            
        elif 'mala' in model_name.lower():
            base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf',token=os.getenv("HF_TOKEN"), device_map="auto",
                                                                cache_dir=os.path.join(model_cache_dir,model_path))
            base_model.resize_token_embeddings(260164)
            tokenizer = AutoTokenizer.from_pretrained(model_path,token=os.getenv("HF_TOKEN"))
            model = PeftModel.from_pretrained(base_model, model_path, device_map="auto",
                                                                cache_dir=os.path.join(model_cache_dir,model_path))
        elif 'mistral' in model_path.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,token=os.getenv("HF_TOKEN"))
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",
                                                                resume_download=True,
                                                                cache_dir=os.path.join(model_cache_dir,model_path),token=os.getenv("HF_TOKEN"))
        
        elif 'merak' in model_path.lower():
            config = AutoConfig.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path,
                                                        device_map="auto",
                                                        trust_remote_code=True,
                                                        resume_download=True,
                                                        cache_dir=os.path.join(model_cache_dir,model_path))

            tokenizer = LlamaTokenizer.from_pretrained(model_path)
        
        elif 'jais' in model_path.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                         device_map="auto", 
                                                         trust_remote_code=True,
                                                         resume_download=True,
                                                         cache_dir=os.path.join(model_cache_dir,model_path))
         
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto",
                                                                resume_download=True,
                                                                cache_dir=os.path.join(model_cache_dir,model_path))

            
    return tokenizer,model

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
    
def get_cohere_response(
    text,
    model_name='command-r-plus',
    temperature=1.0,
    top_p=1.0,
    max_tokens=None,
    greedy=False,
    num_sequence=1,
    max_try=10,
    dialogue_history=None
):
    
    co = cohere.Client(os.getenv("COHERE_API_KEY"))
    
    n_try = 0
    while True:
        if n_try == max_try:
            outputs = ["something wrong"]
            res = None
            break
        try:
            time.sleep(0.5)
            response = co.chat(
                model=model_name,
                message=text,
                temperature=temperature,
                p=top_p,
                max_tokens=max_tokens,
            )
        
            res = response.text.strip()

            break
        except KeyboardInterrupt:
            raise Exception("KeyboardInterrupted!")
        except:
            try:
                print(res.data)
            except:
                print('ERROR')
            print("Exception: Sleep for 10 sec")
            
            time.sleep(10)
            n_try += 1
            continue
            
    return res

def check_gpt_input_list(history):
    check = True
    for i, u in enumerate(history):
        if not isinstance(u, dict):
            check = False
            break
            
        if not u.get("role") or not u.get("content"):
            check = False
            break
        
    return check

def get_gpt_response(
    text,
    model_name,
    temperature=1.0,
    top_p=1.0,
    max_tokens=None,
    greedy=False,
    num_sequence=1,
    max_try=10,
    dialogue_history=None
):
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                    organization=os.getenv("OPENAI_ORG_ID"))

    if (model_name.startswith("gpt-3.5-turbo") and 'instruct' not in model_name) or model_name.startswith("gpt-4"):
        if dialogue_history:
            if not check_gpt_input_list(dialogue_history):
                raise Exception("Input format is not compatible with chatgpt api! Please see https://platform.openai.com/docs/api-reference/chat")
            messages = dialogue_history
        else:
            messages = []
        
        messages.append({'role': 'user', 'content': text})

        prompt = {
            "model": model_name,
            "messages": messages,
            "temperature": 0. if greedy else temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": num_sequence
        }

    else:    
        prompt = {
            "model": model_name,
            "prompt": text,
            "temperature": 0. if greedy else temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": num_sequence
        }
    
    n_try = 0
    while True:
        if n_try == max_try:
            outputs = ["something wrong"]
            break
        
        try:
            if (model_name.startswith("gpt-3.5-turbo") and 'instruct' not in model_name) or model_name.startswith("gpt-4"):
                time.sleep(0.5)
                res = client.chat.completions.create(**prompt)
                outputs = [o['message']['content'].strip("\n ") for o in res['choices']]
            else:
                res = client.chat.completions.create(**prompt)
                outputs = [o['text'].strip("\n ") for o in res['choices']]
            break
        except KeyboardInterrupt:
            raise Exception("KeyboardInterrupted!")
        except:
            print("Exception: Sleep for 10 sec")
            time.sleep(10)
            n_try += 1
            continue
        
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs

def inference_azure(prompt,temperature=0,top_p=1,model_name,max_attempt=10):
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VER"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPT"),
    )
    
    attempt = 0
    while attempt < max_attempt:
        time.sleep(0.5)
        completion = None
        try:
            completion = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                top_p=top_p,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            res = completion.choices[0].message.content
            if res == None:
                attempt += 1
                print(completion.choices[0].finish_reason)
            else:
                break
        except KeyboardInterrupt:
            raise Exception("KeyboardInterrupted!")
        except:
            print("Exception: Sleep for 10 sec")
            time.sleep(10)
            attempt += 1
            continue
    if attempt == max_attempt:
        if completion:
            return completion.choices[0].finish_reason
        else:
            return "openai.BadRequestError"
    return res.strip()

def inference_claude(prompt,temperature=0,top_p=1,model_name,max_attempt=10):
    c =  anthropic.Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))    
    
    attempt = 0
    while attempt < max_attempt:
        time.sleep(0.5)
        completion = None
        try:
            message = c.messages.create(
                model=model_name,
                max_tokens=1024,
                temperature=temperature,
                top_p=top_p,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )   
            res = message.content[0].text
            if res == None:
                attempt += 1
                print(message.stop_reason)
                time.sleep(10)
            else:
                break
        except KeyboardInterrupt:
            raise Exception("KeyboardInterrupted!")
        except:
            print("Exception: Sleep for 10 sec")
            time.sleep(10)
            attempt += 1
            continue
    if attempt == max_attempt:
        if message != None:
            return message.error.message
        else:
            return "UNKNOWN_ERROR"
    return res.strip()
    
def model_inference(prompt,model_path,model,tokenizer,max_length=512):
    if 'Orion' in model_path:
        model.generation_config = GenerationConfig.from_pretrained(model_path)
        messages = [{"role": "user", "content": prompt}]
        result = model.chat(tokenizer, messages,streaming=False)
        result = result.replace(prompt,'').strip()
        
    if 'mistral' in model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

        messages = messages = [{"role": "user", "content": prompt}]

        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

        outputs = model.generate(inputs, max_new_tokens=max_length)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    elif 'Qwen' in model_path:
        messages = messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_length
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
    elif 'c4ai' in model_path:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

        gen_tokens = model.generate(
            input_ids,
            max_new_tokens=max_length,
        )

        s = tokenizer.decode(gen_tokens[0])
        
        start_token = "<|CHATBOT_TOKEN|>"
        end_token = "<|END_OF_TURN_TOKEN|>"

        start_idx = s.find(start_token) + len(start_token)
        end_idx = s.find(end_token, start_idx)

        result = s[start_idx:end_idx]

    elif 'aya-23' in model_path:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        
        gen_tokens = model.generate(
            input_ids, 
            max_new_tokens=max_length, 
        )
        
        s = tokenizer.decode(gen_tokens[0])
        
        start_token = "<|CHATBOT_TOKEN|>"
        end_token = "<|END_OF_TURN_TOKEN|>"

        start_idx = s.find(start_token) + len(start_token)
        end_idx = s.find(end_token, start_idx)

        result = s[start_idx:end_idx]
    
    elif 'SeaLLM' in model_path:
        messages = [{"role": "user", "content": prompt}]
        
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)

        generated_ids = model.generate(
            input_ids, 
            max_new_tokens=max_length, 
        )
        s = tokenizer.batch_decode(generated_ids)[0]
        
        start_token = "<|im_start|>assistant\n"
        end_token = "<eos>"

        start_idx = s.find(start_token) + len(start_token)
        end_idx = s.find(end_token, start_idx)

        result = s[start_idx:end_idx]
    
    elif 'Merak' in model_path:
        messages = [{"role": "user", "content": prompt}] 
        inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).to(model.device)
        inputs = tokenizer(inputs, return_tensors="pt", return_attention_mask=True)
        
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"),
                            attention_mask=inputs.attention_mask,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.eos_token_id,
                            max_new_tokens=max_length)
            response = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

            assistant_start = f'''{prompt} \n assistant\n '''
            response_start = response.find(assistant_start)
            result = response[response_start + len(assistant_start) :].strip()

    elif 'jais' in model_path:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        inputs = input_ids.to(model.device)
        input_len = inputs.shape[-1]
        generate_ids = model.generate(
            inputs,
            max_length=max_length,
        )
        result = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
    
    else:
        input_ids = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)
        outputs = model.generate(**input_ids,max_length=max_length)
        result = tokenizer.decode(outputs[0],skip_special_tokens=True)

        
    return result

def get_gemini_response(prompt,model_name,
    temperature=0,
    top_p=1.0,
    greedy=False,
    max_attempt=10,):
    
    GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    
    safety_settings=[
        {
            "category": category,
            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
        } for category in safety_types._NEW_HARM_CATEGORIES 
    ]
    
    generation_config = genai.types.GenerationConfig(temperature=temperature,top_p=top_p)
    model = genai.GenerativeModel(model_name,safety_settings)
    
    attempt = 0
    while attempt < max_attempt:
        time.sleep(0.5)
        response = model.generate_content(prompt,generation_config=generation_config)
        try:
            response = model.generate_content(prompt,generation_config=generation_config)
            res = response.text
            break
        except ValueError:
            # If the response doesn't contain text, check if the prompt was blocked.
            print(response.prompt_feedback)
            try:
                # Also check the finish reason to see if the response was blocked.
                print(response.candidates[0].finish_reason)
                # If the finish reason was SAFETY, the safety ratings have more details.
                print(response.candidates[0].safety_ratings)
            except:
                print()
            time.sleep(10)
            attempt += 1
            continue
        except KeyboardInterrupt:
            raise Exception("KeyboardInterrupted!")
        except:
            if '1.5' in model_name:
                print("Exception: Sleep for 70 sec")
                time.sleep(70)
            else:
                print("Exception: Sleep for 10 sec")
                time.sleep(10)
            attempt += 1
            continue
    if attempt == max_attempt:
        if response:
            try:
                return response.candidates[0].finish_reason
            except:
                return response.prompt_feedback
        else:
            return ""
    return res.strip() 

def get_palm_response(prompt,model_name,
    temperature=1.0,
    top_p=1.0,
    greedy=False,
    max_attempt=10,):
    
    GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    
    safety_settings=[
        {
            "category": category,
            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
        } for category in safety_types.HarmCategory if category.value <  7
    ]
    
    attempt = 0
    while attempt < max_attempt:
        time.sleep(0.5)
        try:
            completion = genai.generate_text(
                model=model_name,
                prompt=prompt,
                temperature=temperature,
                safety_settings=safety_settings,
                top_p=top_p
            )
            
            res = completion.result
            if res == None:
                attempt += 1
                print(completion.filters)
                print(completion.safety_feedback)
                continue
            break
        except ValueError:
            # If the response doesn't contain text, check if the prompt was blocked.
            print(completion.filters)
            # Also check the finish reason to see if the response was blocked.
            print(completion.safety_feedback)

            attempt += 1
            continue
        except KeyboardInterrupt:
            raise Exception("KeyboardInterrupted!")
        except:
            print("Exception: Sleep for 10 sec")
            time.sleep(10)
            attempt += 1
            continue
    if attempt == max_attempt:
        return completion.filters
    return res.strip()

def get_palm2_response(prompt,model_name,
    temperature=1.0,
    top_p=1.0,
    greedy=False,
    max_attempt=10,):
    credentials = service_account.Credentials.from_service_account_file(os.getenv('GOOGLE_APPLICATION_CREDENTIALS'))
    vertexai.init(project=os.getenv('GOOGLE_PROJECT_NAME'),credentials=credentials)
    
    GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
    
    safety_settings=[
        {
            "category": category,
            "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
        } for category in safety_types.HarmCategory if category.value <  7
    ]
    model = TextGenerationModel.from_pretrained(model_name)
    parameters = {
        "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        "top_p": top_p,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "max_output_tokens": 512
    }
    
    attempt = 0
    while attempt < max_attempt:
        time.sleep(0.5)
        try:
            response = model.predict(
                prompt,
                **parameters,
            )
            
            res = response.text

            if res == None:
                attempt += 1
                print(response.is_blocked)
                print(response.safety_attributes)
                continue
            break
        except ValueError:
            print(response.is_blocked)
            print(response.safety_attributes)

            attempt += 1
            continue
        except KeyboardInterrupt:
            raise Exception("KeyboardInterrupted!")
        except:
            print("Exception: Sleep for 10 sec")
            time.sleep(10)
            attempt += 1
            continue
    if attempt == max_attempt:
        return response.safety_attributes
    return res.strip()  

def get_model_response(model_name,prompt,model,tokenizer,temperature,top_p,gpt_azure):
    model_path = MODEL_PATHS[model_name]
    
    if gpt_azure:
        gpt_inference = inference_azure
    else:
        gpt_inference = get_gpt_response
    
    if 'gpt' in model_name:
        response = gpt_inference(prompt,model_name=model_path,temperature=temperature,top_p=top_p)
    elif 'gemini' in model_name:
        response = get_gemini_response(prompt,model_name=model_path,temperature=temperature,top_p=top_p)
    elif 'bison' in model_name:
        response = get_palm2_response(prompt,model_name=model_path,temperature=temperature,top_p=top_p)
    elif 'claude' in model_name:
        response = inference_claude(prompt,model_name=model_path,temperature=temperature,top_p=top_p)
    elif 'plus' in model_name:
        response = get_cohere_response(prompt,model_name=model_path,temperature=temperature,top_p=top_p)
    else:
        response = model_inference(prompt,model_path=model_path,model=model,tokenizer=tokenizer)
            
    return response

def get_json_str(response,return_list=False):
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
        
        if return_list:
            jsons = re.findall(r'\[\s*{.+}\s*\]',response)
            json_list = []
            json_object = json.loads(jsons[-1])
        else:
            jsons = re.findall(r'{[^}]+}',response)

            response = jsons[0]
            response = response.replace('```json','').replace('`','').replace('\n','').replace(',}','}')
            json_object = json.loads(response)
    except:
        return response 

    return json_object

def import_google_sheet(id,gid=0,file_path='google_sheet_tmp.csv'):

    url = f'https://docs.google.com/spreadsheets/d/{id}/export?format=csv&gid={gid}'
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print('CSV file saved to: {}'.format(file_path))    
    else:
        print(f'Error downloading Google Sheet: {response.status_code}')
        sys.exit(1)

    df = pd.read_csv(file_path)
    return df

def read_jsonl(filename):
    js = []
    with open(filename) as f: 
        for line in f.readlines():
            js.append(json.loads(line)) 
    
    return js

def write_csv_row(values,filename):
    open_trial = 0
    
    while True:
        if open_trial > 10:
            raise Exception("something wrong")

        try:
            with open(filename, "a", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(values)
            break
        except:
            print("open failed")
            continue

def replace_country_name(s,country):
    return s.replace('your country',country)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('True','yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('False','no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def is_time_format(s):
    """
    Check if the given string matches the '%d%d:%d%d' time format.
    
    Args:
    s (str): The string to check.

    Returns:
    bool: True if the string matches the format, False otherwise.
    """
    # Regular expression to match exactly two digits, a colon, and then exactly two more digits
    pattern = r"^\d\d:\d\d$"
    return bool(re.match(pattern, s))

def is_date_format(s):
    """
    Check if the given string matches the '%d%d/%d%d' time format.
    
    Args:
    s (str): The string to check.

    Returns:
    bool: True if the string matches the format, False otherwise.
    """
    # Regular expression to match exactly two digits, a colon, and then exactly two more digits
    pattern = r"^\d{1,2}/\d{1,2}$"
    return bool(re.match(pattern, s))

def is_float(s):
    """
    Check if the given string can be converted to a float.
    
    Args:
    s (str): The string to check.

    Returns:
    bool: True if the string can be converted to a float, False otherwise.
    """
    try:
        float(s)  # Try converting the string to a float
        return True
    except ValueError:  # If conversion to float fails, it raises ValueError
        return False