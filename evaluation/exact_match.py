from evaluation_utils import *

import unicodedata as ud

# pip install konlpy
from konlpy.tag import Okt

# pip install hausastemmer
import hausastemmer

# git clone https://github.com/aznlp-disc/stemmer.git, cp word.txt & suffix.txt.
from stemmer.stemmer import Stemmer as AZStemmer
from string import punctuation

# pip install nlp-id
from nlp_id.lemmatizer import Lemmatizer as IDLemmatizer

# pip install hazm
from hazm import Lemmatizer as PRLemmatizer

# pip install qalsadi
from qalsadi.lemmatizer import Lemmatizer as ARLeammatizer

# pip install cltk
from cltk import NLP

# !pip install spark-nlp==5.3.3 pyspark==3.3.1
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp

from SUSTEM.SUSTEM_S import *

import spacy

# pip install jieba
import jieba

# git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git & https://github.com/anoopkunchukuttan/indic_nlp_resources.git
# The path to the local git repo for Indic NLP library
INDIC_NLP_LIB_HOME=os.path.abspath("./indic_nlp_library")

# The path to the local git repo for Indic NLP Resources
INDIC_NLP_RESOURCES=os.path.abspath("./indic_nlp_resources")

sys.path.append(INDIC_NLP_LIB_HOME)
from indicnlp import common
from indicnlp import loader
from indicnlp.tokenize import indic_tokenize  



def lemma_check(answer,llm_response,nlp_pipeline,language='Korean'):
    if answer in llm_response or answer.replace('-',' ') in llm_response or answer.replace(' ','-') in llm_response:
        return True
    
    if language == 'Korean':
        okt = Okt()
        answer_tokens = okt.morphs(' '.join([w for w,p in okt.pos(answer) if p!='Josa']),stem=True)
        llm_tokens = okt.morphs(' '.join([w for w,p in okt.pos(llm_response) if p!='Josa']),stem=True)
        
    elif language == 'Hausa':
        answer_tokens = [hausastemmer.stem(term.strip('-')) for term in answer.split()]
        llm_tokens = [hausastemmer.stem(term.strip('-')) for term in llm_response.split()]
    
    elif language == 'Amharic':
        answer_tokens = [token.result if lemma.result.startswith('_') else lemma.result for token,lemma in zip(nlp_pipeline.fullAnnotate(answer)[0]['lemma'],nlp_pipeline.fullAnnotate(answer)[0]['token'])]
        llm_tokens = [token.result if lemma.result.startswith('_') else lemma.result for token,lemma in zip(nlp_pipeline.fullAnnotate(llm_response)[0]['lemma'],nlp_pipeline.fullAnnotate(llm_response)[0]['token'])]
        
    elif language == 'Azerbaijani':
        # Instantiate Stemmer object
        my_stemmer = AZStemmer()
        
        def stem_words(my_text):
            my_text=my_text.replace("İ", "I")
            my_text=my_text.replace("“", "")
            my_text=my_text.replace("”", "")
            my_text=my_text.replace("'", "")
            my_text=my_text.replace('"', "")
            my_text=my_text.split()
            my_words=[]
            for word in my_text:
                my_words.append(''.join(c for c in word if (c not in punctuation) or (c == '-')))
            # Apply stemming to the list of words
            my_words = my_stemmer.stem_words(my_words)
            # Print words after stemming
            return my_words
        
        answer_tokens = stem_words(answer)
        llm_tokens = stem_words(llm_response)
    
    elif language == 'Indonesian':
        lemmatizer = IDLemmatizer() 
        answer_tokens = lemmatizer.lemmatize(answer).split()
        llm_tokens = lemmatizer.lemmatize(llm_response).split() 
    
    elif language == 'Persian':
        lemmatizer = PRLemmatizer()
        answer_tokens = [lemmatizer.lemmatize(term) for term in answer.split()]
        llm_tokens = [lemmatizer.lemmatize(term) for term in llm_response.split()]
        
    elif language == 'Arabic':
        lemmatizer = ARLeammatizer()
        answer_tokens = lemmatizer.lemmatize(answer)
        llm_tokens = lemmatizer.lemmatize(llm_response) 
        
    elif language == 'Greek':
        cltk_nlp = NLP(language="grc", suppress_banner=True)
        answer_tokens = cltk_nlp.analyze(text=answer).lemmata
        llm_tokens = cltk_nlp.analyze(text=llm_response).lemmata
        
    elif language == 'Spanish':
        answer_tokens = [lemma.result for lemma in nlp_pipeline.fullAnnotate(answer)[0]['lemma']]
        llm_tokens = [lemma.result for lemma in nlp_pipeline.fullAnnotate(llm_response)[0]['lemma']]
        
    elif language == 'Sundanese':
        stemmer = EcsStemmer()
        answer_tokens = [stemmer.stemmingProcess(word.replace('(','').replace(')','')) for word in answer.split()]
        llm_tokens = [stemmer.stemmingProcess(word.replace('(','').replace(')','')) for word in llm_response.split()]

        
    elif language == 'English':
        answer_tokens = [token.lemma_ for token in nlp_pipeline(answer)]
        llm_tokens = [token.lemma_ for token in nlp_pipeline(llm_response)]
        
    elif language == 'Chinese':
        answer_tokens = list(jieba.cut(answer))
        llm_tokens = list(jieba.cut(llm_response))
        
    elif language == 'Assamese':
        common.set_resources_path(INDIC_NLP_RESOURCES)
        loader.load()
        
        answer_tokens = indic_tokenize.trivial_tokenize(answer)
        llm_tokens = indic_tokenize.trivial_tokenize(llm_response)
        
    d = {ord('\N{COMBINING ACUTE ACCENT}'):None}
    
    answer_tokens = [ud.normalize('NFD',term).translate(d).lower() for term in answer_tokens if term not in punctuation and term != '']
    llm_tokens = [ud.normalize('NFD',term).translate(d).lower() for term in llm_tokens if term not in punctuation and term != '']
    
    for a in answer_tokens:
        if a not in llm_tokens:
            return False        
    
    return True

def hard_exact_match(annotation_dict,response_df,id_col,r_col,annotations_key='annotations'):
    binary_score = 0
    weight_score = 0
    
    for qid,data in annotation_dict.items():
        llm_response = get_llm_response_by_id(response_df,qid,id_col,r_col)
        
        if llm_response and data[annotations_key]:
            max_vote = max(list(data[annotations_key].values()))
        
            for k,v in sorted(data[annotations_key].items(), key=lambda item: item[1],reverse=True):
                if k == llm_response:
                    binary_score += 1
                    weight_score += v/max_vote
                    break
            
    binary_score = binary_score / len(annotation_dict) * 100
    weight_score = weight_score / len(annotation_dict) * 100
    
    print(binary_score)
    print(weight_score)
    
    return binary_score, weight_score

def soft_exact_match(country,language,annotation_dict,response_df,id_col,r_col,annotations_key='aggregated_answers'):
    binary_score = 0
    weight_score = 0
    valid_question_cnt = 0
    
    if language == 'Spanish':
        spark = sparknlp.start()
        
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        lemmatizer = LemmatizerModel.pretrained("lemma", "es") \
                .setInputCols(["token"]) \
                .setOutputCol("lemma")
                
        nlp_pipeline = Pipeline(stages=[document_assembler, tokenizer, lemmatizer])
        nlpPipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF('text')))
    
    elif language == 'Amharic':
        spark = sparknlp.start()
        
        document_assembler = DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        tokenizer = Tokenizer() \
            .setInputCols(["document"]) \
            .setOutputCol("token")

        lemmatizer = LemmatizerModel.pretrained("lemma", "am") \
                .setInputCols(["token"]) \
                .setOutputCol("lemma")

        nlp_pipeline = Pipeline(stages=[document_assembler,tokenizer,lemmatizer])
        nlpPipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF('text')))
    
    else:
        nlpPipeline = None
        
    en_lemmatizer = spacy.load("en_core_web_sm")
        
    response_df['binary_score'] = [None]*response_df.shape[0]
    response_df['weight_score'] = [None]*response_df.shape[0]
    
    pb = tqdm(annotation_dict.items(),total=len(annotation_dict))
    
    for qid,data in pb:
        pb.set_description(qid)
        if data['idks']['no-answer']+data['idks']['not-applicable'] >= 3 or data['idks']['idk']>=5 or len(data[annotations_key])==0:
            continue
        
        valid_question_cnt += 1
        
        llm_response = get_llm_response_by_id(response_df,qid,id_col,r_col)
        flag = False
        if llm_response and data[annotations_key]:
            max_vote = data[annotations_key][0]['count']
            
            for agg_ans in data[annotations_key]:
                if language != 'English':
                    for a in agg_ans['answers']:
                        if lemma_check(a,llm_response,nlpPipeline,language):
                            binary_score += 1
                            weight_score += agg_ans['count']/max_vote
                            flag = True
                            break
                if not flag:
                    for a in agg_ans['en_answers']:
                        if lemma_check(a,llm_response,en_lemmatizer,'English'):
                            binary_score += 1
                            weight_score += agg_ans['count']/max_vote
                            flag = True
                            break
                if flag:
                    break
        if flag:
            response_df.loc[response_df[id_col]==qid,'binary_score'] = 1
            response_df.loc[response_df[id_col]==qid,'weight_score'] = agg_ans['count']/max_vote
            print(response_df.loc[response_df[id_col]==qid])
        else:
            response_df.loc[response_df[id_col]==qid,'binary_score'] = 0
            response_df.loc[response_df[id_col]==qid,'weight_score'] = 0
            
        pb.set_postfix({'bs':binary_score/valid_question_cnt*100,'ws':weight_score/valid_question_cnt*100})
            
    binary_score = binary_score / valid_question_cnt * 100
    weight_score = weight_score / valid_question_cnt * 100
    
    print(binary_score)
    print(weight_score)
    
    return binary_score, weight_score, response_df