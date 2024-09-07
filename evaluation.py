
import json
import re
import spacy


import time
import logging
import requests
from typing import Optional, List, Dict, Mapping, Any
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import langchain
from langchain.llms.base import LLM
from langchain.cache import InMemoryCache
import ipdb
import csv
import random
import nltk

from nltk.tokenize import word_tokenize
import string
import openai

from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')

random.seed(100)

def annotate_text(text, entity_list, annotate_by='sentence'):

  
    if annotate_by == 'sentence':
        text_units = sent_tokenize(text)
    elif annotate_by == 'paragraph':
        text_units = text.split('\n') 
    else:
        raise ValueError("annotate_solely_by 'sentence' or 'paragraph'")


    entity_presence = []
    for i, unit in enumerate(text_units):
        unit_words = word_tokenize(unit)

        entity_count = sum(unit_words.count(entity) for entity in entity_list)
        if entity_count > 0:
            entity_presence.append((i, entity_count))

    entity_presence_sorted = sorted(entity_presence, key=lambda x: x[1], reverse=True)


    annotations = set()
    total_length_annotated = 0
    for index, count in entity_presence_sorted:
        unit = text_units[index]

        if total_length_annotated + len(unit) <= len(text) * 0.5:
            annotations.add(index)
            total_length_annotated += len(unit)
  
            if annotate_by == 'sentence' and len(annotations) >= len(text_units) * 0.5:
                break


    annotated_text = []
    for i, unit in enumerate(text_units):
        if i in annotations:
            annotated_text.append(f"【{unit}】")
        else:
            annotated_text.append(unit)

    if annotate_by == 'sentence':
        return ' '.join(annotated_text)
    else:  
        return '\n'.join(annotated_text)


def highlight_entities(text, entity_list):

    entity_list = [item for item in entity_list if item.strip() and text.count(item) < 10]
    pattern = r"\b(" + "|".join(entity_list) + r")\b"

    highlighted_text = re.sub(pattern, r"【\1】", text, flags=re.IGNORECASE)

    return highlighted_text


def ChatGPTGenerate(prompt):  

    chat_completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": prompt}])
    output= chat_completion.choices[0].message.content
    return output

def ChatGPTGenerate_3(prompt):  

    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
    output= chat_completion.choices[0].message.content
    return output
            

if __name__ == "__main__":


    tokenizer = AutoTokenizer.from_pretrained("[YOUR MODEL PATH HERE]")
    llm = AutoModelForCausalLM.from_pretrained("[YOUR MODEL PATH HERE]",device_map="auto")
    node_name = []
    prompts =[]
    sentences = []
    labels = []
    comments = []
    types = []
    refs = []
    ref_contents = []
    path_name = 'dynamic.json' 

    test_name = "science"

    with open('science.jsonl', 'r', encoding='utf-8') as f:
        for line in f:        
            j = json.loads(line)
            if j["domain"] == test_name:  
                prompts.append(j["prompt"])
                sentences.append(j["segmented_response"])
                labels.append(j["labels"])
                comments.append(j["comment"])
                types.append(j["type"])
                refs.append(j["ref"])
                ref_contents.append(j["ref_contents"])

    with open(path_name, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    print('ok')
    
    id_entity = {}
    for item in entities:
       id_entity[item['id']] = item["entitys"]


    test_num = range(len(prompts))

    TP = 0
    FP = 0
    FN = 0
    true_positive = 0
    positive_predictions = 0
    print('len: ', len(prompts))
    test_num = range(len(prompts))

    print(test_num)


     
    need_con = True
    need_ref = False
    use_zj_api = True
    con = False
    a_type = 'paragraph'
    print('sam_len: ', len(test_num))
    ref_contents_new = []
    log = []
    for i in test_num:
        triples = ''
        
        if len(ref_contents[i]) > 0:
            if ref_contents[i][0].find('Something went wrong') != -1 or len(ref_contents[i][0]) < 50:
                ref_contents[i] = ''
            try:
                entity_list = id_entity[i]
                highlighted_text = annotate_text(ref_contents[i][0], entity_list) #, annotate_by=a_type
                print('ok----------------------------------------------------------------------')
                log.append('****'+str(entity_list))
            except:
                print('not ok----------------------------------------------------------------------')
                ref_contents[i] = ''

        prompt_add = 'Please do not directly judge vague information as false.'   

        if con:
            prompt_add += 'Please generate sequential reading notes for the retrieved reference text, ensuring a systematic assessment of their relevance to the input question before formulating a final response. '
        if len(ref_contents[i]) > 0 and need_con:
            human_input = "I will show you a question, a list of text segments, and a reference text. All the segments can be concatenated to form a complete answer to the question. Your task is to assess whether each text segment contains factual errors or not based on the reference text. Please do not directly judge vague information as false. \
            Please pay close attention to the content in the symbol【 】in the reference text."+prompt_add+"\
            \nPlease generate using the following format:\
            \nAnswer: List the judgment results for each segments, such as: [True, False, ...]. Please only output the list, no more details. Note that the length of the list should be "+str(len(labels[i]))+"\
            \nBelow are my inputs:\
            \nQuestion: "+prompts[i]+\
            "\nSegments:\n"+str(sentences[i])+\
            "\nReference text: "+highlighted_text+"\n"
        else:
            human_input = "I will show you a question, and a list of text segments. All the segments can be concatenated to form a complete answer to the question. Your task is to assess whether each text segment contains factual errors or not. \
            "+prompt_add+"\
            \nPlease generate using the following format:\
            \nAnswer: List the judgment results for each segments, such as: [True, False, ...]. Please only output the list, no more details. Note that the length of the list should be "+str(len(labels[i]))+"\
            \nBelow are my inputs:\
            \nQuestion: "+prompts[i]+\
            "\nSegments:\n"+str(sentences[i])+"\n"

        print(i, 'len: ', len(human_input))
        if use_zj_api:
            input_ids = tokenizer([human_input], return_tensors="pt").to(llm.device)
            output = llm.generate(
                **input_ids,
                do_sample=True,
                temperature=0.1,
                repetition_penalty=10.0,
                max_length=2056
            )
            response = tokenizer.decode(output[0], skip_special_tokens=True)

        else:
            pass
        print("output: "+response)
        print("label: "+str(labels[i]))
        string = response
        log.append('human_input: '+human_input)
        log.append("output: "+response)
        log.append("label: "+str(labels[i]))
        if response == '空值':
            continue
        try:
            start_index = string.index("[")
            end_index = string.index("]") + 1
            list_string = string[start_index:end_index]
            output = eval(list_string)
        except:
            print('No answer')
            continue
        
        for x in range(min(len(output),len(labels[i]))):
            if output[x] and labels[i][x]:
                true_positive += 1
            if output[x]:
                positive_predictions += 1
    
        TP += sum([1 for o, l in zip(output, labels[i]) if o and l])
        FP += sum([1 for o, l in zip(output, labels[i]) if o and not l])
        FN += sum([1 for o, l in zip(output, labels[i]) if not o and l])
      

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)

    precision = true_positive / positive_predictions
    print("Precision:", precision) # Double check
    with open("log_vicuna.txt", "w", encoding="utf-8") as file: # save log
        for entry in log:
            file.write(entry + "\n")
        