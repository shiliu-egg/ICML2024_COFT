import spacy
import requests
import json
import nltk
import time
import random

nlp = spacy.load("en_core_web_sm")


SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"


def extract_nouns(text):
    noun_phrases = []
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
        chunk_parser = nltk.RegexpParser('''
            NP: {<DT>?<JJ>*<NN.*>+}
        ''')
        tree = chunk_parser.parse(tagged_words)
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
            noun_phrase = ' '.join(word for word, _ in subtree.leaves())
            noun_phrases.append(noun_phrase)
    return noun_phrases


def get_entity_id(entity_name):
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'limit': 1,
        'search': entity_name
    }
    try:
        response = requests.get('https://www.wikidata.org/w/api.php', params=params)
        results = response.json().get('search', [])
        if results:
            return results[0]['id'], results[0]['label']
        return '', ''
    except Exception as e:
        print(f"Error retrieving entity ID for {entity_name}: {e}")
        return '', ''

def check_entities_association(entity_id1, entity_id2):
    query = f"""
    ASK {{
        {{ wd:{entity_id1} ?p wd:{entity_id2} }} UNION
        {{ wd:{entity_id2} ?p wd:{entity_id1} }} 
    }}
    """
    response = requests.get(SPARQL_ENDPOINT, params={'query': query, 'format': 'json'})
    return response.json().get('boolean', False)


def process_text(text):
    tokens = nltk.word_tokenize(text)
    stop_words = nltk.corpus.stopwords.words('english')
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    return tokens

def process_text_and_query_wikidata(text):

    doc = nlp(text)


    ner_entities = [ent.text for ent in doc.ents if ent.label_ not in ['DATE', 'CARDINAL', 'PERCENT', 'MONEY', 'QUANTITY', 'TIME']]
    
    nouns = extract_nouns(text)

    unique_entities = set(ner_entities + nouns)
    filtered_entities = process_text(' '.join(unique_entities))


    entity_ids = {entity: None for entity in filtered_entities} 

    for entity in filtered_entities:
        entity_id, entity_label = get_entity_id(entity)
        if entity_id:
            entity_ids[entity] = entity_id  
            print(f"Found entity {entity_label} with ID: {entity_id}")


    associated_entities = []
    entity_list = list(entity_ids.keys())
    for i in range(len(entity_list)):
        for j in range(i + 1, len(entity_list)):
            entity_id1 = entity_ids[entity_list[i]]
            entity_id2 = entity_ids[entity_list[j]]
            if entity_id1 and entity_id2:  
                if check_entities_association(entity_id1, entity_id2):
                    associated_entities.append((entity_list[i], entity_list[j]))
                    print(f"Entities {entity_list[i]} and {entity_list[j]} are associated in Wikidata.")

    return list(entity_ids.keys())

output_data = []

with open('science.jsonl', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        entry = json.loads(line)
        if entry["domain"] == "science":
            text = ' '.join(entry["segmented_response"]) + " " + entry["prompt"]
            print(f"Processing text: {text}")
            entities = process_text_and_query_wikidata(text)
            print(f"Extracted Entities: {entities}")


            output_data.append({
                "id": idx,
                "entitys": entities
            })

with open('recalled_entities.json', 'w', encoding='utf-8') as json_file:
    json.dump(output_data, json_file, ensure_ascii=False, indent=4)

