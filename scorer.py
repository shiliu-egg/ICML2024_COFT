import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import os
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'

class RelevanceCalculator:
    def __init__(self, model_name: str, device: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.model.eval()
        self.device = device

    def calculate_self_info(self, text: str) -> Tuple[List[str], List[float]]:
        with torch.no_grad():
            encoding = self.tokenizer(text, add_special_tokens=True, return_tensors='pt', truncation=True, max_length=1024)
            encoding = encoding.to(self.device)
            outputs = self.model(**encoding, labels=encoding['input_ids'])
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            self_info = -torch.log(probs)

        input_ids = encoding['input_ids']
        input_ids_expanded = input_ids[:, :-1].unsqueeze(-1)

        tokens = [self.tokenizer.decode(token_id) for token_id in input_ids.squeeze().tolist()[1:]]
        self_info_values = self_info[:, 1:].gather(-1, input_ids_expanded).squeeze(-1).squeeze(0).tolist()
        return tokens, self_info_values

    def calculate_ts_isf_score(self, question: str, answer: str) -> List[Tuple[str, float]]:
        tokens, self_infos = self.calculate_self_info(answer)
        relevance_scores = []
        token_tf_isf = {}

        # TF parts
        all_sentences = question.split('.')
        sentence_count = len(all_sentences)
        token_sent_counts = {}

        for sentence in all_sentences:
            seen_tokens = set()
            for token in tokens:
                if token in sentence and token not in seen_tokens:
                    seen_tokens.add(token)
                    if token not in token_sent_counts:
                        token_sent_counts[token] = 0
                    token_sent_counts[token] += 1

        epsilon = 1e-5
        max_self_info = max(self_infos)
        min_self_info = min(self_infos)

        # TF-ISF parts
        for token, self_info in zip(tokens, self_infos):
            word = token.strip()


            tf = tokens.count(word) / len(tokens)  
            isf = math.log(sentence_count / (1 + token_sent_counts.get(word, 0)))  

            tf_isf = tf * isf  


            norm_self_info = (self_info - min_self_info) / (max_self_info - min_self_info) if max_self_info != min_self_info else 1e-5

            # relevance_scores parts
            ts_isf_score = tf_isf * norm_self_info
            relevance_scores.append((word, ts_isf_score))

        return relevance_scores

def process_wiki_file(wiki_file_path, output_file_path, model_name, device):
    calculator = RelevanceCalculator(model_name, device)

    with open(wiki_file_path, 'r', encoding='utf-8') as wiki_file, open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in wiki_file:
            entry = json.loads(line)  
            question = entry['prompt'] + ' ' + ' '.join(entry['segmented_response'])
            answer = str(entry['ref_contents'])  


            relevance_scores = calculator.calculate_ts_isf_score(question, answer)


            output_entry = {
                'id': entry['index'],
                'entity': {token: score for token, score in relevance_scores}
            }
            output_file.write(json.dumps(output_entry) + '\n')



process_wiki_file('science.jsonl', 'scorered_science.json', '[YOUR MODEL PATH HERE]', 'cuda')
