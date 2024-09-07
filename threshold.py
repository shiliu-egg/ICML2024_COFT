import json

base_threshold = 0.3

with open('recalled_entities.json') as f:
    ner_data = json.load(f)

score_data = []
with open('scorered_science.json', 'r') as file:
    for line in file:
        score_data.append(json.loads(line))

max_tokens = 1
min_tokens = float('inf')

for entry in ner_data:
    token_count = len(entry['entitys'])
    if token_count > max_tokens:
        max_tokens = token_count
    if token_count < min_tokens:
        min_tokens = token_count

if min_tokens > max_tokens:
    min_tokens = 1  

final_data = []

for entry in ner_data:
    id = entry['id']
    entitys = entry['entitys']
    entity_scores = score_data[id]['entity']

    token_count = len(entitys) if entitys else 1
    normalized_token_count = (token_count - min_tokens) / (max_tokens - min_tokens) if max_tokens > min_tokens else 0.5

    dynamic_percentage_threshold = 0.5 * min(base_threshold + normalized_token_count, 1)

    num_entities_to_select = int(len(entity_scores) * dynamic_percentage_threshold)

    sorted_entities = sorted(entity_scores.items(), key=lambda item: item[1], reverse=True)

    high_score_entities = [entity for entity, score in sorted_entities[:num_entities_to_select] if score > 0]

    final_data.append({
        'id': id,
        'entitys': high_score_entities
    })


with open('dynamic.json', 'w') as outfile:
    json.dump(final_data, outfile, indent=4)
