import csv
import json

with open('pubmed_subset_50k.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    headers = next(reader)
    data = list(reader)

jsonl = []

for row in data:
    jsonl.append({'Date': row[2], 'Title': row[6], 'Abstract': row[4], 'Publication': row[7]})

with open('fpubmed_subset_50k.jsonl', 'w', encoding='utf-8') as file:
    for line in jsonl:
        file.write(json.dumps(line) + '\n')