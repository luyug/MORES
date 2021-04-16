import argparse
import datasets
import os
import json
from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--ranking_file', type=str)
parser.add_argument('--save_to', type=str)
parser.add_argument('--n_query', type=int, default=-1)
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--truncate', type=int, default=128)

args = parser.parse_args()

# queries = '/home/luyug/data/marco-doc/anserini/anserini/' \
#           'collections/msmarco-passage/queries.dev.small.tsv'
#
# query_map = {}
# with open(queries) as f:
#     for l in f:
#         qid, qry = l.strip().split('\t')
#         query_map[qid] = qry
#
# collection = '/home/luyug/data/marco-psg/raw/collection.tsv'
# collection = datasets.load_dataset(
#     'csv',
#     data_files=collection,
#     column_names=['pid', 'psg'],
#     delimiter='\t',
#     ignore_verifications=True,
# )['train']

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
file_name = os.path.split(args.ranking_file)[1]
if args.n_query > 0:
    file_name = file_name + f'.n{args.n_query}'
query_set = set()
with open(args.ranking_file, 'r') as rank_in_ids, \
        open(os.path.join(args.save_to, file_name), 'w') as rank_texts:
    for l in rank_in_ids:
        qid, pid, qry, psg = l.strip().split('\t')
        query_set.add(qid)
        if 0 < args.n_query < len(query_set):
            break
        encoded_passage = tokenizer.encode(
            psg,
            add_special_tokens=False,
            max_length=args.truncate,
            truncation=True
        )
        encoded_query = tokenizer.encode(
            qry,
            add_special_tokens=False,
            max_length=args.truncate,
            truncation=True
        )
        item_dict = {
            'qid': qid,
            'pid': pid,
            'qry': encoded_query,
            'psg': encoded_passage,
        }
        rank_texts.write(json.dumps(item_dict) + '\n')
