import json
from argparse import ArgumentParser
from multiprocessing import Pool

from tqdm import tqdm
from transformers import AutoTokenizer

parser = ArgumentParser()
parser.add_argument('--input_file', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--max_query_length', type=int, default=16)
parser.add_argument('--tokenizer_name', required=True)

args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)


def encode_line(line: str):
    _, qry, psg, label = line.split('\t')
    qry_encoded = tokenizer.encode(
        qry,
        truncation=True,
        max_length=args.max_query_length,
        add_special_tokens=False,
        padding=False,
    )
    psg_encoded = tokenizer.encode(
        psg,
        truncation=True,
        max_length=args.max_length,
        add_special_tokens=False,
        padding=False
    )

    label = int(label)

    entry = {
        'qry': qry_encoded,
        'psg': psg_encoded,
        'label': label,
    }

    return json.dumps(entry)


with open(args.input_file, 'r') as f:
    lines = f.readlines()

with open(args.save_to, 'w') as jfile:
    with Pool() as p:
        all_json_items = p.imap(
            encode_line,
            tqdm(lines),
            chunksize=512
        )
        for json_item in all_json_items:
            jfile.write(json_item + '\n')