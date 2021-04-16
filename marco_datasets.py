import json
import random
import datasets
from collections import defaultdict
from typing import Union, List, Callable, Dict

from torch.utils.data import Dataset

from arguments import DataArguments
from transformers import PreTrainedTokenizer, BatchEncoding, EvalPrediction


class MarcoTrainDataset(Dataset):
    columns = [
        'qry', 'psg', 'label'
    ]

    def __init__(
            self,
            args: DataArguments,
            path_to_tsv: Union[List[str], str],
            tokenizer: PreTrainedTokenizer
    ):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=path_to_tsv,
            ignore_verifications=False,
        )['train']

        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.args.q_max_len if is_query else self.args.p_max_len,
            padding=False,
            return_attention_mask=False,
        )
        return item

    def __len__(self):
        return len(self.nlp_dataset)

    def __getitem__(self, item) -> [BatchEncoding, BatchEncoding, int]:
        example = self.nlp_dataset[item]
        qry, psg, label = (example[x] for x in self.columns)

        encoded_qry = self.create_one_example(qry, True)
        encoded_doc = self.create_one_example(psg, False)

        return encoded_qry, encoded_doc, label


class MarcoPredDataset(Dataset):
    columns = [
        'qid', 'pid', 'qry', 'psg'
    ]

    @staticmethod
    def read_json_files(ff: List[str]):
        all_items = []
        for f in ff:
            with open(f, 'r') as j:
                lines = j.readlines()

            items = [json.loads(l) for l in lines]
            all_items.extend(items)
        return all_items

    def __init__(self, path_to_json: List[str], tokenizer: PreTrainedTokenizer, q_max_len=16, p_max_len=128):
        self.nlp_dataset = self.read_json_files(path_to_json)

        self.tok = tokenizer
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len

        self._n_split = 0
        self._curr_split = 0
        self._chunk_size = 0

        print('dataset loaded', flush=True)

    def split(self, n_split: int):
        total = len(self)
        self._n_split = n_split
        self._curr_split = 0
        self._chunk_size = int(total / n_split) + 1

    def set_split(self, split: int):
        if self._n_split <= split or split < 0:
            raise ValueError(f'Have {self._n_split} splits, attempt to set to {split}')
        self._curr_split = split

    def __len__(self):
        if self._n_split == 0:
            return len(self.nlp_dataset)
        elif self._curr_split == self._n_split - 1:
            return len(self.nlp_dataset) - self._chunk_size * (self._n_split - 1)
        else:
            return self._chunk_size

    def __getitem__(self, item) -> [BatchEncoding, BatchEncoding]:
        if self._n_split > 0:
            item = self._curr_split*self._chunk_size + item
        qid, pid, qry, psg = (self.nlp_dataset[item][f] for f in self.columns)
        encoded_qry = self.tok.encode_plus(
            qry,
            truncation='only_first',
            max_length=self.q_max_len,
            padding=False,
            return_attention_mask=False,
        )
        encoded_psg = self.tok.encode_plus(
            psg,
            max_length=self.p_max_len,
            truncation='only_first',
            padding=False,
            return_attention_mask=False,
        )
        return encoded_qry, encoded_psg
