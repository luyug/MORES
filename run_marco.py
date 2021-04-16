# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import logging
import os
import sys
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple
from tqdm import tqdm

import numpy as np
import torch

from arguments import ModelArguments, DataArguments, \
    MORESTrainingArguments as TrainingArguments
from marco_datasets import MarcoTrainDataset, MarcoPredDataset
from transformers import AutoConfig, AutoTokenizer
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import (
    HfArgumentParser,
    set_seed,
)

from modeling import MORESSym, MORES
from trainer import MORESTrainer as Trainer

logger = logging.getLogger(__name__)


@dataclass
class QryDocCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 16
    max_p_len: int = 128

    def __call__(
            self, features
    ):
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]
        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )
        if len(features[0]) == 2:
            return q_collated, d_collated
        elif len(features[0]) == 3:
            labels = torch.LongTensor([f[2] for f in features])
            return q_collated, d_collated, labels
        else:
            raise ValueError(f'Got examples should be of lengths 2 or 3, Got {len(features[0])}.')


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    model = MORES.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    if training_args.do_train:
        train_dataset = MarcoTrainDataset(
            data_args, data_args.train_path, tokenizer=tokenizer,
        )
    else:
        train_dataset = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=QryDocCollator(
            tokenizer,
            max_p_len=data_args.p_max_len,
            max_q_len=data_args.q_max_len
        ),
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_predict:
        logging.info("*** Prediction ***")

        if os.path.exists(data_args.rank_score_path):
            if os.path.isfile(data_args.rank_score_path):
                raise FileExistsError(f'score file {data_args.rank_score_path} already exists')
            else:
                raise ValueError(f'Should specify a file name')
        else:
            score_dir = os.path.split(data_args.rank_score_path)[0]
            if not os.path.exists(score_dir):
                logger.info(f'Creating score directory {score_dir}')
                os.makedirs(score_dir)

        test_dataset = MarcoPredDataset(
            data_args.pred_path, tokenizer=tokenizer,
            q_max_len=data_args.q_max_len,
            p_max_len=data_args.p_max_len,
        )
        assert data_args.pred_id_file is not None

        pred_qids = []
        pred_pids = []
        with open(data_args.pred_id_file) as f:
            for l in f:
                q, p = l.split()
                pred_qids.append(q)
                pred_pids.append(p)

        pred_scores = trainer.predict(test_dataset=test_dataset).predictions

        if trainer.is_world_process_zero():
            assert len(pred_qids) == len(pred_scores)
            with open(data_args.rank_score_path, "w") as writer:
                for qid, pid, score in zip(pred_qids, pred_pids, pred_scores):
                    writer.write(f'{qid}\t{pid}\t{score}\n')

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
