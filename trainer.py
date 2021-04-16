import os

from transformers.trainer import Trainer, nested_detach
from transformers.optimization import AdamW, get_scheduler

from lamb import Lamb
from modeling import MORESSym
from typing import Dict, List, Tuple, Optional, Any, Union

import torch
from torch import nn
from torch.cuda.amp import autocast

from transformers.trainer_utils import PredictionOutput, EvalPrediction
from torch.utils.data import DataLoader

from fairscale.optim import OSS

import logging

logger = logging.getLogger(__name__)


class MORESTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save_pretrained'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save_pretrained interface')
        else:
            self.model.save_pretrained(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _prepare_inputs(
            self, inputs: Tuple[Dict[str, Union[torch.Tensor, Any]]]
    ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
        prepared = []
        for x in inputs[:2]:
            prepared.append(super()._prepare_inputs(x))
        if len(inputs) == 3:
            prepared.append(inputs[2].to(self.args.device))
        return prepared

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        if self.args.warmup_ratio > 0:
            self.args.warmup_steps = num_training_steps * self.args.warmup_ratio

        super().create_optimizer_and_scheduler(num_training_steps)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` is a :obj:`torch.utils.data.IterableDataset`, a random sampler
        (adapted to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, model, inputs):
        qry, doc, label = inputs
        return model(qry, doc, label).loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        qry, doc = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if self.args.fp16:
                with autocast():
                    outputs = model(qry, doc)
            else:
                outputs = model(qry, doc)

            loss = outputs.loss
            logits = outputs.logits

            # extract the postive probability
            if self.args.fp16:
                with autocast():
                    logits = torch.softmax(logits, -1)[:, -1]
            else:
                logits = torch.softmax(logits, -1)[:, -1]


        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        labels = None

        return (loss, logits, labels)

    def prediction_loop(
            self,
            *args,
            **kwargs
    ) -> PredictionOutput:
        pred_outs = super().prediction_loop(*args, **kwargs)
        preds, label_ids, metrics = pred_outs.predictions, pred_outs.label_ids, pred_outs.metrics
        preds = preds.squeeze()
        if self.compute_metrics is not None:
            metrics_no_label = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
        else:
            metrics_no_label = {}

        for key in list(metrics_no_label.keys()):
            if not key.startswith("eval_"):
                metrics_no_label[f"eval_{key}"] = metrics_no_label.pop(key)

        print(metrics_no_label, flush=True)
        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics={**metrics, **metrics_no_label})
