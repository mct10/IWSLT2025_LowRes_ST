import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES


@dataclass
class SeqsBatch:
    src_tokens: Tensor
    src_att_mask: Tensor  # attention mask for src

    tgt_inputs: Tensor  # tgt[:-1]
    tgt_outputs: Tensor  # tgt[1:]
    tgt_att_mask: Tensor  # attention mask for `tgt_outputs`; 0 means pad; used for loss computation

    def to_device(self, device):
        self.src_tokens = self.src_tokens.to(device)
        self.src_att_mask = self.src_att_mask.to(device)
        self.tgt_inputs = self.tgt_inputs.to(device)
        self.tgt_outputs = self.tgt_outputs.to(device)
        self.tgt_att_mask = self.tgt_att_mask.to(device)

    def __del__(self) -> None:
        """Explicitly delete tensors to force GPU memory cleanup"""
        for tensor in [
            self.src_tokens,
            self.src_att_mask,
            self.tgt_inputs,
            self.tgt_outputs,
            self.tgt_att_mask,
        ]:
            if tensor is not None:
                del tensor


@dataclass
class BatchingConfig:
    max_num_tokens: int = 128
    """Maximum number of tokens. Will truncate if exceeds."""

    batch_size: int = 5
    """Fixed batch size to use"""

    num_workers: int = 2
    """Parallelism in dataset preparation."""

    float_dtype: torch.dtype = torch.float16
    """Select between fp16/fp32 for float tensors """


@dataclass
class RawBitext:
    src_text: str
    tgt_text: str


class NllbDataset(Dataset):
    def __init__(self, dataset_manifest_path):
        self.data: List[Tuple[str, str]] = self._load_manifest(dataset_manifest_path)

    def __getitem__(self, i) -> RawBitext:
        return RawBitext(
            src_text=self.data[i][0], tgt_text=self.data[i][1],
        )

    def __len__(self):
        return len(self.data)

    def _load_manifest(self, path) -> List[Tuple[str, str]]:
        """
        Expecting a tsv file which has `src_text` and `tgt_text` fields.
        """
        res = []
        with open(path) as fp:
            for item in csv.DictReader(fp, delimiter="\t"):
                res.append(
                    (str(item["src_text"]), str(item["tgt_text"]))
                )
        return res


class NllbDataLoader:
    def __init__(
            self,
            src_lang: str, tgt_lang: str,
            text_tokenizer,
            dataset_manifest_path: Path,
            batching_config: BatchingConfig
    ):
        assert src_lang in FAIRSEQ_LANGUAGE_CODES, f"Unsupported {src_lang}"
        assert tgt_lang in FAIRSEQ_LANGUAGE_CODES, f"Unsupported {tgt_lang}"

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.text_tokenizer = text_tokenizer
        self.batching_config = batching_config
        self.dataset = NllbDataset(
            dataset_manifest_path=dataset_manifest_path
        )

        self.eos_id = self.text_tokenizer.convert_tokens_to_ids("</s>")  # 2
        self.pad_id = self.text_tokenizer.convert_tokens_to_ids("<pad>")  # 1

    def get_dataloader(self) -> DataLoader[SeqsBatch]:
        data_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batching_config.batch_size,
            shuffle=True,
            num_workers=self.batching_config.num_workers,
            collate_fn=self._prepare_batch,
            persistent_workers=True
        )
        return data_loader

    def _prepare_batch(self, raw_samples: List[RawBitext]) -> SeqsBatch:
        src_sents = [sample.src_text for sample in raw_samples]
        # src
        self.text_tokenizer.src_lang = self.src_lang
        src_inputs = self.text_tokenizer(
            src_sents, return_tensors='pt', padding=True, truncation=True,
            return_attention_mask=True,
            max_length=self.batching_config.max_num_tokens
        )

        # tgt
        tgt_sents = [sample.tgt_text for sample in raw_samples]
        self.text_tokenizer.src_lang = self.tgt_lang
        tgt_inputs = self.text_tokenizer(
            tgt_sents, return_tensors='pt', padding=True, truncation=True,
            return_attention_mask=True,
            max_length=self.batching_config.max_num_tokens
        )

        # fix the decoder input by pre-pending <eos>
        decoder_input = torch.cat(
            [torch.zeros(len(raw_samples), 1, dtype=torch.long).fill_(self.eos_id), tgt_inputs["input_ids"][:, :-1]],
            dim=1
        )

        return SeqsBatch(
            src_tokens=src_inputs["input_ids"],
            src_att_mask=src_inputs["attention_mask"],
            tgt_inputs=decoder_input,
            tgt_outputs=tgt_inputs["input_ids"],
            tgt_att_mask=tgt_inputs["attention_mask"]
        )
