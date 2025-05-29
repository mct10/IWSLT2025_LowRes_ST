import csv
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import List, Tuple
from typing import Optional

import numpy as np
import soundfile
import torch
from torch.utils.data import Dataset, DataLoader

from seamlessm4t.enums import FinetuneMode
from utils.log_utils import logging

logger = logging.getLogger(__name__)


@dataclass
class MultiModalBatch:
    # target must be available
    tgt_inputs: torch.Tensor  # tgt[:-1]
    tgt_outputs: torch.Tensor  # tgt[1:]
    tgt_att_mask: torch.Tensor  # attention mask for `tgt_outputs`; 0 means pad; used for loss computation

    # either text or speech or both should be available though
    src_sp_tokens: Optional[torch.Tensor] = None
    src_sp_att_mask: Optional[torch.Tensor] = None

    src_text_tokens: Optional[torch.Tensor] = None
    src_text_att_mask: Optional[torch.Tensor] = None

    def to_device(self, device):
        for field in fields(MultiModalBatch):
            field_value = getattr(self, field.name, None)
            if field_value is not None:
                setattr(self, field.name, field_value.to(device))

    def __del__(self) -> None:
        for field in fields(MultiModalBatch):
            field_value = getattr(self, field.name, None)
            if field_value is not None:
                del field_value


@dataclass
class MultiModalInstance:
    tgt_text: str

    src_text: Optional[str] = None
    src_speech: Optional[str] = None


@dataclass
class BatchingConfig:
    max_text_tokens: int = 128
    """Maximum number of text tokens. Will truncate if exceeds."""

    max_speech_frames: int = 30 * 16000
    """Maximum number of speech frames. Will truncate if exceeds."""

    batch_size: int = 5
    """Fixed batch size to use"""

    num_workers: int = 2
    """Parallelism in dataset preparation."""

    float_dtype: torch.dtype = torch.float16
    """Select between fp16/fp32 for float tensors """


class MultiModalDataset(Dataset):
    def __init__(self, dataset_manifest_path):
        self.data: List[Tuple[str, str, str]] = self._load_manifest(dataset_manifest_path)

    def __getitem__(self, i) -> MultiModalInstance:
        return MultiModalInstance(
            src_speech=self.data[i][0],
            src_text=self.data[i][1],
            tgt_text=self.data[i][2],
        )

    def __len__(self):
        return len(self.data)

    def _load_manifest(self, path) -> List[Tuple[Optional[str], Optional[str], str]]:
        """
        Expecting a tsv file, having `audio_path`, `src_text` and `tgt_text` fields
        """
        res = []
        with open(path) as fp:
            for item in csv.DictReader(fp, delimiter="\t"):
                item: dict
                instance = [
                    item.get("audio_path", None), item.get("src_text", None), item["tgt_text"]
                ]
                res.append(tuple(instance))
        return res


class MultiModalDataLoader:
    def __init__(
            self,
            src_lang: str, tgt_lang: str,
            processor,
            dataset_manifest_path: Path,
            batching_config: BatchingConfig,
            data_mode: FinetuneMode
    ):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.processor = processor
        self.batching_config = batching_config
        self.dataset = MultiModalDataset(
            dataset_manifest_path=dataset_manifest_path
        )
        self.data_mode = data_mode

        self.eos_id = self.processor.tokenizer.convert_tokens_to_ids("</s>")  # should be 3

        logger.info(f"Init data mode={self.data_mode}")
        logger.info(f"Batch config: {self.batching_config}")

    def get_dataloader(self) -> DataLoader[MultiModalBatch]:
        data_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batching_config.batch_size,
            shuffle=True,
            num_workers=self.batching_config.num_workers,
            collate_fn=self._prepare_batch,
            persistent_workers=True
        )
        return data_loader

    def _prepare_batch(self, raw_samples: List[MultiModalInstance]) -> MultiModalBatch:
        # target text is always needed
        tgt_text = self._prep_tokens(
            sents=[ins.tgt_text for ins in raw_samples],
            lang=self.tgt_lang,
            is_target=True
        )

        batch = MultiModalBatch(
            tgt_inputs=tgt_text["input_ids"][:, :-1],
            tgt_outputs=tgt_text["input_ids"][:, 1:],
            tgt_att_mask=tgt_text["attention_mask"][:, 1:]  # fix
        )

        if self.data_mode == FinetuneMode.SPEECH_TO_TEXT or self.data_mode == FinetuneMode.MULTI_TASKING:
            # add speech inputs
            speech_inputs = self._prep_audio_features([ins.src_speech for ins in raw_samples])
            batch.src_sp_tokens = speech_inputs["input_features"]
            batch.src_sp_att_mask = speech_inputs["attention_mask"]
        if self.data_mode == FinetuneMode.TEXT_TO_TEXT or self.data_mode == FinetuneMode.MULTI_TASKING:
            # add text inputs
            src_text = self._prep_tokens(
                sents=[ins.src_text for ins in raw_samples],
                lang=self.src_lang,
                is_target=False
            )
            batch.src_text_tokens = src_text["input_ids"]
            batch.src_text_att_mask = src_text["attention_mask"]

        return batch

    def _prep_audio_features(self, sp_paths: List[str]) -> dict:
        # a list of (C=1, T)
        audio_arrays = [
            np.expand_dims(soundfile.read(path)[0], axis=0)
            for path in sp_paths
        ]
        # truncate long audios
        audio_arrays = [
            audio if audio.shape[1] <= self.batching_config.max_speech_frames
            else audio[:, :self.batching_config.max_speech_frames]
            for audio in audio_arrays
        ]
        # see https://huggingface.co/docs/transformers/v4.50.0/en/model_doc/seamless_m4t#transformers.SeamlessM4TFeatureExtractor
        # returns a dict of {
        #       "input_features": bsz, seq, 80*2=160,
        #       "attention_mask": bsz, seq
        # }
        return self.processor.feature_extractor(
            raw_speech=audio_arrays,
            return_attention_mask=True,
            return_tensors="pt",
            sampling_rate=16000  # it must be...
        )

    def _prep_tokens(self, sents: List[str], lang: str, is_target: bool) -> dict:
        # if target, the tokens follow
        # </s><lang><tok...></s><pad>
        # if source, the tokens follow
        # <lang><tok...></s><pad>
        # see https://huggingface.co/docs/transformers/v4.50.0/en/model_doc/seamless_m4t#transformers.SeamlessM4TTokenizerFast
        args = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "return_attention_mask": True,
            "max_length": self.batching_config.max_text_tokens
        }
        if is_target:
            args["text_target"] = sents
            args["tgt_lang"] = lang
        else:
            args["text"] = sents
            args["src_lang"] = lang
        tokens = self.processor.tokenizer(**args)
        # returns a dict of {
        #       "input_ids": bsz, seq,
        #       "attention_mask": bsz, seq
        # }
        return tokens
