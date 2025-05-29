from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import transformers
from torch.optim import AdamW
from torch.types import Device
from transformers import AutoModelForSeq2SeqLM

from nllb.dataloader import SeqsBatch, NllbDataLoader
from utils.log_utils import logging
from utils.model_utils import log_model_info

logger = logging.getLogger(__name__)


@dataclass
class FinetuneParams:
    model_name: str
    """Model name of model being finetuned."""

    save_model_path: Path
    """Path were to save finetuned model."""

    train_batch_size: int = 5
    """The batch size during train steps"""

    eval_batch_size: int = 5
    """The batch size during evaluation."""

    learning_rate: float = 1e-5
    """ Optimizer learining rate """

    label_smoothing: float = 0.2
    """ Label smoothing coefficient for nll_loss """

    warmup_steps: int = 100
    """ Number of steps with linearly increasing LR"""

    log_steps: int = 10
    """ Log inner loss after each `log_steps` training steps"""

    eval_steps: int = 50
    """ Get eval loss after each `eval_steps` training steps """

    update_freq: int = 1
    """Gradient accumulation."""

    patience: int = 3
    """ Terminate if eval loss did not improve
    over the last `patience * eval_steps` training steps"""

    max_epochs: int = 10
    """ Maximum number of training epochs"""

    float_dtype: torch.dtype = torch.float16
    """Float Dtype"""

    device: Device = torch.device("cuda")
    """ Where to run computation"""


class CalcLoss:
    """Calculate NLL loss for MT."""

    def __init__(
            self,
            num_classes: int,
            label_smoothing: float,
    ):
        """
        Ref: https://discuss.pytorch.org/t/what-is-the-formula-for-cross-entropy-loss-with-label-smoothing/149848
        """
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        eps = label_smoothing / num_classes
        self.negative = eps
        self.positive = (1.0 - label_smoothing) + eps

    def __call__(self, batch: SeqsBatch, text_logits: torch.Tensor):
        """
        text_logits: (bsz, seq, vocab_size)
        """
        assert self.num_classes == text_logits.shape[-1], f"{self.num_classes} != {text_logits.shape}"

        # logits -> log prob; (bsz, seq, vocab_size)
        text_log_softmax = text_logits.log_softmax(dim=2)

        # target distribution; (bsz, seq, vocab_size)
        target = torch.zeros_like(text_log_softmax).to(text_log_softmax.device)
        target.fill_(self.negative)
        # target_tokens: (bsz, seq) -> (bsz, seq, 1)
        target.scatter_(2, batch.tgt_outputs.unsqueeze(2), self.positive)

        # nll loss for log_softmax
        # attention mask: bsz, seq
        loss = torch.sum(-target * text_log_softmax, dim=2) * batch.tgt_att_mask
        # tok-level average: bsz, seq -> 1
        loss = torch.sum(loss) / torch.sum(batch.tgt_att_mask)
        return loss


class NllbFinetune:
    def __init__(
            self,
            params: FinetuneParams,
            train_dataloader: NllbDataLoader,
            eval_dataloader: NllbDataLoader,
    ):
        self.params = params
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.params.model_name).to(params.device)
        self.calc_loss = CalcLoss(
            num_classes=self.model.lm_head.weight.shape[0],  # use embedding layer shape to infer classes
            label_smoothing=params.label_smoothing,
        )

        log_model_info(self.model)

        self.train_data_loader = train_dataloader
        self.eval_data_loader = eval_dataloader

        self.grad_scaler = torch.cuda.amp.GradScaler()  # type: ignore
        self.optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.params.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-08,
            maximize=False,
            weight_decay=0.0,
            fused=(self.params.device.type == "cuda"),
        )

        self.lr_scheduler = transformers.get_inverse_sqrt_schedule(
            optimizer=self.optimizer,
            num_warmup_steps=self.params.warmup_steps,
        )

        self.epoch_idx: int = 0
        self.update_idx: int = 0
        self.patience_left: int = self.params.patience
        self.best_eval_loss: Optional[float] = None
        self.is_best_state: bool = False
        torch.set_float32_matmul_precision("high")

        self.__n_samples_since_last_time = 0
        self.__accumulate_step = 0
        self.__n_tokens_since_last_time = 0
        self.__steps_per_epoch = 0
        self.__train_loss_hist = []

    def run(self):
        logger.info("Start Finetuning")
        self._eval_model()

        train_dataloader = self.train_data_loader.get_dataloader()
        self.__steps_per_epoch = len(train_dataloader) // self.params.update_freq
        while self.epoch_idx < self.params.max_epochs and self.patience_left:
            for train_batch in train_dataloader:

                train_batch.to_device(self.params.device)
                one_step_finished = self._train_step_with_grad_accumulate(train_batch)

                # Perform eval if its time to eval
                if not one_step_finished:
                    continue
                if not self.update_idx or self.update_idx % self.params.eval_steps != 0:
                    continue

                # Clear GPU memory for eval
                torch.cuda.empty_cache()
                self._eval_model()

                # Save the current model if its the best we've ever had
                if self.is_best_state:
                    self._save_model()
                elif not self.patience_left:
                    no_improve_steps = self.params.eval_steps * self.params.patience
                    logger.info(
                        "Early termination, as eval loss did not improve "
                        f"over last {no_improve_steps} updates"
                    )
                    break

            self.epoch_idx += 1
        # do an evaluation at the end...
        torch.cuda.empty_cache()
        self._eval_model()

        # Save the current model if its the best we've ever had
        if self.is_best_state:
            self._save_model()

    def _forward_wrapper(self, batch: SeqsBatch) -> torch.Tensor:
        # embedding has shape: (bsz, seq, h=1024)
        input_embeds = self.model.model.encoder.embed_tokens(batch.src_tokens)

        with torch.autocast(device_type=self.params.device.type, dtype=self.params.float_dtype):
            logits = self.model.forward(
                inputs_embeds=input_embeds,
                attention_mask=batch.src_att_mask,
                decoder_input_ids=batch.tgt_inputs
            )["logits"]

        return logits

    def _train_step_with_grad_accumulate(self, batch: SeqsBatch) -> bool:
        """Return True if a step is finished; return False if not finished."""
        self.model.train()

        logits = self._forward_wrapper(batch)

        loss = self.calc_loss(batch, logits) / self.params.update_freq
        if loss.isnan().any().item():
            logger.error(batch)
            raise RuntimeError("Train loss is NaN! Something is wrong in the model!")

        # always compute loss and gradients
        self.grad_scaler.scale(loss).backward()

        self.__train_loss_hist.append(loss.cpu().item() * self.params.update_freq)

        self.__n_samples_since_last_time += batch.src_tokens.shape[0]  # assume (bsz, seq)
        self.__n_tokens_since_last_time += batch.src_tokens.numel()

        if self.__accumulate_step == self.params.update_freq - 1:
            # just the normal param update stuff
            self.__accumulate_step = 0

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            self._train_step_log()
            self.update_idx += 1
            return True
        else:
            # do nothing and only accumulate the gradients
            self.__accumulate_step += 1
            return False

    def _train_step_log(self) -> None:
        """Log train stats"""
        if (self.update_idx + 1) % self.params.log_steps == 0:
            avg_loss = np.mean(self.__train_loss_hist).item()
            del self.__train_loss_hist
            self.__train_loss_hist = []

            # just log the cuda usage for the main process
            max_cuda_mem_in_bytes = torch.cuda.max_memory_allocated(0)
            max_cuda_mem_in_gb = max_cuda_mem_in_bytes / 1024 / 1024 / 1024

            logger.info(
                f"Epoch {str(self.epoch_idx + 1).zfill(3)} ({self.__steps_per_epoch} steps) / "
                f"update {(self.update_idx + 1) % self.__steps_per_epoch} "
                f"total update {str(self.update_idx + 1).zfill(5)}: "
                f"train_loss={avg_loss:.4f} "
                f"last_lr={self.lr_scheduler.get_last_lr()[0]:.2E} "
                f"peak_cuda_mem={max_cuda_mem_in_gb:.2f}GB "
                f"bsz={self.__n_samples_since_last_time / self.params.log_steps:.1f} "
                f"n_tokens={self.__n_tokens_since_last_time / self.params.log_steps:.1f}"
            )
            torch.cuda.reset_peak_memory_stats(0)
            self.__n_samples_since_last_time = 0
            self.__n_tokens_since_last_time = 0

    @torch.no_grad()
    def _eval_model(self) -> None:
        """Calc avg loss on eval dataset and update evaluation stats"""

        logger.info(f"Evaluation Step {self.update_idx // self.params.eval_steps}...")
        loss_hist = []  # collection of losses
        self.model.eval()

        eval_dataloader = self.eval_data_loader.get_dataloader()
        for batch in eval_dataloader:
            batch: SeqsBatch
            batch.to_device(self.params.device)
            with torch.autocast(device_type=self.params.device.type, dtype=self.params.float_dtype):
                logits = self.model(
                    input_ids=batch.src_tokens, attention_mask=batch.src_att_mask,
                    decoder_input_ids=batch.tgt_inputs
                )["logits"]
                loss = self.calc_loss(batch, logits)
            if loss.isnan():
                logger.warning("Eval batch loss value is NaN, skipping")
                continue
            del batch  # force memory release
            loss_hist.append(loss.cpu().item())
        eval_loss = np.mean(loss_hist).item()
        self._update_eval_stats(eval_loss)

    def _update_eval_stats(self, eval_loss: float) -> None:
        self.is_best_state = (
                self.best_eval_loss is None or eval_loss < self.best_eval_loss
        )
        self.best_eval_loss = eval_loss if self.is_best_state else self.best_eval_loss
        self.patience_left = (
            self.params.patience if self.is_best_state else self.patience_left - 1
        )
        logger.info(
            f"Eval after {self.update_idx} updates: "
            f"loss={eval_loss:.4f} "
            f"best_loss={self.best_eval_loss:.4f} "
            f"patience_steps_left={self.patience_left} "
        )

    def _save_model(self) -> None:
        logger.info("Saving model")
        Path(self.params.save_model_path).parent.mkdir(parents=True, exist_ok=True)
        # ref: https://discuss.huggingface.co/t/saving-a-model-and-loading-it/21492
        # will overwrite previous ckpts
        self.model.save_pretrained(self.params.save_model_path, from_pt=True)
