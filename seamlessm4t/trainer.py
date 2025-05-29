import contextlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import transformers
from torch.optim import AdamW
from torch.types import Device
from transformers.models.seamless_m4t.modeling_seamless_m4t import \
    _compute_new_attention_mask as _compute_new_attention_mask_v1
from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import \
    _compute_new_attention_mask as _compute_new_attention_mask_v2

from seamlessm4t.dataloader import MultiModalBatch, MultiModalDataLoader
from seamlessm4t.enums import FinetuneMode, ModelVer, KdLossType
from utils.log_utils import logging

logger = logging.getLogger(__name__)


@dataclass
class FinetuneParams:
    model_name: str
    """Model name of model being finetuned."""

    save_model_path: Path
    """Path were to save finetuned model."""

    model_ver: ModelVer = ModelVer.VER_1
    """Which version of model."""

    finetune_mode: FinetuneMode = FinetuneMode.SPEECH_TO_TEXT
    """Allows to remove parts from the model"""

    eval_mode: FinetuneMode = FinetuneMode.SPEECH_TO_TEXT
    """The mode to perform during evaluation."""

    finetune_text_encoder: bool = False
    """ Only for multi-tasking. Whether to freeze text encoder. """

    train_batch_size: int = 5
    """The batch size during train steps"""

    eval_batch_size: int = 5
    """The batch size during evaluation."""

    learning_rate: float = 1e-5
    """ Optimizer learining rate """

    alpha: float = 1.0
    """ Only for multi-tasking. Weight for `nll`. """

    beta: float = 1.0
    """ Only for multi-tasking. Weight for `teacher_nll`. """

    gamma: float = 1.0
    """ Only for multi-tasking. Weight for `kd`"""

    kd_loss_type: KdLossType = KdLossType.KLD
    """ Only for KD. Loss for KD. """

    detach_teacher: bool = False
    """ Only for KD. Whether apply KD loss to the teacher side. """

    label_smoothing: float = 0.2  # todo: may want to tune this as well
    """ Label smoothing coefficient for nll_loss """

    prefix_skip_len: int = 0
    """ Ignore losses for the prefix. 1 if skip lang code. """

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
    """ Maximum number of trainign epochs"""

    float_dtype: torch.dtype = torch.float16
    """Float Dtype"""

    device: Device = torch.device("cuda")
    """ Where to run computation"""


class CalcLoss:
    """
    Calculate NLL loss for ST/MT.
    Optionally computes KL-divergence between student logits and teacher logits.
    """

    def __init__(
            self,
            num_classes: int,
            label_smoothing: float,
            kd_loss_type: KdLossType,
            detach_teacher: bool = False,
            prefix_skip_len: int = 0
    ):
        """
        :param detach_teacher: whether call teacher_logits.detach().
        :param prefix_skip_len: will set padding_mask[:, :prefix_skip_len]=0, such that the prefix's loss is ignored.
        """
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        eps = label_smoothing / num_classes
        self.negative = eps
        self.positive = (1.0 - label_smoothing) + eps

        self.kd_loss_type = kd_loss_type
        self.detach_teacher = detach_teacher
        self.kl_div = torch.nn.KLDivLoss(reduction="none")

        self.prefix_skip_len = prefix_skip_len
        logger.info(f"CalcLoss: {self.num_classes} classes "
                    f"neg={self.negative} | pos={self.positive} "
                    f"Kd Loss Type: {kd_loss_type} "
                    f"Detach Teacher: {self.detach_teacher} "
                    f"prefix skip len: {self.prefix_skip_len}"
                    )

    def __call__(self, batch: MultiModalBatch, text_logits: torch.Tensor,
                 teacher_logits: Optional[torch.Tensor] = None):

        assert self.num_classes == text_logits.shape[-1], f"{self.num_classes} != {text_logits.shape}"

        tgt_att_mask = batch.tgt_att_mask
        if self.prefix_skip_len > 0:
            tgt_att_mask[:, :self.prefix_skip_len].fill_(0)

        # logits -> log prob; (bsz, seq, vocab_size)
        text_log_softmax = text_logits.log_softmax(dim=2)

        # target distribution; (bsz, seq, vocab_size)
        target = torch.zeros_like(text_log_softmax).to(text_log_softmax.device)
        target.fill_(self.negative)
        # target_tokens: (bsz, seq) -> (bsz, seq, 1)
        target.scatter_(2, batch.tgt_outputs.unsqueeze(2), self.positive)

        # nll loss for log_softmax
        # attention mask: bsz, seq
        nll_loss = torch.sum(-target * text_log_softmax, dim=2) * tgt_att_mask
        # tok-level average: bsz, seq -> 1
        nll_loss = torch.sum(nll_loss) / torch.sum(tgt_att_mask)

        # 0. student nll loss; always do this
        ret_dict = {
            "nll": nll_loss,
        }

        if teacher_logits is not None:
            # 1. teacher nll loss
            # logits -> log prob; (bsz, seq, vocab_size)
            teacher_log_prob = teacher_logits.log_softmax(dim=2)
            # bsz, seq; sum for each position
            teacher_nll_loss = torch.sum(-target * teacher_log_prob, dim=2) * tgt_att_mask
            # reduce at token-level
            teacher_nll_loss = torch.sum(teacher_nll_loss) / torch.sum(tgt_att_mask)
            ret_dict["teacher_nll"] = teacher_nll_loss

            # 2.1 token-level KD; (bsz, seq, vocab_size)
            if self.detach_teacher:
                teacher_logits = teacher_logits.detach()
            if self.kd_loss_type == KdLossType.KLD:
                # p_teacher * (log p_tea - log p_stu)
                kd_loss = self.kl_div(
                    text_log_softmax, teacher_logits.softmax(dim=2)
                )
            elif self.kd_loss_type == KdLossType.XENT:
                # - p_teacher * log p_stu
                kd_loss = -teacher_logits.softmax(dim=2) * text_log_softmax
            else:
                raise ValueError(f"Unsupported {self.kd_loss_type}")

            # 2.2 average token-level loss to scalar
            # sum for each token -> token-level loss. (bsz, seq)
            kd_loss = torch.sum(kd_loss, dim=2)
            kd_loss = kd_loss * tgt_att_mask  # not computing loss on paddings
            # tok-level average
            kd_loss = torch.sum(kd_loss) / torch.sum(tgt_att_mask)
            ret_dict["kd"] = kd_loss

        return ret_dict


class MultiModalFinetune:
    def __init__(
            self,
            model,
            params: FinetuneParams,
            train_dataloader: MultiModalDataLoader,
            eval_dataloader: MultiModalDataLoader
    ):
        self.model = model.to(params.device)
        self.params = params

        self.loss_fun = CalcLoss(
            num_classes=model.shared.weight.shape[0],
            label_smoothing=params.label_smoothing,
            kd_loss_type=params.kd_loss_type,
            detach_teacher=params.detach_teacher,
            prefix_skip_len=params.prefix_skip_len
        )

        self.train_data_loader = train_dataloader
        self.eval_data_loader = eval_dataloader

        self.grad_scaler = torch.cuda.amp.GradScaler()  # type: ignore

        model_params = self.model.parameters()

        self.optimizer = AdamW(
            params=model_params,
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

        self._n_samples_since_last_time = 0
        self._accumulate_step = 0
        self._n_tokens_since_last_time = 0
        self._steps_per_epoch = 0
        self._train_loss_hist = defaultdict(list)

    def run(self):
        logger.info("Start Finetuning")
        self._eval_model()

        train_dataloader = self.train_data_loader.get_dataloader()
        self._steps_per_epoch = len(train_dataloader) // self.params.update_freq
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
        # evaluation at the end...
        torch.cuda.empty_cache()
        self._eval_model()

        # Save the current model if its the best we've ever had
        if self.is_best_state:
            self._save_model()

    def _train_step_with_grad_accumulate(self, batch: MultiModalBatch) -> bool:
        """Return True if a step is finished; return False if a step is not finished."""
        self.model.train()
        if self.params.finetune_mode == FinetuneMode.MULTI_TASKING:
            # the frozen components should be at eval().
            if not self.params.finetune_text_encoder:
                self.model.text_encoder.eval()

        logits = self._model_forward(batch, mode=self.params.finetune_mode)

        loss_dict = self._cal_loss(batch, logits, mode=self.params.finetune_mode)
        train_loss = self._summarize_loss(loss_dict, mode=self.params.finetune_mode)

        train_loss = train_loss / self.params.update_freq  # average
        if train_loss.isnan().any().item():
            logger.error(batch)
            raise RuntimeError("Train loss is NaN! Something is wrong in the model!")

        # always compute loss and gradients
        self.grad_scaler.scale(train_loss).backward()

        for loss_key in loss_dict:
            # here just use the raw loss
            self._train_loss_hist[loss_key].append(loss_dict[loss_key].cpu().item())

        if batch.src_sp_tokens is not None:
            self._n_samples_since_last_time += batch.src_sp_tokens.shape[0]  # assume (bsz, seq)
            self._n_tokens_since_last_time += batch.src_sp_tokens.numel()
        else:
            self._n_samples_since_last_time += batch.src_text_tokens.shape[0]
            self._n_tokens_since_last_time += batch.src_text_tokens.numel()

        if self._accumulate_step == self.params.update_freq - 1:
            # just the normal param update stuff
            self._accumulate_step = 0

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            self._train_step_log()
            self.update_idx += 1
            return True
        else:
            # do nothing and only accumulate the gradients
            self._accumulate_step += 1
            return False

    def _train_step_log(self) -> None:
        """Log train stats"""
        if (self.update_idx + 1) % self.params.log_steps == 0:
            train_loss_dict = {
                key: np.mean(self._train_loss_hist[key]).item()
                for key in self._train_loss_hist.keys()
            }
            del self._train_loss_hist
            self._train_loss_hist = defaultdict(list)

            # just log the cuda usage for the main process
            max_cuda_mem_in_bytes = torch.cuda.max_memory_allocated(0)
            max_cuda_mem_in_gb = max_cuda_mem_in_bytes / 1024 / 1024 / 1024

            log_str = (f"Epoch {str(self.epoch_idx + 1).zfill(3)} ({self._steps_per_epoch} steps) / "
                       f"update {(self.update_idx + 1) % self._steps_per_epoch} "
                       f"total update {str(self.update_idx + 1).zfill(5)}: "
                       f"train_loss={self._summarize_loss(train_loss_dict, mode=self.params.finetune_mode):.4f} "
                       f"last_lr={self.lr_scheduler.get_last_lr()[0]:.2E} "
                       f"peak_cuda_mem={max_cuda_mem_in_gb:.2f}GB "
                       f"bsz={self._n_samples_since_last_time / self.params.log_steps:.1f} "
                       f"n_tokens={self._n_tokens_since_last_time / self.params.log_steps:.1f} "
                       )

            if len(train_loss_dict) > 1:
                loss_decompose_log = f"nll_loss={train_loss_dict['nll']:.4f} " \
                                     f"teacher_nll={train_loss_dict['teacher_nll']:.4f} " \
                                     f"kd_loss={train_loss_dict['kd']:.4f} "
                log_str += loss_decompose_log

            logger.info(log_str)
            torch.cuda.reset_peak_memory_stats(0)
            self._n_samples_since_last_time = 0
            self._n_tokens_since_last_time = 0

    @torch.no_grad()
    def _eval_model(self) -> None:
        """Calc avg loss on eval dataset and update evaluation stats"""

        logger.info(f"Evaluation Step {self.update_idx // self.params.eval_steps}...")
        loss_hist = defaultdict(list)  # collection of losses
        self.model.eval()

        eval_dataloader = self.eval_data_loader.get_dataloader()
        for batch in eval_dataloader:
            batch: MultiModalBatch
            batch.to_device(self.params.device)
            with torch.autocast(device_type=self.params.device.type, dtype=self.params.float_dtype):
                logits = self._model_forward(batch, mode=self.params.eval_mode)
                loss = self._cal_loss(batch, logits, mode=self.params.eval_mode)

            has_nan_loss = False
            for loss_key in loss:
                if loss[loss_key].isnan():
                    logger.warning("Eval batch loss value is NaN, skipping")
                    has_nan_loss = True
                    break
            if has_nan_loss:
                continue
            del batch  # force memory release

            for loss_key in loss:
                loss_hist[loss_key].append(loss[loss_key].cpu().item())

        eval_loss = defaultdict(float)
        for loss_key in loss_hist.keys():
            eval_loss[loss_key] = np.mean(loss_hist[loss_key]).item()
        self._update_eval_stats(eval_loss)

    def _update_eval_stats(self, eval_loss: Dict[str, float]) -> None:
        tot_eval_loss = self._summarize_loss(eval_loss, mode=self.params.eval_mode)

        self.is_best_state = (
                self.best_eval_loss is None or tot_eval_loss < self.best_eval_loss
        )
        self.best_eval_loss = tot_eval_loss if self.is_best_state else self.best_eval_loss
        self.patience_left = (
            self.params.patience if self.is_best_state else self.patience_left - 1
        )

        log_str = f"Eval after {self.update_idx} updates: " \
                  f"tot_loss={tot_eval_loss:.4f} " \
                  f"best_loss={self.best_eval_loss:.4f} " \
                  f"patience_steps_left={self.patience_left} "

        if self.params.eval_mode == FinetuneMode.MULTI_TASKING:
            loss_decompose_str = f"nll={eval_loss['nll']:4f} "
            loss_decompose_str += f"teacher_nll={eval_loss['teacher_nll']:.4f} "
            loss_decompose_str += f"kd_loss={eval_loss['kd']:.4f} "
            log_str += loss_decompose_str

        logger.info(log_str)

    def _cal_loss(self, batch: MultiModalBatch, logits: dict, mode: FinetuneMode) -> dict:
        if mode in [FinetuneMode.SPEECH_TO_TEXT, FinetuneMode.TEXT_TO_TEXT]:
            return self.loss_fun(batch, text_logits=logits["logits"])
        else:
            return self.loss_fun(
                batch,
                text_logits=logits["speech_to_text_logits"],
                teacher_logits=logits["text_to_text_logits"]
            )

    def _summarize_loss(self, loss: dict, mode: FinetuneMode) -> torch.Tensor:
        if mode in [FinetuneMode.SPEECH_TO_TEXT, FinetuneMode.TEXT_TO_TEXT]:
            sum_loss = loss["nll"]
        else:
            sum_loss = self.params.alpha * loss["nll"] + \
                       self.params.beta * loss["teacher_nll"] + \
                       self.params.gamma * loss["kd"]
        return sum_loss

    def _model_forward(self, batch: MultiModalBatch, mode: FinetuneMode) -> dict:
        """
        Helper method for running forward. Return a dict of loss.
        """
        if mode == FinetuneMode.MULTI_TASKING:
            return self._multi_task_forward(batch)
        if mode == FinetuneMode.SPEECH_TO_TEXT:
            return self._speech_to_text_forward(batch)
        if mode == FinetuneMode.TEXT_TO_TEXT:
            return self._text_to_text_forward(batch)

        raise ValueError(f"Unsupported {mode}")

    def _multi_task_forward(self, batch: MultiModalBatch) -> dict:
        """
        Text encoder should not have gradients.
        While speech encoder and text decoder follows the caller's context.
        """
        # cannot do with torch.no_grad() to decoder. ignore its gradients by manipulating the optimizer.
        speech_to_text_logits = self._speech_to_text_forward(batch)["logits"]

        t_enc_ctx = torch.no_grad() if not self.params.finetune_text_encoder else contextlib.nullcontext()
        with t_enc_ctx:
            text_enc_output = self._text_encoder_forward(batch)

        # todo: use decoder.eval() as teacher?
        text_to_text_logits = self._text_decoder_forward(batch, encoder_hidden_states=text_enc_output)

        return {
            "speech_to_text_logits": speech_to_text_logits,
            "text_to_text_logits": text_to_text_logits
        }

    def _speech_to_text_forward(self, batch: MultiModalBatch) -> dict:
        """
        https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L2959
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py#L3248
        """
        speech_enc_output, speech_enc_att_mask = self._speech_encoder_forward(batch)
        logits = self._text_decoder_forward(batch, encoder_hidden_states=speech_enc_output,
                                            src_speech_attention_mask=speech_enc_att_mask)
        return {
            "logits": logits
        }

    def _text_to_text_forward(self, batch: MultiModalBatch) -> dict:
        """
        https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/models/seamless_m4t/modeling_seamless_m4t.py#L2698
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py#L2978
        """
        encoder_last_hidden_state = self._text_encoder_forward(batch)
        logits = self._text_decoder_forward(batch, encoder_hidden_states=encoder_last_hidden_state)
        return {
            "logits": logits
        }

    def _text_encoder_forward(self, batch: MultiModalBatch) -> torch.Tensor:
        encoder_outputs = self.model.text_encoder(
            input_ids=batch.src_text_tokens,
            attention_mask=batch.src_text_att_mask,
            output_attentions=False, output_hidden_states=False,
            return_dict=True
        )
        # bsz, seq, 1024
        return encoder_outputs["last_hidden_state"]

    def _text_decoder_forward(self,
                              batch: MultiModalBatch,
                              encoder_hidden_states: torch.Tensor,
                              src_speech_attention_mask: Optional[torch.Tensor] = None
                              ) -> torch.Tensor:

        # bsz, seq, 1024
        decoder_outputs = self.model.text_decoder(
            input_ids=batch.tgt_inputs,
            attention_mask=batch.tgt_att_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=src_speech_attention_mask if src_speech_attention_mask is not None else batch.src_text_att_mask,
            output_attentions=False, output_hidden_states=False,
            return_dict=True
        )
        # bsz, seq, vocab size
        lm_logits = self.model.lm_head(decoder_outputs["last_hidden_state"])
        return lm_logits

    def _speech_encoder_forward(self, batch: MultiModalBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.model.speech_encoder(
            input_features=batch.src_sp_tokens,
            attention_mask=batch.src_sp_att_mask,
            output_attentions=False, output_hidden_states=False,
            return_dict=True
        )["last_hidden_state"]

        sub_sampled_lengths = self.model._compute_sub_sample_lengths_from_attention_mask(batch.src_sp_att_mask).to(
            encoder_output.device
        )

        _compute_new_attention_mask = _compute_new_attention_mask_v1 if self.params.model_ver == ModelVer.VER_1 \
            else _compute_new_attention_mask_v2

        new_encoder_attention_mask = _compute_new_attention_mask(
            hidden_states=encoder_output, seq_lens=sub_sampled_lengths
        )
        return encoder_output, new_encoder_attention_mask

    def _save_model(self) -> None:
        logger.info("Saving model")
        Path(self.params.save_model_path).parent.mkdir(parents=True, exist_ok=True)
        # ref: https://discuss.huggingface.co/t/saving-a-model-and-loading-it/21492
        # will overwrite previous ckpts
        self.model.save_pretrained(self.params.save_model_path, from_pt=True)
