import torch.nn as nn

from utils.log_utils import logging

logger = logging.getLogger(__name__)


def log_model_info(model: nn.Module):
    logger.info(f"Model:\n{model}")
    n_trainable = n_total = 0
    for p in model.parameters():
        if p.requires_grad:
            n_trainable += p.numel()
        n_total += p.numel()
    logger.info(f"Total num params: {n_total / 1_000_000:.2f}M")
    logger.info(f"Trainable num params: {n_trainable / 1_000_000:.2f}M")
