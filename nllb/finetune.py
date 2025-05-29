import argparse
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

from nllb.dataloader import NllbDataLoader, BatchingConfig
from nllb.trainer import NllbFinetune, FinetuneParams
from utils.log_utils import logging

logger = logging.getLogger(__name__)


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Example finetuning script for NLLB models."
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        required=True
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        required=True
    )
    # data setup
    parser.add_argument(
        "--train_dataset",
        type=Path,
        required=True,
        help="Path to manifest with train samples",
    )
    parser.add_argument(
        "--eval_dataset",
        type=Path,
        required=True,
        help="Path to manifest with eval samples",
    )
    # model setup
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/nllb-200-distilled-1.3B",
        help="Base model name (`facebook/nllb-200-distilled-1.3B`, `facebook/nllb-200-distilled-600M`)",
    )
    parser.add_argument(
        "--save_model_to",
        type=Path,
        required=True,
        help="Path to save best finetuned model",
    )
    # opt
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-7,
        help="Finetuning learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of steps with linearly increasing learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--update_freq",
        type=int,
        default=1,
        help="gradient accumulate."
    )
    parser.add_argument(
        "--max_src_tokens",
        type=int,
        default=128,
        help="Maximum number of src_tokens per sentence, used to avoid GPU OOM and maximize the effective batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to fine-tune on. See `torch.device`.",
    )
    # training time
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help=(
            "Set early termination after `patience` number of evaluations "
            "without eval loss improvements"
        ),
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Max number of training epochs",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="Get eval loss after each `eval_steps` training steps ",
    )
    # aux
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help="Log inner loss after each `log_steps` training steps",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Randomizer seed value",
    )
    return parser


def main() -> None:
    args = init_parser().parse_args()
    logger.info(args)

    seed = args.seed
    # DO NOT set this: https://discuss.pytorch.org/t/assigning-tensor-to-multiple-rows-on-gpu/154421/2
    # transformers.enable_full_determinism(seed)
    transformers.set_seed(seed)
    logger.info(f"Set seed to {seed}")

    # 0. process arguments
    float_dtype = torch.float16 if torch.device(args.device).type != "cpu" else torch.bfloat16

    update_freq = args.update_freq
    reduced_train_bsz = args.batch_size // update_freq
    logger.info(f"Will use real batch size {reduced_train_bsz} with gradient accumulation {update_freq}")
    logger.info(f"The effective batch size is {reduced_train_bsz * update_freq}")

    finetune_params = FinetuneParams(
        model_name=args.model_name,
        save_model_path=args.save_model_to,
        train_batch_size=reduced_train_bsz,
        eval_batch_size=reduced_train_bsz,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        patience=args.patience,
        max_epochs=args.max_epochs,
        log_steps=args.log_steps,
        device=torch.device(args.device),
        float_dtype=float_dtype,
        update_freq=update_freq,
    )
    logger.info(f"Finetune Params: {finetune_params}")

    # 1. load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(finetune_params.model_name)

    # 2. load data
    train_dataloader = NllbDataLoader(
        src_lang=args.src_lang, tgt_lang=args.tgt_lang,
        text_tokenizer=tokenizer,
        dataset_manifest_path=args.train_dataset,
        batching_config=BatchingConfig(
            batch_size=finetune_params.train_batch_size,
            float_dtype=finetune_params.float_dtype
        )
    )

    eval_dataloader = NllbDataLoader(
        src_lang=args.src_lang, tgt_lang=args.tgt_lang,
        text_tokenizer=tokenizer,
        dataset_manifest_path=args.eval_dataset,
        batching_config=BatchingConfig(
            batch_size=finetune_params.eval_batch_size,
            float_dtype=finetune_params.float_dtype,
            num_workers=1
        )
    )

    trainer = NllbFinetune(
        params=finetune_params,
        train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
    )
    trainer.run()


if __name__ == '__main__':
    main()
