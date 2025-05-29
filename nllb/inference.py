import argparse
import csv
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils.log_utils import logging, my_tqdm
from utils.text_utils import count_lines

logger = logging.getLogger(__name__)


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Example inference script for NLLB models."
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
    parser.add_argument(
        "--eval_dataset",
        type=str,
        required=True,
        help="Path to manifest with eval samples",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="name of the model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=False,
        help="if provided, load this ckpt. if not provided, 0-shot."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="save the hypo here."
    )
    # decoding configs
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="beam search size."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="maximum tokens."
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=0.0
    )
    return parser


def main() -> None:
    args = init_parser().parse_args()
    logger.info(args)

    src_lang, tgt_lang = args.src_lang, args.tgt_lang
    model_name = args.model_name

    device = torch.device("cuda")

    decoding_configs = {
        "do_sample": False,
        "num_beams": args.beam_size,
        "max_length": args.max_length,
        "length_penalty": args.length_penalty
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)

    model_path = args.model_path
    if model_path is None:
        logger.info(f"Use 0-shot {model_name}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        logger.info(f"Load {model_name} from {model_path}")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    model.to(device)
    model.eval()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tot = count_lines(args.eval_dataset) - 1  # first line is header
    with torch.autocast(device_type=device.type, dtype=torch.float16):
        with torch.no_grad():
            with open(args.eval_dataset) as in_fp, \
                    open(output_path, mode="w") as out_fp:
                for item in my_tqdm(csv.DictReader(in_fp, delimiter="\t"), total=tot):
                    src_sent = item["src_text"]
                    inputs = tokenizer(src_sent, return_tensors="pt")
                    translated_tokens = model.generate(
                        input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device),
                        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                        **decoding_configs
                    )
                    hypo_str = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                    out_fp.write(hypo_str + "\n")

    logger.info(f"Done!")


if __name__ == '__main__':
    main()
