"""
Either Text-to-Text or Speech-to-Text.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile
import torch
from transformers import (AutoProcessor, SeamlessM4Tv2ForTextToText, SeamlessM4Tv2ForSpeechToText,
                          SeamlessM4TForTextToText, SeamlessM4TForSpeechToText)

from seamlessm4t.enums import ModelVer, FinetuneMode
from seamlessm4t.model_utils import fix_model_dropout
from utils.log_utils import logging, my_tqdm
from utils.text_utils import count_lines

logger = logging.getLogger(__name__)


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Example inference script for Seamless-HF models."
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
    # model_name and model_path are mutually exclusive.
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="The official huggingface name of the model."
             "When provided, will load BOTH model and tokenizer from this."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=False,
        help="if provided, load this ckpt."
    )
    parser.add_argument(
        "--model_ver",
        type=ModelVer,
        choices=list(ModelVer),
        default=ModelVer.VER_2
    )
    parser.add_argument(
        "--eval_mode",
        type=FinetuneMode,
        choices=[FinetuneMode.SPEECH_TO_TEXT, FinetuneMode.TEXT_TO_TEXT],
        default=FinetuneMode.SPEECH_TO_TEXT,
        help="Which mode to perform at evaluation."
    )
    parser.add_argument(
        "--processor_path",
        type=str,
        default=None,
        required=False,
        help="if provided, load this processor."
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
        "--max_text_tokens",
        type=int,
        default=256,
        help="Maximum number of src_tokens/tgt_tokens per sentence",
    )
    parser.add_argument(
        "--max_speech_dur",
        type=float,
        default=45.0,
        help="Max num of frames per speech batch."
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--dropout_fix",
        action="store_true",
        default=False,
        help="fix a few dropout."
    )
    return parser


def main() -> None:
    args = init_parser().parse_args()
    logger.info(f"input args: {args}")

    src_lang, tgt_lang = args.src_lang, args.tgt_lang
    logger.info(f"{src_lang=} {tgt_lang=}")

    device = torch.device("cuda")
    dtype = torch.float16

    decoding_configs = {
        "do_sample": False,
        "num_beams": args.beam_size,
        "max_new_tokens": args.max_text_tokens,  # ignores the length of prompt, although I don't have one
        "length_penalty": args.length_penalty
    }

    # load processor from path when provided
    processor_name = args.processor_path if args.processor_path else args.model_name
    processor = AutoProcessor.from_pretrained(processor_name)
    logger.info(f"Loaded processor from {processor_name}")

    # Load model from path when provided
    model_name = args.model_path if args.model_path else args.model_name
    # only load specific components
    model_ver = args.model_ver
    eval_mode = args.eval_mode
    model_class = None
    if model_ver == ModelVer.VER_1:
        if eval_mode == FinetuneMode.SPEECH_TO_TEXT:
            model_class = SeamlessM4TForSpeechToText
        if eval_mode == FinetuneMode.TEXT_TO_TEXT:
            model_class = SeamlessM4TForTextToText
    if model_ver == ModelVer.VER_2:
        if eval_mode == FinetuneMode.SPEECH_TO_TEXT:
            model_class = SeamlessM4Tv2ForSpeechToText
        if eval_mode == FinetuneMode.TEXT_TO_TEXT:
            model_class = SeamlessM4Tv2ForTextToText
    assert model_class is not None

    model = model_class.from_pretrained(model_name)
    # I may have set `self.config.tie_word_embeddings=False` to untie lm_head in multi-task ST training.
    # here load `shared` params into text decoder input embeddings
    if hasattr(model, "text_decoder"):
        logger.info(f"Tie decoder input embedding with model.shared")
        model._tie_or_clone_weights(model.text_decoder.embed_tokens, model.shared)
    if args.dropout_fix:
        fix_model_dropout(model)
    logger.info(f"Loaded {model_class} from {model_name}")

    model.to(dtype).to(device)
    model.eval()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tot = count_lines(args.eval_dataset) - 1  # not counting header
    max_speech_frames = int(16000 * args.max_speech_dur)
    with torch.no_grad():
        with open(args.eval_dataset) as in_fp, \
                open(output_path, mode="w") as out_fp:
            with torch.autocast(device_type=device.type, dtype=dtype):
                for item in my_tqdm(csv.DictReader(in_fp, delimiter="\t"), total=tot):
                    translated_tokens = None  # expected to be (1, seq)
                    try:
                        if eval_mode == FinetuneMode.SPEECH_TO_TEXT:
                            raw_audio, sr = soundfile.read(item["audio_path"], dtype="float32")
                            assert sr == 16000
                            if raw_audio.shape[0] > max_speech_frames:
                                logger.info(f"Truncate speech from {raw_audio.shape} to {max_speech_frames}")
                                raw_audio = raw_audio[:max_speech_frames]
                            raw_audio = np.expand_dims(raw_audio, axis=0)
                            speech_inputs = processor.feature_extractor(
                                raw_speech=[raw_audio, ],
                                return_attention_mask=True,
                                return_tensors="pt",
                                sampling_rate=16000  # it must be...
                            )
                            translated_tokens = model.generate(
                                input_features=speech_inputs["input_features"].to(dtype).to(device),
                                attention_mask=speech_inputs["attention_mask"].to(device),
                                tgt_lang=tgt_lang,
                                return_dict_in_generate=False,
                                **decoding_configs
                            )
                        elif eval_mode == FinetuneMode.TEXT_TO_TEXT:
                            model_inputs = processor.tokenizer(
                                text=[item["src_text"], ],
                                src_lang=src_lang,
                                return_tensors="pt",
                                truncation=True,
                                return_attention_mask=True,
                                max_length=args.max_text_tokens
                            )
                            translated_tokens = model.generate(
                                input_ids=model_inputs["input_ids"].to(device),
                                attention_mask=model_inputs["attention_mask"].to(device),
                                tgt_lang=tgt_lang,
                                return_dict_in_generate=False,
                                **decoding_configs
                            )
                        else:
                            raise ValueError(f"Unsupported {eval_mode}")

                        assert translated_tokens is not None
                        hypo_str = processor.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                    except RuntimeError as e:
                        logger.warning(f"Got {e} for {item}")
                        logger.warning(f"Will use empty string")
                        hypo_str = ""

                    out_fp.write(hypo_str + "\n")

    logger.info(f"Done!")


if __name__ == '__main__':
    main()
