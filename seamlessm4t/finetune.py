import argparse
from pathlib import Path

import tokenizers
import torch
import torch.nn as nn
import transformers
from transformers import AutoProcessor, SeamlessM4Tv2Model, SeamlessM4TModel, SeamlessM4Tv2ForTextToText, \
    SeamlessM4TForTextToText, SeamlessM4Tv2ForSpeechToText, SeamlessM4TForSpeechToText

from seamlessm4t.dataloader import MultiModalDataLoader, BatchingConfig
from seamlessm4t.enums import FinetuneMode, ModelVer, KdLossType
from seamlessm4t.model_utils import fix_model_dropout
from seamlessm4t.trainer import MultiModalFinetune, FinetuneParams
from utils.log_utils import logging
from utils.model_utils import log_model_info

logger = logging.getLogger(__name__)


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Example finetuning script for HF's Seamless model."
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
        required=False,
        help="Path to manifest with train samples. "
             "It should be either MT/ASR/ST or multi-task.",
    )
    parser.add_argument(
        "--eval_dataset",
        type=Path,
        required=True,
        help="Path to manifest with eval samples. Could be a single task (MT/ASR/ST) or a multi task one.",
    )
    # model setup
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/seamless-m4t-v2-large",
        help="Base model name (`facebook/seamless-m4t-v2-large`)",
    )
    parser.add_argument(
        "--load_from_mt_pretrained",
        required=False,
        type=str,
        help="Load pretrained MT components."
    )
    parser.add_argument(
        "--load_from_asr_pretrained",
        required=False,
        type=str,
        help="Load pretrained ASR encoder."
    )
    parser.add_argument(
        "--save_model_to",
        type=Path,
        required=True,
        help="Path to save best finetuned model.",
    )
    parser.add_argument(
        "--finetune_text_encoder",
        default=False,
        action="store_true",
        help="Only effective for Multitasking."
    )
    # processor setup
    parser.add_argument(
        "--processor_path",
        type=str,
        help="if provided, will load this processor. If not, use `model_name`."
    )
    parser.add_argument(
        "--save_processor_path",
        required=True,
        type=str,
        help="where to save the processor."
    )
    # opt
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Finetuning learning rate",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1,
        help="weight for student NLL loss. Only effective for multitasking."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1,
        help="weight for teacher NLL loss. Only effective for multitasking."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1,
        help="weight for KD loss. Only effective for multitasking."
    )
    parser.add_argument(
        "--kd_loss_type",
        type=KdLossType,
        choices=list(KdLossType),
        default=KdLossType.KLD,
        help="which loss to use for KD."
    )
    parser.add_argument(
        "--detach_teacher",
        default=False,
        action="store_true",
        help="whether to apply KD to the teacher side."
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
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
        "--max_text_tokens",
        type=int,
        default=128,
        help="Maximum number of src_tokens/tgt_tokens per sentence",
    )
    parser.add_argument(
        "--max_speech_dur",
        type=float,
        default=30.0,
        help="Max num of frames per speech batch."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to fine-tune on. See `torch.device`.",
    )
    parser.add_argument(
        "--mode",
        type=FinetuneMode,
        choices=list(FinetuneMode),
        default=FinetuneMode.SPEECH_TO_TEXT
    )
    parser.add_argument(
        "--eval_mode",
        type=FinetuneMode,
        choices=list(FinetuneMode),
        default=FinetuneMode.SPEECH_TO_TEXT,
        help="Which mode to perform at evaluation."
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
    # additional options to match the behavior of the official script
    parser.add_argument(
        "--untie_lm_head",
        action="store_true",
        default=False,
        help="Whether to disable parameter sharing for lm_head."
    )
    parser.add_argument(
        "--prefix_skip_len",
        type=int,
        default=0,
        help="Ignore losses for the prefix. Set to 1 to ignore lang code."
    )
    parser.add_argument(
        "--dropout_fix",
        action="store_true",
        default=False,
        help="fix a few dropout."
    )
    return parser


def try_to_add_new_lang_code(processor, model, lang_code: str, is_ver1_model: bool):
    tok_lang_code = f"__{lang_code}__"

    if tok_lang_code not in processor.tokenizer.get_added_vocab():
        logger.info(f"Adding {tok_lang_code} to tokenizer")
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [
            tokenizers.AddedToken(tok_lang_code,
                                  # true for v1; false for v2
                                  rstrip=is_ver1_model, lstrip=is_ver1_model,
                                  single_word=False, normalized=False, special=True)
        ]}, replace_additional_special_tokens=False)

    tok_id = processor.tokenizer.get_added_vocab()[tok_lang_code]

    if lang_code not in model.generation_config.text_decoder_lang_to_code_id:
        logger.info(f"Adding {lang_code} to model")
        model.generation_config.text_decoder_lang_to_code_id[lang_code] = tok_id
        # this should be fixed by https://github.com/huggingface/transformers/commit/bb1d0d0d9e7ca356cf5673031183e955cc160158
        # but I was using 4.49.0 which still has this issue
        # these two configs are not used anyways
        model.generation_config.t2u_lang_code_to_id[lang_code] = tok_id
        model.generation_config.vocoder_lang_code_to_id[lang_code] = tok_id


def _load_mt_components(model, pt_model):
    """
    Load MT components for model from pt_model.
    If model does not have a text encoder, then do not load it.
    """
    # load components
    if model.text_encoder is None:
        logger.info(f"Not loading MT Encoder")
    else:
        model.text_encoder.load_state_dict(pt_model.text_encoder.state_dict())
    model.text_decoder.load_state_dict(pt_model.text_decoder.state_dict())
    model.lm_head.load_state_dict(pt_model.lm_head.state_dict())

    # double check if loading was successful
    assert torch.allclose(model.shared.weight, pt_model.shared.weight), f"shared.weight wrong!"
    assert torch.allclose(model.lm_head.weight, pt_model.lm_head.weight), f"lm_head.weight wrong!"

    if model.text_encoder is not None:
        # check text encoder
        pt_text_encoder_state_dict = pt_model.text_encoder.state_dict()
        for name, param in model.text_encoder.named_parameters():
            assert torch.allclose(param, pt_text_encoder_state_dict[name]), f"{name} wrong!"
        del pt_text_encoder_state_dict

    # check text decoder
    pt_text_decoder_state_dict = pt_model.text_decoder.state_dict()
    for name, param in model.text_decoder.named_parameters():
        assert torch.allclose(param, pt_text_decoder_state_dict[name]), f"{name} wrong!"
    del pt_text_decoder_state_dict

    logger.info(f"Checked all loaded params.")
    return model


def _load_asr_encoder(model, pt_model):
    """
    Load model.speech_encoder from pt_model.
    """
    pt_speech_encoder_state_dict = pt_model.speech_encoder.state_dict()
    model.speech_encoder.load_state_dict(pt_speech_encoder_state_dict)

    for name, param in model.speech_encoder.named_parameters():
        assert torch.allclose(param, pt_speech_encoder_state_dict[name]), f"{name} wrong!"

    del pt_speech_encoder_state_dict
    logger.info(f"Checked loaded params.")
    return model


def main() -> None:
    args = init_parser().parse_args()
    logger.info(f"input args: {args}")

    seed = args.seed
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
        model_ver=ModelVer.VER_2 if "v2" in args.model_name else ModelVer.VER_1,
        finetune_mode=args.mode,
        eval_mode=args.eval_mode,
        finetune_text_encoder=args.finetune_text_encoder,
        train_batch_size=reduced_train_bsz,
        eval_batch_size=reduced_train_bsz,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        kd_loss_type=args.kd_loss_type,
        detach_teacher=args.detach_teacher,
        prefix_skip_len=args.prefix_skip_len,
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

    # 1. load model
    if finetune_params.model_name in ["facebook/hf-seamless-m4t-medium", "facebook/hf-seamless-m4t-large"]:
        model = SeamlessM4TModel.from_pretrained(finetune_params.model_name)
    else:
        assert finetune_params.model_name == "facebook/seamless-m4t-v2-large"
        model = SeamlessM4Tv2Model.from_pretrained(finetune_params.model_name)

    # remove speech output always
    model.t2u_model = None
    model.vocoder = None
    if finetune_params.finetune_mode == FinetuneMode.TEXT_TO_TEXT:
        model.speech_encoder = None
    elif finetune_params.finetune_mode == FinetuneMode.SPEECH_TO_TEXT:
        model.text_encoder = None

    if args.load_from_asr_pretrained:
        pt_model_class = SeamlessM4Tv2ForSpeechToText if finetune_params.model_ver == ModelVer.VER_2 else SeamlessM4TForSpeechToText
        pt_model = pt_model_class.from_pretrained(args.load_from_asr_pretrained)
        model = _load_asr_encoder(model, pt_model)
        logger.info(f"Loaded {pt_model_class} from {args.load_from_asr_pretrained}")
        del pt_model
        del pt_model_class

    if args.load_from_mt_pretrained:
        pt_model_class = SeamlessM4Tv2ForTextToText if finetune_params.model_ver == ModelVer.VER_2 else SeamlessM4TForTextToText
        pt_model = pt_model_class.from_pretrained(args.load_from_mt_pretrained)
        model = _load_mt_components(model, pt_model)
        logger.info(f"Loaded {pt_model_class} from {args.load_from_mt_pretrained}")
        del pt_model  # release mem
        del pt_model_class

    if finetune_params.finetune_mode == FinetuneMode.MULTI_TASKING:
        if not finetune_params.finetune_text_encoder:
            # freeze text encoder when multi-tasking
            assert model.text_encoder is not None
            # copy, so that it is unchanged when fine-tuning the decoder embeddings.
            model.text_encoder.embed_tokens.weight = nn.Parameter(model.shared.weight.clone().detach())
            for name, param in model.text_encoder.named_parameters():
                logger.info(f"{name}.requires_grad -> False")
                param.requires_grad = False

    # disable parameter sharing for LM Head.
    # this is to match the behaviour of the official fine-tuning script, which
    # accidentally breaks the parameter sharing.
    if args.untie_lm_head:
        logger.info(f"Will untie lm head")
        model.lm_head.weight = nn.Parameter(model.shared.weight.clone().detach())
        # to avoid weight typing in post-init
        model.config.tie_word_embeddings = False
        # just some sanity checks...
        assert torch.allclose(model.lm_head.weight, model.shared.weight)
        assert id(model.lm_head.weight) != id(model.shared.weight)

    if args.dropout_fix:
        fix_model_dropout(model)

    log_model_info(model)

    # 2. load processor_path
    processor_name_or_path = args.processor_path if args.processor_path else finetune_params.model_name
    logger.info(f"Load processor from {processor_name_or_path}")
    processor = AutoProcessor.from_pretrained(processor_name_or_path)

    # 3. deal with potentially new lang codes
    try_to_add_new_lang_code(processor, model, args.src_lang,
                             is_ver1_model=(finetune_params.model_ver == ModelVer.VER_1))
    try_to_add_new_lang_code(processor, model, args.tgt_lang,
                             is_ver1_model=(finetune_params.model_ver == ModelVer.VER_1))
    logger.info(f"Save process to {args.save_processor_path}")
    Path(args.save_processor_path).parent.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(args.save_processor_path)

    # 4. load data
    train_dataloader = MultiModalDataLoader(
        src_lang=args.src_lang, tgt_lang=args.tgt_lang,
        processor=processor,
        dataset_manifest_path=args.train_dataset,
        batching_config=BatchingConfig(
            max_text_tokens=args.max_text_tokens,
            max_speech_frames=int(args.max_speech_dur * 16000),
            batch_size=finetune_params.train_batch_size,
            float_dtype=finetune_params.float_dtype
        ),
        data_mode=finetune_params.finetune_mode
    )

    eval_dataloader = MultiModalDataLoader(
        src_lang=args.src_lang, tgt_lang=args.tgt_lang,
        processor=processor,
        dataset_manifest_path=args.eval_dataset,
        batching_config=BatchingConfig(
            max_text_tokens=args.max_text_tokens,
            max_speech_frames=int(args.max_speech_dur * 16000),
            batch_size=finetune_params.eval_batch_size,
            float_dtype=finetune_params.float_dtype,
            num_workers=1
        ),
        data_mode=finetune_params.eval_mode
    )

    trainer = MultiModalFinetune(
        model,
        finetune_params,
        train_dataloader,
        eval_dataloader
    )
    trainer.run()


if __name__ == '__main__':
    main()
