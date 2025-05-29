# Codebase for GMU's submission to IWSLT 2025 Low-Resource Speech Translation Shared Task

> [**GMU Systems for the IWSLT 2025 Low-Resource Speech Translation Shared Task**](https://arxiv.org/abs/2505.21781)

## Introduction
This repository contains the codebase for GMU's submission to IWSLT 2025 Low-Resource Speech Translation Shared Task.
It supports fine-tuning the HuggingFace [NLLB](https://huggingface.co/docs/transformers/en/model_doc/nllb) and [SeamlessM4T](https://huggingface.co/docs/transformers/main/model_doc/seamless_m4t_v2) checkpoints.

**Features:**
- For NLLB, MT fine-tuning is supported.
- For SeamlessM4T, ASR, MT, and E2E-ST are supported. If 3-way ST data is available, multitask training can be supported.
  - Additionally, we support fine-tuning on new languages that are not supported by the official model. In other words, new language codes can be added.
- For both models, gradient accumulation can be used for larger batch sizes.

However, currently the codebase only supports single-gpu training, which we found sufficient.
If you need multi-gpu/multi-node training, you may consider taking a look at the [official SeamlessM4T repository](https://github.com/facebookresearch/seamless_communication), or [HuggingFace Accelerate](https://huggingface.co/docs/transformers/v4.51.3/accelerate).

## Usage

### Installation
We tested with Python 3.10, CUDA 12.6,  and the following packages:
```
pip install torch==2.2.2
pip install transformers==4.49.0
pip install numpy==1.26.4
pip install sentencepiece==0.2.0 google==3.0.0 protobuf==6.30.2
pip install soundfile
```

### Data Preparation
The input metadata is a `tsv` file, where the first line contains headers and the rest contains data samples.
Fields at each line are separated by `\t`.

For audios, please make sure they have a sampling rate of 16khz.
They should have a format that [soundfile](https://pypi.org/project/soundfile/) supports, e.g., `FLAC`, `WAV`.

Below are examples for different tasks.
Please ignore the spaces around `\t` in the following examples - they are used only for better visualization.
- MT task (for both NLLB and SeamlessM4T):
  ```
  src_text \t tgt_text
  src_sent_1 \t tgt_sent_1
  src_sent_2 \t tgt_sent_2
  ...
  ```
- ASR task:
  ```
  audio_path \t tgt_text
  path_1 \t transcription_1
  path_2 \t transcription_2
  ...
  ```
- E2E ST task:
  ```
  audio_path \t tgt_text
  path_1 \t translation_1
  path_2 \t translation_2
  ...
  ```
- Multitask ST task:
  ```
  audio_path \t src_text \t tgt_text
  path_1 \t transcription_1 \t translation_1
  path_2 \t transcription_2 \t translation_2
  ...
  ```
  
### Example and Computing Resources
We provide examples of fine-tuning on the `bem-eng` dataset [BIG-C](https://aclanthology.org/2023.acl-long.115/).

We provide training logs and (public) test set hypothesis in [examples/big_c](./examples/big_c).
The first column of hypothesis is utterance id (or `audio_id` in BIG-C).
- Fine-tuning requires less than or about 20GB of memory and a single A100-80GB GPU. 
It takes less than 1 day.
- Inference require less than or about 10GB of memory and less than 10GB GPU memory.

### NLLB
#### Fine-tune
```
python -m nllb.finetune \
    --src_lang bem_Latn --tgt_lang eng_Latn \
    --train_dataset /path/to/train/tsv --eval_dataset /path/to/dev/tsv \
    --model_name facebook/nllb-200-distilled-1.3B \
    --save_model_to /dir/to/save/checkpoints \
    --learning_rate 1e-4 --warmup_steps 1000 --max_epochs 10 \
    --batch_size 256 --update_freq 8 --max_src_tokens 128 \
    --eval_steps 250 --patience 10 \
    --log_steps 50
```
Explanation of arguments:
- **Language codes** (`src_lang` and `tgt_lang`): please use the ones listed in the [paper](https://arxiv.org/pdf/2207.04672) (Section 3).
- **Gradient accumulation** (`batch_size` and `update_freq`):
`batch_size` refers to the "effective" number of samples per step. 
`batch_size // update_freq` is the "actual" number of samples used in a single `forward()` function call, and it is passed to `dataloader`s.
`loss.backward()` is called every `update_freq` of forward calls.
In other words, one step is equivalent to `update_freq` of forward calls.
- `--patience`: if the dev loss is not improved for this number of steps, training will terminate.

_Log_: [examples/big_c/nllb/mt.train.log](./examples/big_c/nllb/mt.train.log)

#### Inference
```
python -m nllb.inference \
    --src_lang bem_Latn --tgt_lang eng_Latn \
    --eval_dataset /path/to/test/tsv \
    --model_name facebook/nllb-200-distilled-1.3B \
    --model_path /dir/to/saved/checkpoint \
    --output_path /path/to/output \
    --beam_size 5 --length_penalty 1.0
```
- The input file (`eval_dataset`) should be a `tsv` file containing one field of `src_text`.
The output file will contain lines of hypothesis.
- If `--model_path` is not provided, it uses the `model_name` model in a zero-shot manner.

_Test Set Hypothesis_: [examples/big_c/nllb/mt.hypo.tsv](./examples/big_c/nllb/mt.hypo.tsv)

_Metrics_: BLEU = 29.85, chrF2++ = 53.43

### SeamlessM4T

Explanation of common arguments:
- **Language codes** (`src_lang` and `tgt_lang`): please use the ones listed in the [paper](https://dl.fbaipublicfiles.com/seamless/seamless_m4t_paper.pdf) (Section 2.3). Should be three letters.
  - New language codes are supported. They will be added to the vocab as special tokens. The word embeddings contain unused entries, so the new language code can simply use an unused entry without resizing the embedding weight.
- **Gradient accumulation** (`batch_size` and `update_freq`): Same as the usage in NLLB.
- **patience**: Same as the usage in NLLB.
- **max_src_tokens**: the maximum number of input and output tokens. Long sentences will be truncated to this length.
- **max_speech_dur**: the maximum duration of input speech, in seconds. Long speech will be truncated to this length.

Explanation of matching the official SeamlessM4T codebase:
- As we describe in Appendix A of our paper, there are three discrepancies between this codebase and the [official one](https://github.com/facebookresearch/seamless_communication/tree/main/src/seamless_communication/cli/m4t/finetune).
  - **Loss on the target language code.** By default, we _do_ compute loss on the target language code (`--prefix_skip_len 0`).
To match the behavior of the official codebase, use `--prefix_skip_len 1`.
  - **Parameter sharing of word embeddings.** Three word embeddings are shared in the HuggingFace model. However, the `lm_head` in the official codebase is untied accidentally due to [this line](https://github.com/facebookresearch/fairseq2/blob/v0.2.0/src/fairseq2/models/utils/generic_loaders.py#L274).
To match this behavior, use `--untie_lm_head`.
  - **Dropout modules.** To (almost) match the dropout behaviors of the official codebase, use `--dropout_fix`.

#### MT Fine-tune
```
python -m seamlessm4t.finetune \
    --src_lang bem --tgt_lang eng \
    --train_dataset /path/to/train/tsv --eval_dataset /path/to/dev/tsv \
    --model_name facebook/seamless-m4t-v2-large \
    --save_model_to /dir/to/model --save_processor_path /dir/to/proc \
    --mode TEXT_TO_TEXT --eval_mode TEXT_TO_TEXT \
    --learning_rate 1e-4 --warmup_steps 1000 --max_epochs 10 \
    --batch_size 256 --update_freq 8 --max_text_tokens 128 \
    --eval_steps 250 --patience 10 \
    --log_steps 50 \
    [--dropout_fix]
```

_Log_: [examples/big_c/seamlessm4t/mt.train.log](./examples/big_c/seamlessm4t/mt.train.log)

#### MT Inference
Use `--dropout_fix` if you used it in training.
```
python -m seamlessm4t.inference \
    --src_lang bem --tgt_lang eng \
    --eval_dataset /path/to/test/tsv \
    --model_path /dir/to/model --model_ver VER_2 --eval_mode TEXT_TO_TEXT --processor_path /dir/to/proc \
    --output_path /path/to/output \
    --beam_size 5 --length_penalty 1.0 \
    [--dropout_fix]
```
The input file (`eval_dataset`) should be a `tsv` file containing one field of `src_text`.
The output file will contain lines of hypothesis.

_Test Set Hypothesis_: [examples/big_c/seamlessm4t/mt.test.tsv](./examples/big_c/seamlessm4t/mt.test.tsv)

_Metrics_: BLEU = 29.35, chrF2++ = 52.82

#### ASR & ST Fine-tune
Example of `bem-eng`.
- ASR.
```
python -m seamlessm4t.finetune \
    --src_lang bem --tgt_lang bem \
    --train_dataset /path/to/train/tsv --eval_dataset /path/to/dev/tsv \
    --model_name facebook/seamless-m4t-v2-large \
    --save_model_to /dir/to/model --save_processor_path /dir/to/proc \
    --mode SPEECH_TO_TEXT --eval_mode SPEECH_TO_TEXT \
    --learning_rate 1e-4 --warmup_steps 1000 --max_epochs 10 \
    --batch_size 120 --update_freq 30 --max_speech_dur 30.0 \
    --eval_steps 250 --patience 10 \
    --log_steps 50 \
    [--dropout_fix]
```

_Log_: [examples/big_c/seamlessm4t/asr.train.log](./examples/big_c/seamlessm4t/asr.train.log)


- ST.

Set `--untie_lm_head --prefix_skip_len 1 --dropout_fix` to match the behavior of the official codebase.
Set `--learning_rate 5e-5` if using pre-trained components.

If you use `--dropout_fix` for ST and you initialize parameters using MT/ASR checkpoints, 
then it is recommended that your MT/ASR models were trained with `--dropout_fix` as well.
```
python -m seamlessm4t.finetune \
    --src_lang bem --tgt_lang eng \
    --train_dataset /path/to/train/tsv --eval_dataset /path/to/dev/tsv \
    --model_name facebook/seamless-m4t-v2-large \
    --save_model_to /dir/to/model --save_processor_path /dir/to/proc \
    --mode SPEECH_TO_TEXT --eval_mode SPEECH_TO_TEXT \
    --learning_rate 1e-4 --warmup_steps 1000 --max_epochs 10 \
    --batch_size 120 --update_freq 30 --max_speech_dur 30.0 \
    --eval_steps 250 --patience 10 \
    --log_steps 50 \
    [--load_from_asr_pretrained /dir/to/asr/checkpoint] \
    [--load_from_mt_pretrained /dir/to/mt/checkpoint] \
    [--untie_lm_head --prefix_skip_len 1 --dropout_fix]
```
- `--load_from_mt_pretrained`: Load MT ckpt for text decoder.
- `--load_from_asr_pretrained`: Load ASR encoder for speech encoder.

_Log_: [examples/big_c/seamlessm4t/st.train.log](./examples/big_c/seamlessm4t/st.train.log)

_Log with ASR pretrain_: [examples/big_c/seamlessm4t/st-asrinit.train.log](./examples/big_c/seamlessm4t/st-asrinit.train.log)

#### ASR & ST Inference
- ASR.
Set `--dropout_fix` if you use this in training.
```
python -m seamlessm4t.inference \
    --src_lang bem --tgt_lang bem \
    --eval_dataset /path/to/test/tsv \
    --model_path /dir/to/model --model_ver VER_2 --eval_mode SPEECH_TO_TEXT --processor_path /dir/to/proc \
    --output_path /path/to/output \
    --beam_size 5 --length_penalty 1.0 \
    [--dropout_fix]
```

_Test Set Hypothesis_: [examples/big_c/seamlessm4t/asr.test.tsv](./examples/big_c/seamlessm4t/asr.test.tsv)

_Metrics_: CER = 9.16, WER = 31.79

- ST.
Set `--dropout_fix` if you use this in training.
```
python -m seamlessm4t.inference \
    --src_lang bem --tgt_lang eng \
    --eval_dataset /path/to/test/tsv \
    --model_path /dir/to/model --model_ver VER_2 --eval_mode SPEECH_TO_TEXT --processor_path /dir/to/proc \
    --output_path /path/to/output \
    --beam_size 5 --length_penalty 1.0 \
    [--dropout_fix]
```
The test sets (`eval_dataset`) should be a `tsv` file containing one field of `audio_path`.
The output file will contain lines of hypothesis.

_E2E ST Test Set Hypothesis_: [examples/big_c/seamlessm4t/st.test.tsv](./examples/big_c/seamlessm4t/st.test.tsv)

_E2E ST Metrics_: BLEU = 30.74, chrF2++ = 53.29

_E2E ST_ASRinit Test Set Hypothesis_: [examples/big_c/seamlessm4t/st-asrinit.test.tsv](./examples/big_c/seamlessm4t/st-asrinit.test.tsv)

_E2E ST_ASRinit Metrics_: BLEU = 32.39, chrF2++ = 54.27

#### Multi-task ST Fine-tune
The ASR and MT initializations are optional but recommended.
```
python -m seamlessm4t.finetune \
    --src_lang bem --tgt_lang eng \
    --train_dataset /path/to/train/tsv --eval_dataset /path/to/dev/tsv \
    --model_name facebook/seamless-m4t-v2-large \
    --load_from_mt_pretrained /dir/to/mt/ckpt  --processor_path /dir/to/mt/proc \
    --load_from_asr_pretrained /dir/to/asr/ckpt \
    --save_model_to /dir/to/ckpt --save_processor_path /dir/to/proc \
    --mode MULTI_TASKING --eval_mode SPEECH_TO_TEXT \
    --learning_rate 6e-5 --warmup_steps 2000 --max_epochs 10 \
    --kd_loss_type KLD --detach_teacher \
    --alpha 1 --beta 1 --gamma 2 \
    --batch_size 120 --update_freq 60 --max_text_tokens 128 --max_speech_dur 30.0 \
    --eval_steps 250 --patience 10 \
    --log_steps 50 \
    [--untie_lm_head --prefix_skip_len 1 --dropout_fix] \
    [--finetune_text_encoder]
```
- `--load_from_mt_pretrained`: Load MT ckpt for text encoder and decoder.
- `--processor_path`: Load MT's processor.
- `--load_from_asr_pretrained`: Load ASR encoder for speech encoder.
- `--eval_mode`: you may use `MULTI_TASKING` for evaluation, but we always used `SPEECH_TO_TEXT` as we only cared about ST performance.
- `--kd_loss_type`: which loss to use for knowledge distillation. Supports either KL-Divergence (`KLD`) or cross-entropy (`XENT`).
- `--detach_teacher`: whether to detach the teacher logits before loss computation. If set, then the teacher's part of the KD loss will not backpropagate to the teacher modules.
- `--alpha 1 --beta 1 --gamma 2`: `alpha` is the weight for ST loss; `beta` is the weight for MT loss; `gamma` is the weight for KD loss.
- `--finetune_text_encoder`: whether to fine-tune text encoder. By default, it is frozen.

_Log_: [examples/big_c/seamlessm4t/st-mlt.train.log](./examples/big_c/seamlessm4t/st-mlt.train.log)


#### Multi-task ST Inference
Set `--dropout_fix` if using this in training.
```
python -m exps.seamless_hf_scripts.inference \
    --src_lang bem --tgt_lang eng \
    --eval_dataset /path/to/test/tsv \
    --model_path /dir/to/ckpt --model_ver VER_2 --eval_mode SPEECH_TO_TEXT --processor_path /dir/to/proc \
    --output_path /path/to/output \
    --beam_size 5 --length_penalty 1.0 \
    [--dropout_fix]
```
The test set (`eval_dataset`) should be a `tsv` file containing one field of `audio_path`.
The output file will contain lines of hypothesis.

_MLT ST Test Set Hypothesis_: [examples/big_c/seamlessm4t/st-mlt.test.tsv](./examples/big_c/seamlessm4t/st-mlt.test.tsv)

_MLT ST Metrics_: BLEU = 30.28, chrF2++ = 53.28

## Evaluation
Our internal evaluation code is not necessarily the same as the official IWSLT evaluation procedure.

We use the following function to normalize text.
It is borrowed from [here](https://multilingual.superbbenchmark.org/challenge-interspeech2025/challenge_overview#Evaluation-and-Scoring).
`keep_list` can contain apostrophes for French hypothesis.
```python
from typing import Iterable
import unicodedata

def remove_punctuation(text: str, keep_list: Iterable[str] = None) -> str:
    """
    Remove punctuations. Chars in the `keep_list` will be kept.
    """
    new_sentence = ""
    for char in text:
        # all unicode punctuation is of type P
        if unicodedata.category(char).startswith("P"):
            if keep_list and char in keep_list:
                # in the keep list, keep it even if it is a punctuation
                new_sentence = f"{new_sentence}{char}"
            else:
                # not in the keep list, so skip it
                continue
        else:
            new_sentence = f"{new_sentence}{char}"
    return new_sentence

hypo_sents: list[str]
hypo_sents = [remove_punctuation(s) for s in hypo_sents]
hypo_sents = [s.lower() for s in hypo_sents]
ref_sents: list[str]
ref_sents = [remove_punctuation(s) for s in ref_sents]
ref_sents = [s.lower() for s in ref_sents]
```

We use [jiwer](https://github.com/jitsi/jiwer) for WER and CER.
```python
import jiwer
wer = jiwer.wer(ref_sents, hypo_sents)
cer = jiwer.cer(ref_sents, hypo_sents)
```
We use [sacrebleu](https://github.com/mjpost/sacrebleu) for BLEU and chrF++.
```python
from sacrebleu.metrics import BLEU, CHRF
bleu_obj = BLEU(lowercase=True)
bleu = bleu_obj.corpus_score(hypo_sents, [ref_sents])
print(bleu)
print(bleu_obj.get_signature())

chrf_obj = CHRF(word_order=2, lowercase=True)
chrf = chrf_obj.corpus_score(hypo_sents, [ref_sents])
print(chrf)
print(chrf_obj.get_signature())
```

## Acknowledgement
The design of this codebase is largely based on the [official SeamlessM4T repository](https://github.com/facebookresearch/seamless_communication).
We thank them for open-sourcing a well-structured codebase.

## Citation
If you find this repository helpful, please cite the following article:
```
@misc{meng2025gmusystemsiwslt2025,
      title={GMU Systems for the IWSLT 2025 Low-Resource Speech Translation Shared Task}, 
      author={Chutong Meng and Antonios Anastasopoulos},
      year={2025},
      eprint={2505.21781},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.21781}, 
}
```

## License
MIT.
