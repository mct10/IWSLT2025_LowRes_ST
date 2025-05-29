from utils.log_utils import logging

logger = logging.getLogger(__name__)


def fix_model_dropout(model):
    logger.info(f"Will update decoder's ffn_dropout")
    for dec_layer in model.text_decoder.layers:
        # decoder output dropout
        dec_layer.ffn_dropout.p = 0.1
    if hasattr(model, "speech_encoder") and model.speech_encoder is not None:
        logger.info(f"Will update adapter's attention score & ffn dropout")
        for adp_layer in model.speech_encoder.adapter.layers:
            # attention score dropout
            adp_layer.self_attn.dropout.p = 0.1
            # ffn after activation dropout
            adp_layer.ffn.intermediate_dropout.p = 0.0
    # todo: seamless' text_decoder_frontend has a dropout for word embeddings
    # this is easy to implement in fine-tuning, but requires extra efforts for inference...
