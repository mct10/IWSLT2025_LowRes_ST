from enum import Enum


class FinetuneMode(Enum):
    SPEECH_TO_TEXT = "SPEECH_TO_TEXT"
    TEXT_TO_TEXT = "TEXT_TO_TEXT"
    MULTI_TASKING = "MULTI_TASKING"


class ModelVer(Enum):
    VER_1 = "VER_1"
    VER_2 = "VER_2"


class KdLossType(Enum):
    XENT = "XENT"
    KLD = "KLD"
