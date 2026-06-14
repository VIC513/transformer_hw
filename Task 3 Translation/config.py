from __future__ import annotations

from dataclasses import dataclass


PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN)


D_MODEL = 256
NHEAD = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

MAX_SEQ_LEN = 40

BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 8
WEIGHT_DECAY = 1e-4

TRAIN_SPLIT = 0.98
MIN_FREQ = 1


@dataclass(frozen=True)
class DataPaths:
    pairs_tsv: str = "../data/tatoeba_en_zh.tsv"

