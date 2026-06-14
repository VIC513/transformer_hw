import torch

from dataset import PAD_ID, VOCAB_SIZE, collate_skip_bad
from model import build_default_model


def main():
    batch = [
        {"input_ids": torch.randint(0, VOCAB_SIZE, (300,), dtype=torch.long), "label": torch.tensor(0), "path": "a"},
        None,
        {"input_ids": torch.randint(0, VOCAB_SIZE, (300,), dtype=torch.long), "label": torch.tensor(1), "path": "b"},
    ]
    batch[0]["input_ids"][-10:] = PAD_ID
    batch[2]["input_ids"][-20:] = PAD_ID

    packed = collate_skip_bad(batch)
    assert packed is not None

    model = build_default_model(seq_len=packed["input_ids"].size(1), vocab_size=VOCAB_SIZE, num_classes=4)
    logits = model(packed["input_ids"], packed["key_padding_mask"])
    assert logits.shape == (2, 4)


if __name__ == "__main__":
    main()
    print("ok")

