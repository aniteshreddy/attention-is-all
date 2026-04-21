import torch
from torch.utils.data import Dataset


class LanguageDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_trgt, src_key, trgt_key, seq_len):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trgt = tokenizer_trgt
        self.src_key = src_key
        self.trgt_key = trgt_key
        self.seq_len = seq_len

        # MuRIL uses [CLS] as SOS, [SEP] as EOS
        self.sos_token = torch.tensor([tokenizer_src.cls_token_id], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_trgt.sep_token_id], dtype=torch.int64)
        self.pad_token_id = tokenizer_src.pad_token_id
        self.pad_token = torch.tensor([self.pad_token_id], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        input_text = self.ds[index]

        src_text, trgt_text = input_text[self.src_key], input_text[self.trgt_key]

        # encode() returns a list of ints directly when called on a tokenizer
        encoder_input_tokens = self.tokenizer_src.encode(src_text, add_special_tokens=False)
        decoder_input_tokens = self.tokenizer_trgt.encode(trgt_text, add_special_tokens=False)

        encode_padding_length = self.seq_len - len(encoder_input_tokens) - 2
        decode_padding_length = self.seq_len - len(decoder_input_tokens) - 1

        encoder_input = torch.concat(
            [
                self.sos_token,
                torch.tensor(encoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token_id] * encode_padding_length, dtype=torch.int64)
            ], dim=0)

        decoder_input = torch.concat(
            [
                self.sos_token,
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token_id] * decode_padding_length, dtype=torch.int64)
            ], dim=0)

        label = torch.concat(
            [
                torch.tensor(decoder_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token_id] * decode_padding_length, dtype=torch.int64)
            ])

        assert encoder_input.size(0) == self.seq_len, \
            f"Encoder input size mismatch: {encoder_input.size(0)} != {self.seq_len}"
        assert decoder_input.size(0) == self.seq_len, \
            f"Decoder input size mismatch: {decoder_input.size(0)} != {self.seq_len}"
        assert label.size(0) == self.seq_len, \
            f"Label size mismatch: {label.size(0)} != {self.seq_len}"

        return {
            "encoder_input": encoder_input,
            "encoder_mask": (encoder_input != self.pad_token_id).unsqueeze(0).unsqueeze(0).int(),
            "decoder_input": decoder_input,
            "decoder_mask": (decoder_input != self.pad_token_id).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,
            "src_text": src_text,
            "trgt_text": trgt_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0
