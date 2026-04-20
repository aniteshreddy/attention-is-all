import torch
import torch.nn as nn
from torch.utils.data import Dataset
from model import build_transformer


class LanguageDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_trgt, src_key, trgt_key, seq_len):
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_trgt = tokenizer_trgt
        self.src_key = src_key
        self.trgt_key = trgt_key
        self.seq_len = seq_len

        self.sos_token = torch.tensor([self.tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([self.tokenizer_trgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([self.tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        input_text = self.ds[index]

        src_text, trgt_text = input_text[self.src_key], input_text[self.trgt_key]
        
        encoder_input_tokens = self.tokenizer_src.encode(src_text).ids 
        decoder_input_tokens = self.tokenizer_trgt.encode(trgt_text).ids

        encode_padding_length = self.seq_len - len(encoder_input_tokens) - 2
        decode_padding_length = self.seq_len - len(decoder_input_tokens) - 1

        encoder_input = torch.concat(
            [
                self.sos_token, 
                torch.tensor([encoder_input_tokens], dtype=torch.int64),  
                self.eos_token,  
                torch.tensor([[self.pad_token] * encode_padding_length], dtype=torch.int64)
            ], dim=0)

        decoder_input = torch.concat(
            [
                self.sos_token, 
                torch.tensor([decoder_input_tokens], dtype=torch.int64),    
                torch.tensor([[self.pad_token] * decode_padding_length], 
                dtype=torch.int64)
            ], dim=0)

        label = torch.concat(
            [
                torch.tensor([decoder_input_tokens], dtype=torch.int64), 
                self.eos_token,     
                torch.tensor([[self.pad_token] * decode_padding_length], 
                dtype=torch.int64)
            ])
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input" : encoder_input,
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            "decoder_input" : decoder_input,
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label" : label,
            "src_text": src_text,
            "trgt_text": trgt_text
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


        



