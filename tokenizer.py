import torch
from torch.utils.data import random_split, DataLoader

from transformers import AutoTokenizer

from datasets import load_dataset

from dataset import LanguageDataset


def get_ds(config):
    dataset = load_dataset(config["dataset"])

    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")

    total = len(dataset['train'])
    train_ds_size = int(0.9 * total)
    val_ds_size = total - train_ds_size
    print(f"Total: {total}, Train: {train_ds_size}, Val: {val_ds_size}")

    seed = config.get('seed', 42)
    split_generator = torch.Generator().manual_seed(seed)
    train_ds_raw, val_ds_raw = random_split(
        dataset=dataset['train'],
        lengths=[train_ds_size, val_ds_size],
        generator=split_generator,
    )

    train_ds = LanguageDataset(
        ds=train_ds_raw,
        tokenizer_src=tokenizer,
        tokenizer_trgt=tokenizer,
        src_key=config['src_key'],
        trgt_key=config['trgt_key'],
        seq_len=config['seq_len']
    )

    val_ds = LanguageDataset(
        ds=val_ds_raw,
        tokenizer_src=tokenizer,
        tokenizer_trgt=tokenizer,
        src_key=config['src_key'],
        trgt_key=config['trgt_key'],
        seq_len=config['seq_len']
    )

    max_len_src = 0
    max_len_tgt = 0

    for item in dataset['train']:
        src_ids = tokenizer.encode(item[config['src_key']], add_special_tokens=False)
        tgt_ids = tokenizer.encode(item[config['trgt_key']], add_special_tokens=False)
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    if max_len_src + 2 > config['seq_len'] or max_len_tgt + 2 > config['seq_len']:
        print(f"WARNING: seq_len={config['seq_len']} may be too short. "
              f"Consider increasing to at least {max(max_len_src, max_len_tgt) + 2}.")

    loader_generator = torch.Generator().manual_seed(seed)
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, generator=loader_generator)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True, generator=loader_generator)

    return train_dataloader, val_dataloader, tokenizer, tokenizer
