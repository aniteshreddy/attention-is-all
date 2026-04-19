from torch.utils.data import random_split, DataLoader

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from datasets import load_dataset

from dataset import LanguageDataset

from pathlib import Path



def get_all_sentences(ds, key):
    for i in ds['train']:
        yield i[key]

def get_or_build_tokeniser(config, dataset, key):
    tokenizer_path = Path(config['tokenizer_file'].format(key))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]" ], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds=dataset, key= key), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_ds(config):
    dataset = load_dataset(config["dataset"])
    tokenizer_src = get_or_build_tokeniser(config=config, dataset= dataset, key=config["src_key"] )
    tokenizer_trgt = get_or_build_tokeniser(config=config, dataset= dataset, key=config["trgt_key"])
    
    total = len(dataset['train'])

    train_ds_size = int(0.9 * len(dataset['train']))
    val_ds_size = len(dataset['train']) - train_ds_size
    print(f"Total: {total}, Train: {train_ds_size}, Val: {val_ds_size}")


    train_ds_raw, val_ds_raw = random_split(dataset=dataset['train'], lengths=[train_ds_size, val_ds_size])

    train_ds = LanguageDataset(
            ds=train_ds_raw,
            tokenizer_src=tokenizer_src, 
            tokenizer_trgt=tokenizer_trgt, 
            src_key=config['src_key'],
            trgt_key=config['trgt_key'],
            seq_len=config['seq_len']
        )

    val_ds = LanguageDataset(
            ds=val_ds_raw,
            tokenizer_src=tokenizer_src, 
            tokenizer_trgt=tokenizer_trgt, 
            src_key=config['src_key'],
            trgt_key=config['trgt_key'],
            seq_len=config['seq_len']
        )
    
    max_len_src = 0
    max_len_tgt = 0

    for item in dataset['train']:
        src_ids = tokenizer_src.encode(item[config['src_key']]).ids
        tgt_ids = tokenizer_trgt.encode(item[config['trgt_key']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_trgt




    





