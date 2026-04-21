def get_config():
    return {
        "batch_size": 128,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 150,
        "d_model": 512,
        "src_key": "Input",
        "trgt_key" : "Output",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "dataset": "MRR24/English_to_Telugu_Bilingual_Sentence_Pairs",
        "checkpoint_dir": "/content/drive/MyDrive/translator_checkpoints",
        "num_val_examples": 5,
        "warmup_steps": 4000,
    }