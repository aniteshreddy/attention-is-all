from model import build_transformer
from tokenizer import get_ds
import torch
import torch.nn as nn
import tqdm
from dataset import causal_mask

from config import get_config


def get_model(config, src_vocab_size, trgt_vocab_size):
    model = build_transformer(src_vocab_size=src_vocab_size, trgt_vocab_size=trgt_vocab_size, src_seq_len=config['seq_len'], trgt_seq_len=config['seq_len'], d_model=config['d_model'])
    return model

def validate(model, val_dataloader, tokenizer_src, tokenizer_trgt, config, device, num_examples=2):
    model.eval()
    
    count = 0
    console_width = 80

    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            # Greedy decode
            sos_id = tokenizer_trgt.token_to_id('[SOS]')
            eos_id = tokenizer_trgt.token_to_id('[EOS]')

            # Encode the source sentence
            encoder_output = model.encode(encoder_input, encoder_mask)

            # Start with SOS token
            decoder_input = torch.empty(1, 1).fill_(sos_id).type_as(encoder_input).to(device)

            while True:
                if decoder_input.size(1) == config['seq_len']:
                    break

                decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

                # Get the next token probabilities
                prob = model.project(decoder_output[:, -1])
                _, next_word = torch.max(prob, dim=1)

                decoder_input = torch.cat([
                    decoder_input,
                    torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)
                ], dim=1)

                if next_word == eos_id:
                    break

            # Decode the tokens back to text
            model_out = decoder_input.squeeze(0)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_trgt.decode(model_out.detach().cpu().numpy())

            print('-' * console_width)
            print(f"{'SOURCE: ':>12}{source_text}")
            print(f"{'TARGET: ':>12}{target_text}")
            print(f"{'PREDICTED: ':>12}{model_out_text}")

            count += 1
            if count == num_examples:
                break

def train(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using Device: ", device)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trgt = get_ds(config=config)
    model = get_model(
                config=config, 
                src_vocab_size=tokenizer_src.get_vocab_size(), 
                trgt_vocab_size= tokenizer_trgt.get_vocab_size
            )

    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'], eps=1e-9 )

    model = model.to(device=device)

    initial_epoch = 0

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) 
            decoder_input = batch['decoder_input'].to(device) 
            encoder_mask = batch['encoder_mask'].to(device) 
            decoder_mask = batch['decoder_mask'].to(device) 

            encoder_output = model.encode(encoder_input, encoder_mask) 
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output) 

            label = batch['label'].to(device) 

            loss = loss_fn(proj_output.view(-1, tokenizer_trgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

           
            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        validate(model, val_dataloader, tokenizer_src, tokenizer_trgt, config, device)


        
