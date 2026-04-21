from model import build_transformer
from tokenizer import get_ds
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import causal_mask
from pathlib import Path
import re


def get_model(config, src_vocab_size, trgt_vocab_size):
    model = build_transformer(src_vocab_size=src_vocab_size, trgt_vocab_size=trgt_vocab_size, src_seq_len=config['seq_len'], trgt_seq_len=config['seq_len'], d_model=config['d_model'])
    return model


def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint by scanning for the highest epoch number."""
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoint_files:
        return None

    latest_epoch = -1
    latest_file = None
    for f in checkpoint_files:
        match = re.search(r'checkpoint_epoch_(\d+)\.pt', f.name)
        if match:
            epoch = int(match.group(1))
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_file = f

    return latest_file


def validate(model, val_dataloader, tokenizer_src, tokenizer_trgt, loss_fn, config, device):
    model.eval()
    num_examples = config.get('num_val_examples', 2)
    console_width = 80

    total_loss = 0.0
    num_batches = 0
    example_batches = []

    with torch.no_grad():
        batch_iterator = tqdm(val_dataloader, desc="  Validating")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, tokenizer_trgt.vocab_size), label.view(-1))
            total_loss += loss.item()
            num_batches += 1

            batch_iterator.set_postfix({"val_loss": f"{total_loss / num_batches:6.3f}"})

            # Collect batches for example printing
            if len(example_batches) < num_examples:
                example_batches.append(batch)

    avg_val_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"  Validation Loss: {avg_val_loss:.4f}")

    # Print example translations using greedy decoding
    sos_id = tokenizer_trgt.cls_token_id
    eos_id = tokenizer_trgt.sep_token_id

    with torch.no_grad():
        for batch in example_batches:
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_input = torch.empty(1, 1).fill_(sos_id).type_as(encoder_input).to(device)

            while True:
                if decoder_input.size(1) == config['seq_len']:
                    break

                decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

                prob = model.project(decoder_output[:, -1])
                _, next_word = torch.max(prob, dim=1)

                decoder_input = torch.cat([
                    decoder_input,
                    torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)
                ], dim=1)

                if next_word == eos_id:
                    break

            model_out = decoder_input.squeeze(0)

            source_text = batch['src_text'][0]
            target_text = batch['trgt_text'][0]
            model_out_text = tokenizer_trgt.decode(model_out.detach().cpu().numpy(), skip_special_tokens=True)

            print('-' * console_width)
            print(f"{'SOURCE: ':>12}{source_text}")
            print(f"{'TARGET: ':>12}{target_text}")
            print(f"{'PREDICTED: ':>12}{model_out_text}")

    return avg_val_loss


def get_lr_scheduler(optimizer, config, steps_per_epoch):
    warmup_steps = config.get('warmup_steps', 4000)

    def lr_lambda(step):
        step = max(step, 1)
        # Linear warmup from 0 → 1 over warmup_steps, then inverse sqrt decay
        if step < warmup_steps:
            return step / warmup_steps
        return (warmup_steps / step) ** 0.5

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using Device: ", device)

    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_trgt = get_ds(config=config)
    model = get_model(
                config=config,
                src_vocab_size=tokenizer_src.vocab_size,
                trgt_vocab_size=tokenizer_trgt.vocab_size
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    scheduler = get_lr_scheduler(optimizer, config, steps_per_epoch=len(train_dataloader))

    model = model.to(device=device)

    initial_epoch = 0
    global_step = 0

    # Load latest checkpoint by scanning epoch numbers
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is not None:
        print(f"Loading checkpoint: {latest_checkpoint.name}")
        checkpoint = torch.load(str(latest_checkpoint), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        initial_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint.get('global_step', initial_epoch * len(train_dataloader))
        print(f"Resuming from epoch {initial_epoch}, global step {global_step}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.pad_token_id, label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch:02d} | Learning Rate: {current_lr:.6e}")
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d} [Train]")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer_trgt.vocab_size), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        print(f"Saving checkpoint for epoch {epoch}...")
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, str(checkpoint_path))
        print(f"Checkpoint saved to {checkpoint_path}")

        # Validate
        validate(model, val_dataloader, tokenizer_src, tokenizer_trgt, loss_fn, config, device)