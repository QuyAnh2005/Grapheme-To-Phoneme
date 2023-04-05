import torch
from .utils import accuracy, word_error_rate


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    epoch_wer = 0
    epoch_acc_word = 0

    for batch in iterator:
        grapheme, phoneme = batch
        optimizer.zero_grad()

        output = model(grapheme, phoneme)
        output_dim = output.shape[-1]
        seq_len = output.shape[-2]
        output = output[:, :].reshape(-1, output_dim)
        phoneme = phoneme[:, :].reshape(-1)

        loss = criterion(output, phoneme)
        acc = accuracy(output, phoneme)
        wer_score, acc_word = word_error_rate(output, phoneme, seq_len, model.detokenizer)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_wer += wer_score.item()
        epoch_acc_word += acc_word.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_acc_word / len(iterator), epoch_wer / len(iterator)