import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

import g2p_seq2seq
from g2p_seq2seq.utils import read_file, Vocab, Tokenize, Detokenize
from g2p_seq2seq.g2p import G2PModel
from g2p_seq2seq.loader import GraphemePhonemeDataset
from g2p_seq2seq.train import train
from g2p_seq2seq.evaluate import evaluate
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_argument():

    # Create the parser
    parser = argparse.ArgumentParser(description='Training G2P Model')
    parser.add_argument("--train_file", required=True, help="Path to train file")
    parser.add_argument("--out_path", required=True, help="Path to save model after training")
    parser.add_argument("--dev_file", required=True, help="Path to dev file")
    parser.add_argument("--lr", required=False, default=0.005, type=float)
    parser.add_argument("--batch_size", required=False, default=32, type=int)
    parser.add_argument("--num_epoch", required=False, default=500, type=int)
    parser.add_argument("--hidden_size", required=False, default=36, type=int)
    parser.add_argument("--num_layers", required=False, default=1, type=int)
    parser.add_argument("--dropout", required=False, default=0.1, type=float)
    parser.add_argument("--max_length_grapheme", required=False, default=24, type=int)
    parser.add_argument("--max_length_phoneme", required=False, default=24, type=int)
    parser.add_argument("--attention", required=False, default=True, type=bool)
    parser.add_argument("--verbose", required=False, default=True, type=bool)

    return parser


if __name__ == "__main__":
    # Create the parser
    parser = get_argument()

    # Parse the arguments
    args = parser.parse_args()
    train_path = args.train_file
    out_path = args.out_path
    valid_path = args.dev_file
    lr = args.lr
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    dropout = args.dropout
    max_length_grapheme = args.max_length_grapheme
    max_length_phoneme = args.max_length_phoneme
    attention = args.attention
    verbose = args.verbose

    # Load dataset
    raw_train = read_file(train_path)
    raw_valid = read_file(valid_path)

    # Get vocabulary
    vocab = Vocab(raw_train)
    grapheme_to_id, phoneme_to_id = vocab.grapheme_to_id, vocab.phoneme_to_id

    # Tokenizer object
    g_tokenizer = Tokenize(grapheme_to_id)
    p_tokenizer = Tokenize(phoneme_to_id)
    train_data = GraphemePhonemeDataset(
        raw_train,
        g_tokenizer,
        p_tokenizer,
        max_length_grapheme,
        max_length_phoneme
    )
    train_loader = DataLoader(train_data, batch_size=batch_size)

    valid_data = GraphemePhonemeDataset(
        raw_valid,
        g_tokenizer,
        p_tokenizer,
        max_length_grapheme,
        max_length_phoneme
    )
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    # Detokenize object
    id_to_phoneme = {phone: idx for idx, phone in phoneme_to_id.items()}
    detokenizer = Detokenize(id_to_phoneme)

    # Define model
    input_size = len(grapheme_to_id)
    output_size = len(phoneme_to_id)

    model = G2PModel(input_size, output_size, hidden_size, num_layers, dropout, attention)
    model.g_tokenizer = g_tokenizer
    model.detokenizer = detokenizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=verbose)
    clip = 5

    best_loss = np.inf
    for e in range(num_epoch):
        train_loss, train_acc, train_acc_word, train_wer = train(model, train_loader, optimizer, criterion, clip)
        valid_loss, valid_acc, valid_acc_word, valid_wer = evaluate(model, valid_loader, criterion)
        print(f'Epoch: {e + 1:02}')
        print(
            f'\tTrain Loss: {train_loss:.3f} | '
            f'Train Acc: {train_acc * 100:.2f}% | '
            f'Train Acc-Word: {train_acc_word * 100:.2f}% | '
            f'Train WER: {train_wer * 100:.2f}%'
        )
        print(
            f'\tValid Loss: {valid_loss:.3f} | '
            f'Valid Acc: {valid_acc * 100:.2f}% | '
            f'Valid Acc-Word: {valid_acc_word * 100:.2f}% | '
            f'Valid WER: {valid_wer * 100:.2f}%'
        )

        # if valid_loss < best_loss:
        #     best_loss = valid_loss
        #     print(f"Model is saved with loss {valid_loss}")
        #     torch.save(model, out_path)
        # scheduler.step(valid_loss)
