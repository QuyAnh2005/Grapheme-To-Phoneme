import torch
import g2p_seq2seq

from jiwer import wer


class Vocab:
    """This class is used to build vocabulary for graphemes and phonemes from available file."""

    def __init__(self, data):
        self.data = data
        self.grapheme_to_id, self.phoneme_to_id = self.build_vocab()

    def build_vocab(self):
        # Create a set of unique graphemes and phonemes
        graphemes = set()
        phonemes = set()
        for word, pron in self.data:
            graphemes.update(set(word))
            phonemes.update(set(pron))
        # Sort the sets to ensure consistent ordering
        graphemes = sorted(list(graphemes))
        phonemes = sorted(list(phonemes))

        # Add a special character for vocab
        graphemes.insert(g2p_seq2seq.PAD_token, g2p_seq2seq.PAD)
        phonemes.insert(g2p_seq2seq.PAD_token, g2p_seq2seq.PAD)

        graphemes.insert(g2p_seq2seq.SOS_token, g2p_seq2seq.SOS)
        phonemes.insert(g2p_seq2seq.SOS_token, g2p_seq2seq.SOS)

        graphemes.insert(g2p_seq2seq.EOS_token, g2p_seq2seq.EOS)
        phonemes.insert(g2p_seq2seq.EOS_token, g2p_seq2seq.EOS)

        # Create dictionaries to map graphemes/phonemes to integers
        grapheme_to_id = {g: i for i, g in enumerate(graphemes)}
        phoneme_to_id = {p: i for i, p in enumerate(phonemes)}

        return grapheme_to_id, phoneme_to_id


class Tokenize:
    def __init__(self, vocab_to_id):
        self.vocab_to_id = vocab_to_id

    def tokenize(self, sequence, max_length):
        # Convert to vector of number
        num_vec = [self.vocab_to_id[g2p_seq2seq.SOS]] + [self.vocab_to_id[w] for w in sequence] + [self.vocab_to_id[g2p_seq2seq.EOS]]

        # Padding and truncate
        length = len(num_vec)
        if length < max_length:
            tokenized_vector = num_vec + (max_length - length) * [self.vocab_to_id[g2p_seq2seq.PAD]]
        else:
            tokenized_vector = num_vec[:length]

        return torch.tensor(tokenized_vector)


class Detokenize:
    def __init__(self, id_to_vocab):
        self.id_to_vocab = id_to_vocab

    def detokenize(self, num_vec):
        detokenized_vector = []
        for id in num_vec:
            if self.id_to_vocab[id] == g2p_seq2seq.SOS:
                continue

            if self.id_to_vocab[id] == g2p_seq2seq.EOS:
                break

            detokenized_vector.append(self.id_to_vocab[id])

        return detokenized_vector


def accuracy(output, target):
    preds = output.argmax(dim=1)
    correct = (preds == target).float()
    acc = correct.sum() / len(correct)
    return acc


def decode(sequence, detokenizer):
    detokenized_seq = detokenizer.detokenize(sequence)
    return detokenized_seq


def word_error_rate(output, target, seq_len, detokenizer):
    preds = output.argmax(dim=1)
    preds = preds.reshape(-1, seq_len)
    target = target.reshape(-1, seq_len)
    batch_size = preds.size(0)

    wer_score = 0.0
    correct_word = 0
    for idx in range(batch_size):
        pred_phoneme = decode(preds[idx].numpy(), detokenizer)
        pred_phoneme = " ".join(pred_phoneme)
        target_phoneme = decode(target[idx].numpy(), detokenizer)
        target_phoneme = " ".join(target_phoneme)
        correct_word += 1 if target_phoneme == pred_phoneme else 0
        wer_score += wer(target_phoneme, pred_phoneme)

    return torch.tensor(wer_score / batch_size), torch.tensor(correct_word / batch_size)


def read_file(file_path):
    f_obj = open(file_path, 'r').readlines()
    data = []
    for line in f_obj:
        line_elements = line.split()
        word, phones = line_elements[0], line_elements[1:]
        data.append((word, phones))

    return data
