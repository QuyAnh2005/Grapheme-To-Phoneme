import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder, DecoderAttn
from g2p_seq2seq import SOS_token


class G2PModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=1, dropout=0.1, attention=True):
        super(G2PModel, self).__init__()
        self.attention = attention
        self.g_tokenizer = None
        self.detokenizer = None

        self.encoder = Encoder(input_size, hidden_size, num_layers, dropout)
        if self.attention:
            assert num_layers == 1, "Not support for attention with num_layers > 1."
            self.decoder = DecoderAttn(hidden_size, output_size, num_layers, dropout)
        else:
            self.decoder = Decoder(hidden_size, output_size, num_layers, dropout)
        self.input_len = None
        self.target_len = None

    def forward(self, input_seq, target=None):
        # input_seq shape: (batch_size, seq_len)
        # target shape: (batch_size, target_len)
        self.input_len = input_seq.shape[1]
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(input_seq)
        decoder_hidden = encoder_hidden

        decoder_input = torch.tensor([[SOS_token]] * input_seq.size(0), dtype=torch.long,
                                     device=input_seq.device)  # start symbol index is 1
        # decoder_input shape: (batch_size, 1)
        outputs = []
        if target is not None:
            self.target_len = target.shape[1]
            for i in range(self.target_len):
                if self.attention:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs.append(decoder_output)
                decoder_input = target[:, i].unsqueeze(1)
            outputs = torch.cat(outputs, dim=1)  # shape: (batch_size, target_len, output_size)

        else:  # inference
            for i in range(self.target_len):
                if self.attention:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_output)
                else:
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs.append(decoder_output)
                decoder_input = decoder_output.argmax(dim=2)
            outputs = torch.cat(outputs, dim=1)  # shape: (batch_size, target_len, output_size)
        return outputs

    def predict(self, input_seq):
        with torch.no_grad():
            phoneme_seq = self.forward(input_seq)
        return phoneme_seq

    def decode_word(self, word):
        # Tokenize word
        word_seq = self.g_tokenizer.tokenize(word, self.input_len)
        input_seq = torch.tensor(word_seq, dtype=torch.long).unsqueeze(0)
        predictions = self.predict(input_seq)
        seq = predictions.argmax(dim=2)

        # Detokenize
        phoneme = self.detokenizer.detokenize(seq.squeeze(0).numpy())
        return " ".join(phoneme)
