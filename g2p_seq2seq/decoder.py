import torch
import torch.nn as nn
from torch.nn import functional as F


# RNN Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        output, (hidden, cell) = self.lstm(embedded, hidden)
        output = self.dropout(output)
        output = self.out(output)
        output = self.softmax(output)
        return output, hidden


# Attention Decoder
class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden: (1, batch_size, hidden_size)
        # encoder_outputs: (batch_size, max_seq_len, hidden_size)
        # Reshape hidden to (batch_size, hidden_size )
        hidden = hidden.transpose(0, 1).reshape(hidden.size(1), -1)
        attn_energies = torch.sum(
            self.v *
            torch.tanh(self.attn(
                torch.cat((hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1), encoder_outputs), dim=2))),
            dim=2
        )
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class DecoderAttn(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(DecoderAttn, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attn(hidden_size)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input_seq, hidden, encoder_outputs):
        # input_seq: (batch_size, 1)
        # hidden: (1, batch_size, hidden_size)
        # encoder_outputs: (batch_size, max_seq_len, hidden_size)

        embedded = self.embedding(input_seq)  # embedded: (batch_size, 1, hidden_size)
        attn_weights = self.attn(hidden, encoder_outputs)  # attn_weights: (batch_size, 1, max_seq_len)
        context = attn_weights.bmm(encoder_outputs)  # context: (batch_size, 1, hidden_size)
        rnn_input = torch.cat((embedded, context), 2)  # rnn_input: (batch_size, 1, hidden_size * 2)
        output, (hidden, cell) = self.lstm(rnn_input)
        # output: (batch_size, 1, hidden_size), hidden: (num_layers * num_directions, batch_size, hidden_size)
        output = self.dropout_layer(output)
        output = self.out(output)  # output: (batch_size, 1, output_size)
        output = self.softmax(output)
        return output, hidden
