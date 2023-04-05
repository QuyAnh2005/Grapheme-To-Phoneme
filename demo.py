import torch

# Demo for prediction
model = torch.load('g2p_seq2seq/pretrained/model.pth')
print(model.decode_word("stekstop"))  # s t e k s t o p
