# Grapheme-to-Phoneme Conversion using Sequence-to-Sequence Models
This repository contains an implementation of a G2P conversion model using sequence-to-sequence neural networks. Given a sequence of graphemes, the model predicts the corresponding sequence of phonemes.

## 1. Dataset
The dataset used to train the model consists of 3 files which are: 
- `train.dict`: used for training 
- `dev.dict`: used for validation
- `test.dict`: used for prediction 

Each file will have the following format:
```
instant     i n s t a n t
ne-am   n e a m
prescriu    p r e s k r i u
schimbat    s k i m b a t
companii	k o m p a n i iX
berărie	    b e r at r i e
protv   p r o t v
lux     l u k s
ne-ndepărtăm    n e n d e p at r t at m
turbat      t u r b a t
retrovizoarele	   r e t r o v i z oa r e l e
wolfensohn	w o l f e n s o h n
gusta	g u s t a
...
```
where first element (as `instant`) are graphemes, the following element are phonemes (as `i n s t a n t`).
The above example is of Romani language. Other languages need to have the same format.


## 2. Model
The model architecture consists of an encoder-decoder with attention mechanism. 
The encoder is a  LSTM network that processes the input sequence of graphemes. 
The decoder is also an LSTM network that generates the corresponding sequence of phonemes, with attention mechanism applied to the encoder outputs at each decoding step. In conclusion, model includes:
- An Encoder Layer 
- A Decoder Layer: RNN Decoder or Attention Decoder

## 3. Program
### 3.1 Install requirement
To run program, you firstly need to install some required packages by
```
pip install -r requirements.txt
```

### 3.2 Training
You only need to specify format dataset (`1. Dataset`) as discussed. The preprocessing is automatic.
To train:
```
python train_g2p.py --train_file <train.dict> --model_path <path_to_save_model> --dev_file <dev.dict> {other parameters}
```
Example:
```
python train_g2p.py --train_file dataset/train.dict  --out_path g2p_seq2seq/pretrained/model.pth --dev_file dataset/dev.dict
```

### 3.3 Results
|         | Accuracy (Token) | Accuracy (Word - Grapheme) | Word Error Rate |
|:---------------------:|:---------:|---:|---------------------------------:|
|Training Dataset| 99.74% | 94.09% | 0.82%|
|Evaluate (Dev) Dataset| 99.52% | 93.77% | 0.93%|

### 3.4 Prediction
The prediction is demo in `demo.py`.

## 4. References
- Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint [arXiv:1409.0473](https://arxiv.org/abs/1409.0473).
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In Advances in neural information processing systems (pp. 3104-3112).
- Sean Robertson. [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)