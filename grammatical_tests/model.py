import torch.nn as nn
import torch.nn.functional as F

from weight_drop import WeightDrop


class WordNLM(nn.Module):
    def __init__(self, char_embedding_size, vocab_size, hidden_dim, layer_num,
                 weight_dropout_in=0, weight_dropout_hidden=0, char_dropout_prob=0):
        super(WordNLM, self).__init__()
        # Hyperparams
        self.char_embedding_size = char_embedding_size
        self.vocab_size = vocab_size  # FIXME: Vocab size + 3
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.weight_dropout_in = weight_dropout_in
        self.weight_dropout_hidden = weight_dropout_hidden
        self.char_dropout_prob = char_dropout_prob

        # Model architecture
        self.char_embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.char_embedding_size)
        self.char_dropout = nn.Dropout2d(p=self.char_dropout_prob)
        self.rnn = nn.LSTM(self.char_embedding_size, self.hidden_dim, self.layer_num)
        self.rnn.flatten_parameters()
        weight_drop_params = self.get_weigh_drop_parameters()
        self.rnn_drop = WeightDrop(self.rnn, weight_drop_params)
        self.output = nn.Linear(self.hidden_dim, self.vocab_size)

    def get_weigh_drop_parameters(self):
        dropout_in = [(name, self.weight_dropout_in) for name, _ in self.rnn.named_parameters()
                      if name.startswith("weight_ih_")]
        dropout_hidden = [(name, self.weight_dropout_hidden) for name, _ in self.rnn.named_parameters()
                          if name.startswith("weight_hh_")]
        return dropout_in + dropout_hidden

    def forward(self, sentence_except_last_word):
        embedded_forward = self.char_dropout(self.char_embeddings(sentence_except_last_word))
        out_forward, hidden_forward = self.rnn_drop(embedded_forward, None)
        prediction = F.log_softmax(self.output(out_forward), dim=2)
        return prediction
