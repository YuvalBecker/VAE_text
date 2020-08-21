import torch
from torch import nn, optim
from Data_loader import Data_for_train
from Data_loader import freinds_parsing
import random
from losses import KLD, similarity_loss
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch.nn.functional as F
import argparse



class AttnDecoderGen_lstm(nn.Module):
    def __init__(self, hidden_size, embedding_size, input_size, output_size, n_layers=4, embbed_pretrained_weight=None,
                 seq_len=19, bidirectional=False, encoder_bidirictional=True):
        super(AttnDecoderGen_lstm, self).__init__()

        # Keep parameters for reference
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_size, embedding_size)
        # using matched pretrained embeddings:
        if not embbed_pretrained_weight is None:  # Using zero initialization for nonvocabulary words:
            self.embedding.weight.data.copy_(torch.from_numpy(embbed_pretrained_weight))
        self.embedding.weight.requires_grad = True
        self.drop_out_embedding = nn.Dropout(0.3)

        self.drop_out = nn.Dropout(0.4)

        # Define layers
        self.encoder_bidirictional = encoder_bidirictional
        self.lstm2 = nn.LSTM(self.embedding_size + self.hidden_size, hidden_size, n_layers, batch_first=True,
                             dropout=0.5, bidirectional=self.bidirectional)
        self.out = nn.Linear(hidden_size, output_size)

        #
        self.attn = Attn(hidden_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(last_hidden[0], encoder_outputs)
        context = attn_weights.unsqueeze(1).bmm(encoder_outputs)  # B x 1 x N

        word_embedded = self.embedding(word_input.squeeze())  # S=1 x B x N

        word_embedded = self.drop_out_embedding(word_embedded)
        # Combine embedded input word and last context, run through RNN
        context = context.squeeze()
        word_embedded = word_embedded.squeeze()
        if len(np.shape(word_embedded)) < 2:
            word_embedded = word_embedded.unsqueeze(0)
            context = context.unsqueeze(0)
        rnn_input = torch.cat((word_embedded, context), 1)
        rnn_output, hidden = self.lstm2(rnn_input.unsqueeze(1), last_hidden)

        # Final output layer (next word prediction) using the RNN hidden state and context vector

        rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        output = self.drop_out(self.out(rnn_output))

        # Return final output, hidden state
        return output, hidden

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda(),
                  torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda())
        return hidden


class Eecoder_Gen_lstm(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size, n_layers=4, bidirectional=True, embedding=None,
                 sampling=False):
        super(Eecoder_Gen_lstm, self).__init__()

        # Keep parameters for reference
        self.sampling = sampling
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.const_embed = []
        self.bidirectional = bidirectional

        # Define layers
        self.embedding = embedding
        # using matched pretrained embeddings:

        self.embedding.weight.requires_grad = True
        self.conv2_rnn = nn.Conv1d(in_channels=hidden_size * 2, out_channels=hidden_size * 2, stride=2, kernel_size=3)

        self.conv1_rnn = nn.Conv1d(in_channels=hidden_size * 2, out_channels=hidden_size * 2, stride=1, kernel_size=3)

        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=0.5, batch_first=True,
                            bidirectional=self.bidirectional)
        self.mu_est = nn.Parameter(torch.FloatTensor(self.hidden_size * 2, self.hidden_size * 2).cuda())

        self.sigma_est = nn.Parameter(torch.FloatTensor(self.hidden_size * 2, self.hidden_size * 2).cuda())

    def forward(self, word_input, last_hidden):
        # Note: we run this one step at a time (word by word...)
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input)  # S=1 x B x N
        # run through LSTM
        rnn_output, hidden = self.lstm(word_embedded, last_hidden)
        # Reducing dimentions
        out_conv1d = self.conv2_rnn(self.conv1_rnn(rnn_output.transpose(2, 1)))
        # correcting dims for LSTM later :
        output_correct_dim = out_conv1d.transpose(2, 1)

        if self.sampling:
            mu = torch.matmul(output_correct_dim, self.mu_est)
            logvar = torch.matmul(output_correct_dim, self.sigma_est)

            logvar = torch.clamp(logvar, -10, 10)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            mu = torch.clamp(mu, -100, 100)

            sampled_tensor = mu + eps * std
            rnn_output = sampled_tensor

        return rnn_output, hidden, mu, logvar

    def init_hidden(self, batch_size):
        hidden = (torch.zeros((1 + self.bidirectional) * self.n_layers, batch_size, self.hidden_size).cuda(),
                  torch.zeros((1 + self.bidirectional) * self.n_layers, batch_size, self.hidden_size).cuda())
        return hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()

        self.hidden_size = hidden_size
        self.v = nn.Parameter(torch.FloatTensor(self.hidden_size).cuda())
        self.attn = nn.Linear(self.hidden_size * (3), hidden_size)

    def forward(self, hidden, encoder_outputs):
        seq_len = np.shape(encoder_outputs)[1]

        attn_energies = []
        for i in range(seq_len):
            attn_energies.append(self.score(hidden.squeeze(), encoder_outputs[:, i, :]))
        concat_atten = []
        for i in range(len(attn_energies)):
            if len(concat_atten) == 0:
                concat_atten = attn_energies[i]
            else:
                concat_atten = torch.cat((concat_atten, attn_energies[i]), 0)
        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return torch.transpose(F.softmax(concat_atten, 0), 0, 1)

    def score(self, hidden, encoder_output):
        '''Aditive Attention'''

        if len(np.shape(hidden)) <= 2:
            hidden_b1 = hidden[0].unsqueeze(0)
            hidden_b2 = hidden[1].unsqueeze(0)
            hidden2 = torch.cat((hidden_b1, hidden_b2), 1)
        else:
            hidden2 = torch.cat((hidden[0], hidden[1]), 1)

        attn_input = torch.cat((hidden2, encoder_output), 1)
        energy = self.attn(attn_input)
        energy = torch.mm(self.v.unsqueeze(0), (torch.transpose(energy, 0, 1)))
        return energy


def vae_auto_encoder(opt, data_train):
    embedding_size = opt.embedding_size
    hidden_size = opt.n_hidden
    n_layers_encoder = opt.n_layers_E
    n_layers_decoder = opt.n_layers_D

    decoder = AttnDecoderGen_lstm(hidden_size=hidden_size * (2), embedding_size=embedding_size, \
                                  output_size=data_train.n_words, input_size=data_train.n_words,
                                  embbed_pretrained_weight=data_train.embedding, \
                                  n_layers=n_layers_decoder, bidirectional=False,
                                  encoder_bidirictional=True).cuda()

    encoder = Eecoder_Gen_lstm(hidden_size=hidden_size, embedding_size=embedding_size, \
                               output_size=data_train.n_words, \
                               n_layers=n_layers_encoder, bidirectional=True, embedding=decoder.embedding,
                               sampling=True).cuda()

    return encoder, decoder