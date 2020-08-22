import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class AttnDecoderRNN(nn.Module):
    def __init__(self,input_size,embedding_size=50, output_size=None,embbed_pretrained_weight=None, dropout_p=0.1, max_length=20):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = embedding_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        #self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.embedding = nn.Embedding(input_size, embedding_size)
        # using matched pretrained embeddings:
        if not embbed_pretrained_weight is None:  # Using zero initialization for nonvocabulary words:
            self.embedding.weight.data.copy_(torch.from_numpy(embbed_pretrained_weight))

        self.attn = nn.Linear(embedding_size * 2, 10)
        self.attn_combine = nn.Linear(self.hidden_size + max_length+1 , self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded).squeeze()
        if len(np.shape(embedded) ) ==1:
            embedded =embedded.unsqueeze(0)
            attn_weights = F.softmax(
                self.attn(torch.cat((embedded, hidden.squeeze(0)), 1)), dim=1)
        else:
            attn_weights = F.softmax(
                self.attn(torch.cat((embedded, hidden.squeeze()), 1)), dim=1)
        attn_applied = torch.bmm(encoder_outputs,attn_weights.unsqueeze(1).transpose(1,2)
                                 )

        output = torch.cat((embedded, attn_applied[:,:,0]   ), 1)
        output = self.attn_combine(output).unsqueeze(1)

        output = F.relu(output.transpose(1,0))
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self, batch_size):
        return torch.zeros( 1,batch_size, self.hidden_size).cuda()



class Eecoder_Gen_lstm(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size, n_layers=4, bidirectional=True, embedding=None,use_conv= False,
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
        self.use_conv = use_conv

        # Define layers
        self.embedding = embedding
        # using matched pretrained embeddings:

        self.embedding.weight.requires_grad = True
        self.conv2_rnn = nn.Conv1d(in_channels=hidden_size * 2, out_channels=hidden_size * 2, stride=2, kernel_size=3)

        self.conv1_rnn = nn.Conv1d(in_channels=hidden_size * 2, out_channels=hidden_size * 2, stride=1, kernel_size=3)


        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=0.5, batch_first=True,
                            bidirectional=self.bidirectional)
        self.mu_est =nn.Linear(hidden_size* 2, int(hidden_size/2) )
        #self.mu_est = nn.Parameter(torch.FloatTensor(self.hidden_size * 2, self.hidden_size * 2).cuda())
        self.sigma_est =nn.Linear(hidden_size* 2, int(hidden_size/2) )

        #self.sigma_est = nn.Parameter(torch.FloatTensor(self.hidden_size * 2, self.hidden_size * 2).cuda())

    def forward(self, word_input, last_hidden):
        # Note: we run this one step at a time (word by word...)
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input)  # S=1 x B x N
        # run through LSTM
        rnn_output, hidden = self.lstm(word_embedded, last_hidden)

        if self.sampling:
            mu= self.mu_est(rnn_output)
            logvar= self.sigma_est(rnn_output)

            #mu = torch.matmul(output_correct_dim, self.mu_est)
            #logvar = torch.matmul(output_correct_dim, self.sigma_est)

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
    max_length = 20 # Input sentence to encoder -> any input less , should be padded.
    decoder =AttnDecoderRNN(embedding_size=embedding_size, output_size=data_train.n_words ,input_size=data_train.n_words, embbed_pretrained_weight=data_train.embedding, dropout_p=0.1, max_length=max_length).cuda()

    encoder = Eecoder_Gen_lstm(hidden_size=hidden_size, embedding_size=embedding_size, \
                               output_size=data_train.n_words, \
                               n_layers=n_layers_encoder, bidirectional=True, embedding=decoder.embedding,
                               sampling=True).cuda()

    return encoder, decoder