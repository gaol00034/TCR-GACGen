import math
import torch
import torch.nn as nn
from Model import transformer_gated
from collections import OrderedDict
@torch.no_grad()
def truncated_normal_(tensor, mean=0.0, std=1.0):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
class MLPEncoder(nn.Module):
    def __init__(self,
                 n_inputs=4000,
                 n_outputs=25,
                 decoder_dim=128,
                 hiddens=None,#[512,265,128,64]
                 activation='relu',
                 output_activation='linear',
                 dropout=0.2,
                 batch_norm=True,
                 regularize_last_layer=False):
        super(MLPEncoder, self).__init__()

        # create network architecture
        layers = []
        if hiddens is None:  # no hidden layers
            layers.append(self._fc(n_inputs, n_outputs, activation=output_activation,
                                   dropout=dropout if regularize_last_layer else None,
                                   batch_norm=regularize_last_layer))
        else:
            layers.append(self._fc(n_inputs, hiddens[0], activation=activation, dropout=dropout, batch_norm=batch_norm))  # first layer
            for l in range(1, len(hiddens)):  # inner layers
                layers.append(self._fc(hiddens[l-1], hiddens[l], activation=activation, dropout=dropout, batch_norm=batch_norm))

            layers.append(self._fc(hiddens[-1], n_outputs, activation=output_activation,
                                   dropout=dropout if regularize_last_layer else None,
                                   batch_norm=regularize_last_layer))

        self.network = nn.Sequential(*layers)

    def _fc(self, n_inputs, n_outputs, activation='leakyrelu', dropout=None, batch_norm=True):
        layers = [nn.Linear(n_inputs, n_outputs, bias=not batch_norm)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(n_outputs))
        if activation != 'linear':
            layers.append(self._activation(activation))
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def _activation(self, name='leakyrelu'):
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'leakyrelu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'softmax':
            return nn.Softmax()
        elif name == 'exponential':
            return Exponential()
        else:
            raise NotImplementedError(f'activation function {name} is not implemented.')

    def forward(self, x):
        return self.network(x)

    def through(self, x):
        outputs = []
        for layer in self.network:
            x = layer(x)
            outputs.append(x)
        return outputs


class Exponential(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.exp(x)
        return x

# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class TrigonometricPositionalEncoding(nn.Module):
    """
        This trigonometric positional embedding was taken from:
        Title: Sequence-to-Sequence Modeling with nn.Transformer and TorchText
        Date: 17th March 2021
        Availability: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, embedding_dim, dropout, max_len):
        super(TrigonometricPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        temp = self.pe[:x.size(0), :]
        x = x + temp
        return self.dropout(x)

class TransformerDecoder(nn.Module):
    def __init__(self, params, hdim, num_seq_labels, tgt_len):
        """

        :param params: hyperparameters as dict
        :param hdim: input feature dimension
        :param num_seq_labels: number of aa labels, output dim
        """
        super(TransformerDecoder, self).__init__()
        self.params = params
        self.hdim = hdim
        self.num_seq_labels = num_seq_labels
        self.tgt_len = tgt_len

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.fc_upsample = nn.Linear(hdim, self.tgt_len * params['embedding_size'])#nn.Linear(hdim, self.params['max_tcr_length'] * params['embedding_size'])
        #xavier_uniform_(self.fc_upsample.weight)
        self.layernorm = nn.LayerNorm(params['embedding_size'])
        # the embedding size remains constant over all layers

        self.embedding = nn.Embedding(num_seq_labels, params['embedding_size'], padding_idx=0)
        self.positional_encoding = TrigonometricPositionalEncoding(params['embedding_size'],
                                                                   params['dropout'],
                                                                   self.tgt_len)

        decoding_layers = modified_transformer_gated.TransformerDecoderLayer(params['embedding_size'],
                                                                             params['num_heads'],
                                                                             params['embedding_size'] * params['forward_expansion'],
                                                                             params['dropout'],
                                                                             bias=False)
        self.transformer_decoder = transformer_gated.TransformerDecoder(decoding_layers, params['decoding_layers'])

        self.fc_out = nn.Linear(params['embedding_size'], num_seq_labels)

    def forward(self, hidden_state, pep_hidden_state, target_sequence, pep_padding_mask):
        """
        Forward pass of the Decoder module
        :param hidden_state: gex and cond for fusion
        :param target_sequence: Ground truth output tcr
        :param pep_hidden_state: peptide encoding result for cross attention
        :param pep_mask: peptide attention mask
        :param pep_padding_mask: peptide padding mask
        :return:
        """
        hidden_state = self.fc_upsample(hidden_state)
        shape = (hidden_state.shape[0], self.tgt_len, self.params[
            'embedding_size'])  # (hidden_state.shape[0], self.params['max_tcr_length'], self.params['embedding_size'])
        hidden_state = torch.reshape(hidden_state, shape)
        hidden_state = self.layernorm(hidden_state)
        hidden_state = hidden_state.transpose(0, 1)

        # target_sequence = target_sequence[:, :-1]
        target_sequence = target_sequence.transpose(0, 1)

        target_sequence = self.embedding(target_sequence) * math.sqrt(self.num_seq_labels)
        target_sequence = target_sequence + self.positional_encoding(target_sequence)
        target_mask = nn.Transformer.generate_square_subsequent_mask(target_sequence.shape[0]).to(self.device)

        x = self.transformer_decoder(target_sequence, hidden_state, pep_hidden_state, tgt_mask=target_mask,
                                     memory_key_padding_mask=pep_padding_mask)
        x = self.fc_out(x)
        x = x.transpose(0, 1)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, params, num_seq_labels):
        """

        :param params: hyperparameters as dict
        :param hdim: input feature dimension
        :param num_seq_labels: number of aa labels, output dim
        """
        super(TransformerEncoder, self).__init__()
        self.params = params
        self.num_seq_labels = num_seq_labels

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.layernorm = nn.LayerNorm(params['embedding_size'])
        # the embedding size remains constant over all layers

        self.embedding = nn.Embedding(num_seq_labels, params['embedding_size'], padding_idx=0)
        self.positional_encoding = TrigonometricPositionalEncoding(params['embedding_size'],
                                                                   params['dropout'],
                                                                   self.params['max_pep_length'])

        encoding_layers = nn.TransformerEncoderLayer(params['embedding_size'],
                                                     params['num_heads'],
                                                     params['embedding_size'] * params['forward_expansion'],
                                                     params['dropout'],
                                                     )
        self.transformer_encoder = nn.TransformerEncoder(encoding_layers, params['encoding_layers'])


    def forward(self, pep, padding_mask):
        """
        Forward pass of the Encoder module
        :param pep: peptide (bs, pep_max_len)
        :param padding_mask: padding mask for peptides (bs, pep_max_len) bool
        :return:
        """

        pep = pep.transpose(0, 1)
        encode_sequence = self.embedding(pep)*math.sqrt(self.num_seq_labels)
        encode_sequence = encode_sequence + self.positional_encoding(encode_sequence)
        x = self.transformer_encoder(encode_sequence, src_key_padding_mask=padding_mask)

        return x


from transformers import T5ForConditionalGeneration

class condTransformer(nn.Module):
    def __init__(self, encoder_input=4000,
                 encoder_hidden=[1024],
                 encoder_output,
                 cond_encoder_input,
                 cond_encoder_output,
                 encoder_params,
                 decoder_params,
                 decoder_hdim,
                 encoder_num_seq_labels,
                 decoder_num_seq_labels,
                 lrlen,
                 rllen):
        super(condTransformer, self).__init__()
        self.encoder = MLPEncoder(n_inputs=encoder_input, hiddens=encoder_hidden, n_outputs=encoder_output, output_activation='exponential', dropout=0.4, batch_norm=False)

        self.cond_encoder = MLPEncoder(n_inputs=cond_encoder_input, n_outputs=cond_encoder_output, output_activation='exponential', dropout=0.4, batch_norm=True)

        self.pep_encoder = T5ForConditionalGeneration.from_pretrained("./dkarthikeyan1/tcrt5_pre_tcrdb").encoder
        self.pep_downdim = nn.Linear(256, encoder_params['embedding_size'], bias=False)
        self.decoder_lr = TransformerDecoder(decoder_params, decoder_hdim, decoder_num_seq_labels, lrlen)
        self.decoder_rl = TransformerDecoder(decoder_params, decoder_hdim, decoder_num_seq_labels, rllen)
    def forward(self, pep, scrna, tcr, cond):
        pep_encode = self.pep_encoder(pep[0], pep[1])#hiddenstate: bs, len, dim=256
        pep_enc = self.pep_downdim(pep_encode.last_hidden_state).transpose(0, 1)
        scrna_encode = self.encoder(scrna)
        cond_encode = self.cond_encoder(cond)
        hidden_state = torch.concat([scrna_encode, cond_encode], dim=-1)
        out_lr = self.decoder_lr(hidden_state, pep_enc, torch.tensor(tcr[0],dtype=torch.int), pep[1].bool())
        out_rl = self.decoder_rl(hidden_state, pep_enc, torch.tensor(tcr[1],dtype=torch.int), pep[1].bool())

        return out_lr, out_rl