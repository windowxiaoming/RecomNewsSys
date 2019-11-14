#-*-coding:utf-8-*-
import torch
from torch.backends import cudnn
import torch.nn as nn
import torch.nn.functional as F

from hoge.layers import ScaledEmbedding, ZeroEmbedding


PADDING_IDX = 0


def _to_iterable(val, num):

    try:
        iter(val)
        return val
    except TypeError:
        return (val,) * num


class PoolNet(nn.Module):
    def __init__(self, num_items, embedding_dim=32,
                 item_embedding_layer=None, sparse=False):

        super(PoolNet, self).__init__()

        self.embedding_dim = embedding_dim

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   padding_idx=PADDING_IDX,
                                                   sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

    def user_representation(self, item_sequences):
        # Make the embedding dimension the channel dimension
        sequence_embeddings = (self.item_embeddings(item_sequences)
                               .permute(0, 2, 1))

        # Add a trailing dimension of 1
        sequence_embeddings = (sequence_embeddings
                               .unsqueeze(3))

        # Pad it with zeros from left
        sequence_embeddings = F.pad(sequence_embeddings,
                                    (0, 0, 1, 0))

        # Average representations, ignoring padding.
        sequence_embedding_sum = torch.cumsum(sequence_embeddings, 2)
        non_padding_entries = (
            torch.cumsum((sequence_embeddings != 0.0).float(), 2)
            .expand_as(sequence_embedding_sum)
        )

        user_representations = (
            sequence_embedding_sum / (non_padding_entries + 1)
        ).squeeze(3)

        return user_representations[:, :, :-1], user_representations[:, :, -1]

    def forward(self, user_representations, targets):

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1)
                            .squeeze())
        target_bias = self.item_biases(targets).squeeze()

        dot = ((user_representations * target_embedding)
               .sum(1))

        return target_bias + dot


class LSTMNet(nn.Module):
    def __init__(self, num_items, embedding_dim=32,
                 item_embedding_layer=None, sparse=False):

        super(LSTMNet, self).__init__()

        self.embedding_dim = embedding_dim

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   padding_idx=PADDING_IDX,
                                                   sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

        self.lstm = nn.LSTM(batch_first=True,
                            input_size=embedding_dim,
                            hidden_size=embedding_dim)

    def user_representation(self, item_sequences):

        # Make the embedding dimension the channel dimension
        sequence_embeddings = (self.item_embeddings(item_sequences)
                               .permute(0, 2, 1))
        # Add a trailing dimension of 1
        sequence_embeddings = (sequence_embeddings
                               .unsqueeze(3))
        # Pad it with zeros from left
        sequence_embeddings = (F.pad(sequence_embeddings,
                                     (0, 0, 1, 0))
                               .squeeze(3))
        sequence_embeddings = sequence_embeddings.permute(0, 2, 1)

        user_representations, _ = self.lstm(sequence_embeddings)
        user_representations = user_representations.permute(0, 2, 1)

        return user_representations[:, :, :-1], user_representations[:, :, -1]

    def forward(self, user_representations, targets):
        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1)
                            .squeeze())
        target_bias = self.item_biases(targets).squeeze()

        dot = ((user_representations * target_embedding)
               .sum(1)
               .squeeze())

        return target_bias + dot


class CNNNet(nn.Module):
    def __init__(self, num_items,
                 embedding_dim=32,
                 kernel_width=3,
                 dilation=1,
                 num_layers=1,
                 nonlinearity='tanh',
                 residual_connections=True,
                 sparse=False,
                 benchmark=True,
                 item_embedding_layer=None):

        super(CNNNet, self).__init__()

        cudnn.benchmark = benchmark

        self.embedding_dim = embedding_dim
        self.kernel_width = _to_iterable(kernel_width, num_layers)
        self.dilation = _to_iterable(dilation, num_layers)
        if nonlinearity == 'tanh':
            self.nonlinearity = F.tanh
        elif nonlinearity == 'relu':
            self.nonlinearity = F.relu
        else:
            raise ValueError('Nonlinearity must be one of (tanh, relu)')
        self.residual_connections = residual_connections

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   padding_idx=PADDING_IDX,
                                                   sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

        self.cnn_layers = [
            nn.Conv2d(embedding_dim,
                      embedding_dim,
                      (_kernel_width, 1),
                      dilation=(_dilation, 1)) for
            (_kernel_width, _dilation) in zip(self.kernel_width,
                                              self.dilation)
        ]

        for i, layer in enumerate(self.cnn_layers):
            self.add_module('cnn_{}'.format(i),
                            layer)

    def user_representation(self, item_sequences):
        # Make the embedding dimension the channel dimension
        sequence_embeddings = (self.item_embeddings(item_sequences)
                               .permute(0, 2, 1))
        # Add a trailing dimension of 1
        sequence_embeddings = (sequence_embeddings
                               .unsqueeze(3))

        # Pad so that the CNN doesn't have the future
        # of the sequence in its receptive field.
        receptive_field_width = (self.kernel_width[0] +
                                 (self.kernel_width[0] - 1) *
                                 (self.dilation[0] - 1))

        x = F.pad(sequence_embeddings,
                  (0, 0, receptive_field_width, 0))
        x = self.nonlinearity(self.cnn_layers[0](x))

        if self.residual_connections:
            residual = F.pad(sequence_embeddings,
                             (0, 0, 1, 0))
            x = x + residual

        for (cnn_layer, kernel_width, dilation) in zip(self.cnn_layers[1:],
                                                       self.kernel_width[1:],
                                                       self.dilation[1:]):
            receptive_field_width = (kernel_width +
                                     (kernel_width - 1) *
                                     (dilation - 1))
            residual = x
            x = F.pad(x, (0, 0, receptive_field_width - 1, 0))
            x = self.nonlinearity(cnn_layer(x))

            if self.residual_connections:
                x = x + residual

        x = x.squeeze(3)

        return x[:, :, :-1], x[:, :, -1]

    def forward(self, user_representations, targets):

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1)
                            .squeeze())
        target_bias = self.item_biases(targets).squeeze()

        dot = ((user_representations * target_embedding)
               .sum(1)
               .squeeze())

        return target_bias + dot


class MixtureLSTMNet(nn.Module):
    def __init__(self, num_items,
                 embedding_dim=32,
                 num_mixtures=4,
                 item_embedding_layer=None,
                 sparse=False):

        super(MixtureLSTMNet, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_mixtures = num_mixtures

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(num_items, embedding_dim,
                                                   padding_idx=PADDING_IDX,
                                                   sparse=sparse)

        self.item_biases = ZeroEmbedding(num_items, 1, sparse=sparse,
                                         padding_idx=PADDING_IDX)

        self.lstm = nn.LSTM(batch_first=True,
                            input_size=embedding_dim,
                            hidden_size=embedding_dim)
        self.projection = nn.Conv1d(embedding_dim,
                                    embedding_dim * self.num_mixtures * 2,
                                    kernel_size=1)

    def user_representation(self, item_sequences):

        batch_size, sequence_length = item_sequences.size()

        # Make the embedding dimension the channel dimension
        sequence_embeddings = (self.item_embeddings(item_sequences)
                               .permute(0, 2, 1))
        # Add a trailing dimension of 1
        sequence_embeddings = (sequence_embeddings
                               .unsqueeze(3))
        # Pad it with zeros from left
        sequence_embeddings = (F.pad(sequence_embeddings,
                                     (0, 0, 1, 0))
                               .squeeze(3))
        sequence_embeddings = sequence_embeddings
        sequence_embeddings = sequence_embeddings.permute(0, 2, 1)

        user_representations, _ = self.lstm(sequence_embeddings)
        user_representations = user_representations.permute(0, 2, 1)
        user_representations = self.projection(user_representations)
        user_representations = user_representations.view(batch_size,
                                                         self.num_mixtures * 2,
                                                         self.embedding_dim,
                                                         sequence_length + 1)

        return user_representations[:, :, :, :-1], user_representations[:, :, :, -1:]

    def forward(self, user_representations, targets):
        user_components = user_representations[:, :self.num_mixtures, :, :]
        mixture_vectors = user_representations[:, self.num_mixtures:, :, :]

        target_embedding = (self.item_embeddings(targets)
                            .permute(0, 2, 1))
        target_bias = self.item_biases(targets).squeeze()

        mixture_weights = (mixture_vectors * target_embedding
                           .unsqueeze(1)
                           .expand_as(user_components))
        mixture_weights = (F.softmax(mixture_weights.sum(2), 1)
                           .unsqueeze(2)
                           .expand_as(user_components))
        weighted_user_representations = (mixture_weights * user_components).sum(1)

        dot = ((weighted_user_representations * target_embedding)
               .sum(1)
               .squeeze())

        return target_bias + dot
