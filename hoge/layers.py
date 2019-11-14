import numpy as np
from sklearn.utils import murmurhash3_32
import torch
import torch.nn as nn


SEEDS = [
    179424941, 179425457, 179425907, 179426369,
    179424977, 179425517, 179425943, 179426407,
    179424989, 179425529, 179425993, 179426447,
    179425003, 179425537, 179426003, 179426453,
    179425019, 179425559, 179426029, 179426491,
    179425027, 179425579, 179426081, 179426549
]


class ScaledEmbedding(nn.Embedding):
    def reset_parameters(self):
        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    def reset_parameters(self):
        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ScaledEmbeddingBag(nn.EmbeddingBag):
    def reset_parameters(self):
        self.weight.data.normal_(0, 1.0 / self.embedding_dim)


class BloomEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 compression_ratio=0.2,
                 num_hash_functions=4,
                 bag=False,
                 padding_idx=0):

        super(BloomEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.compression_ratio = compression_ratio
        self.compressed_num_embeddings = int(compression_ratio *
                                             num_embeddings)
        self.num_hash_functions = num_hash_functions
        self.padding_idx = padding_idx
        self._bag = bag

        if num_hash_functions > len(SEEDS):
            raise ValueError('Can use at most {} hash functions ({} requested)'
                             .format(len(SEEDS), num_hash_functions))

        self._masks = SEEDS[:self.num_hash_functions]

        if self._bag:
            self.embeddings = ScaledEmbeddingBag(self.compressed_num_embeddings,
                                                 self.embedding_dim,
                                                 mode='sum')
        else:
            self.embeddings = ScaledEmbedding(self.compressed_num_embeddings,
                                              self.embedding_dim,
                                              padding_idx=self.padding_idx)

        self._hashes = None
        self._offsets = None

    def __repr__(self):

        return ('<BloomEmbedding (compression_ratio: {}): {}>'
                .format(self.compression_ratio,
                        repr(self.embeddings)))

    def _get_hashed_indices(self, original_indices):

        def _hash(x, seed):

            # TODO: integrate with padding index
            result = murmurhash3_32(x, seed=seed)
            result[self.padding_idx] = 0

            return result % self.compressed_num_embeddings

        if self._hashes is None:
            indices = np.arange(self.num_embeddings, dtype=np.int32)
            hashes = np.stack([_hash(indices, seed)
                               for seed in self._masks],
                              axis=1).astype(np.int64)
            assert hashes[self.padding_idx].sum() == 0

            self._hashes = torch.from_numpy(hashes)

            if original_indices.is_cuda:
                self._hashes = self._hashes.cuda()

        hashed_indices = torch.index_select(self._hashes,
                                            0,
                                            original_indices.squeeze())

        return hashed_indices

    def forward(self, indices):
        if indices.dim() == 2:
            batch_size, seq_size = indices.size()
        else:
            batch_size, seq_size = indices.size(0), 1

        if not indices.is_contiguous():
            indices = indices.contiguous()

        indices = indices.data.view(batch_size * seq_size, 1)

        if self._bag:
            if (self._offsets is None or
                    self._offsets.size(0) != (batch_size * seq_size)):

                self._offsets = torch.arange(0,
                                             indices.numel(),
                                             indices.size(1)).long()

                if indices.is_cuda:
                    self._offsets = self._offsets.cuda()

            hashed_indices = self._get_hashed_indices(indices)
            embedding = self.embeddings(hashed_indices.view(-1), self._offsets)
            embedding = embedding.view(batch_size, seq_size, -1)
        else:
            hashed_indices = self._get_hashed_indices(indices)

            embedding = self.embeddings(hashed_indices)
            embedding = embedding.sum(1)
            embedding = embedding.view(batch_size, seq_size, -1)

        return embedding
