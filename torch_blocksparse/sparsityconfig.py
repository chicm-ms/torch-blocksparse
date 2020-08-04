import torch.nn as nn
from torch.nn.functional import *
import torch
from collections import namedtuple
import sys
import random


class SparsityConfig:
    def __init__(self,
            num_heads = 8,
            seq_len = 1024,
            block = 16,
            different_layout_per_head = False):
        self.num_heads = num_heads
        if (seq_len % block != 0):
            raise ValueError(f'Sequence Length, {seq_len}, needs to be dividable by Block size {block}!')
        self.seq_len = seq_len
        self.block = block
        self.num_blocks = seq_len // block
        self.different_layout_per_head = different_layout_per_head
        self.layout = torch.zeros((self.num_heads, self.num_blocks, self.num_blocks), dtype=torch.int64)
        self.num_layout_heads = num_heads if different_layout_per_head else 1

    def check_and_propagate_first_head_layout(self):
        if not self.different_layout_per_head:
            self.layout[1:self.num_heads, :, :] = self.layout[0, :, :]

class DenseSparsityConfig(SparsityConfig):
    def __init__(self,
            num_heads = 8,
            seq_len = 1024,
            block = 16,
            different_layout_per_head = False):
        super().__init__(num_heads, seq_len, block, different_layout_per_head)
        self.make_layout()

    def make_layout(self):
        self.layout[:, :, :] = 1
        return self.layout

class FixedSparsityConfig(SparsityConfig):
    def __init__(self,
            num_heads = 8,
            seq_len = 1024,
            block = 16,
            different_layout_per_head = False,
            stride = 64,
            attention = 'bidirectional',
            num_verts = 1,
            vert_size = 1):  
        super().__init__(num_heads, seq_len, block, different_layout_per_head)

        if (stride % block != 0):
            raise ValueError(f'Stride, {stride}, must be dividable by block size, {block}!')
        if (seq_len % stride != 0):
            raise ValueError(f'Sequence Length, {seq_len}, must be dividable by Stride, {stride}!')
        self.stride = stride
        self.block_stride = stride // block

        if attention != 'unidirectional' and attention != 'bidirectional':
            raise NotImplementedError('only \"uni/bi-directional\" attentions are supported for now!')
        self.attention = attention

        if (num_verts > 1 and not different_layout_per_head):
            raise ValueError(f'Number of different layouts cannot be more than one when you have set a single layout for all heads!')
        if (self.block_stride % vert_size != 0):
            raise ValueError(f'Number of blocks in a stride (local window), {self.block_stride}, must be dividable by vertical block size, {vert_size}!')
        if num_verts > (self.block_stride // vert_size):
                raise ValueError(f'Number of layout versions, {num_verts}, cannot be larger than blocks in a stride window divided by vertical block size, {self.block_stride} / {vert_size} = {self.block_stride//vert_size}!')

        self.num_verts = num_verts
        self.vert_size = vert_size
        self.make_layout()

    def set_local_layout(self, h):
        for i in range(0, self.num_blocks, self.block_stride):
            for j in range(i, i + self.block_stride):
                for k in range(i, (j + 1 if self.attention == 'unidirectional' else i + self.block_stride)):
                    self.layout[h, j, k] = 1

    def set_global_layout(self, h):
        start = self.block_stride - (1 + h % self.num_verts) * self.vert_size
        for i in range(0, self.num_blocks):
            end = i if self.attention == 'unidirectional' else self.num_blocks
            for j in range(start, end, self.block_stride):
                for k in range(j, min(j + self.vert_size, self.num_blocks)):
                    self.layout[h, i, k] = 1

    def make_layout(self):
        for h in range(0, self.num_layout_heads):
            self.set_local_layout(h)
            self.set_global_layout(h)

        self.check_and_propagate_first_head_layout()
        return self.layout

class BigBirdSparsityConfig(SparsityConfig):
    def __init__(self,
            num_heads = 8,
            seq_len = 1024,
            block = 16,
            different_layout_per_head = False,
            num_random_blocks = 1,
            num_sliding_window_blocks = 3,
            num_global_blocks = 1):
        super().__init__(num_heads, seq_len, block, different_layout_per_head)

        if (self.num_blocks < num_random_blocks):
            raise ValueError(f'Number of random blocks, {num_random_blocks}, must be smaller than overal number of blocks in a row, {self.num_blocks}!')
        self.num_random_blocks = num_random_blocks

        if (self.num_blocks < num_sliding_window_blocks):
            raise ValueError(f'Number of sliding window blocks, {num_sliding_window_blocks}, must be smaller than overal number of blocks in a row, {self.num_blocks}!')
        self.num_sliding_window_blocks = num_sliding_window_blocks

        if (self.num_blocks < num_global_blocks):
            raise ValueError(f'Number of global blocks, {num_global_blocks}, must be smaller than overal number of blocks in a row, {self.num_blocks}!')
        self.num_global_blocks = num_global_blocks
        self.make_layout()
 
    def set_random_layout(self, h):
        for row in range(0, self.num_blocks):
            rnd_cols = random.sample(range(0, self.num_blocks), self.num_random_blocks)
            self.layout[h, row, rnd_cols] = 1

    def set_sliding_window_layout(self, h):
        w = self.num_sliding_window_blocks // 2
        for row in range(0, self.num_blocks):
            start = max(0, row - w)
            end = min(row + w + 1, self.num_blocks)
            self.layout[h, row, start:end] = 1

    def set_global_layout_itc(self, h):
        #global rows
        self.layout[h, 0:self.num_global_blocks, :] = 1

        #global columns 
        self.layout[h, :, 0:self.num_global_blocks] = 1

    def make_layout(self):
        for h in range(0, self.num_layout_heads):
            self.set_random_layout(h)
            self.set_sliding_window_layout(h)
            self.set_global_layout_itc(h)

        self.check_and_propagate_first_head_layout()
        return self.layout
