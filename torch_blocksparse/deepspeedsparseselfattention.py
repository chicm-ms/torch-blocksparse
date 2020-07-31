import torch.nn as nn
from torch.nn.functional import *
import torch
from collections import namedtuple
import torch_blocksparse
import sys
import random

class SparsityConfig:
    Modes = ['dense', 'fixed', 'bigbird']
    def __init__(self,
            mode = 'fixed',
            block = 16,
            stride = 64,
            attention = 'unidirectional',
            numverts = 1,
            vertsize = 1,
            rndblocks = 1):
        if not mode in SparsityConfig.Modes:
            raise NotImplementedError(f'only supported modes are: {SparsityConfig.Modes}')
        self.mode = mode
        self.block = block
        self.stride = stride
        if attention != 'unidirectional' and attention != 'bidirectional':
            raise NotImplementedError('only \"uni/bi-directional\" attentions are supported for now')
        self.attention = attention
        self.numverts = numverts
        self.vertsize = vertsize
        self.rndblocks = rndblocks


class DeepSpeedSparseSelfAttention(nn.Module):

    @staticmethod
    def _set_local_fixed_layout(layout, h, num_blocks, block_stride, attention):
        for i in range(0, num_blocks, block_stride):
            for j in range(i, i + block_stride):
                for k in range(i, (j + 1 if attention == 'unidirectional' else i + block_stride)):
                    layout[h, j, k] = 1
        return layout

    @staticmethod
    def _set_global_fixed_layout(layout, h, num_blocks, block_stride, attention, numverts, vertsize):
        start = block_stride - (1 + h % numverts) * vertsize
        for i in range(0, num_blocks):
            end = i if attention == 'unidirectional' else num_blocks
            for j in range(start, end, block_stride):
                for k in range(j, min(j + vertsize, num_blocks)):
                    layout[h, i, k] = 1
        return layout

    @staticmethod
    def _set_layout_bigbird_rnd(layout, h, num_blocks, num_rnd_blocks):
        for row in range(0, num_blocks):
            rnd_cols = random.sample(range(0, num_blocks), num_rnd_blocks)
            layout[h, row, rnd_cols] = 1
        return layout

    @staticmethod
    def _set_layout_bigbird_sliding_window(layout, h, num_blocks, num_window_blocks):
        w = num_window_blocks // 2
        for row in range(0, num_blocks):
            start = max(0, row - w)
            end = min(row + w + 1, num_blocks)
            layout[h, row, start:end] = 1
        return layout

    @staticmethod
    def _set_layout_bigbird_global_itc(layout, h, num_blocks, num_global_blocks):
        #global rows
        layout[h, 0:num_global_blocks, :] = 1

        #global columns 
        layout[h, :, 0:num_global_blocks] = 1
        return layout

    @staticmethod
    def _make_layout_dense(num_heads, num_blocks, block_stride, attention, numverts, vertsize, rndblocks, different_layout_per_head = False):
        layout = torch.ones((num_heads, num_blocks, num_blocks), dtype=torch.int64)
        return layout

    @staticmethod
    def _make_layout_fixed(num_heads, num_blocks, block_stride, attention, numverts, vertsize, rndblocks, different_layout_per_head = True):
        if (block_stride / vertsize) != (block_stride // vertsize):
                raise ValueError(f'Number of blocks in a stride window {block_stride} must be dividable by vertical block size {vertsize}')
        
        if numverts > (block_stride / vertsize):
                raise ValueError(f'Number of layout versions {num_verts} cannot be larger than blocks in a stride window divided by vertical block size {block_stride} / {vertsize} = {block_stride/vertsize}')

        layout = torch.zeros((num_heads, num_blocks, num_blocks), dtype=torch.int64)
        heads = num_heads if different_layout_per_head else 1
        for i in range(0, heads):
            layout = DeepSpeedSparseSelfAttention._set_local_fixed_layout(layout, i, num_blocks, block_stride, attention)
            layout = DeepSpeedSparseSelfAttention._set_global_fixed_layout(layout, i, num_blocks, block_stride, attention, numverts, vertsize)

        if not different_layout_per_head:
            layout[1:num_heads, :, :] = layout[0, :, :]
        return layout

    @staticmethod
    def _make_layout_bigbird(num_heads, num_blocks, num_window_blocks, attention, numverts, num_global_blocks, num_rnd_blocks, different_layout_per_head = False):
        if (num_rnd_blocks > num_blocks):
            raise ValueError(f'Number of random blocks must be <= number of blocks in a row; {num_rnd_blocks} <= {num_blocks}')
        if (num_window_blocks > num_blocks):
            raise ValueError(f'Number of blocks in sliding window must be <= number of blocks in a row; {num_window_blocks} <= {num_blocks}')
        if (num_global_blocks > num_blocks):
            raise ValueError(f'Number of global blocks must be <= number of blocks in a row; {num_global_blocks} <= {num_blocks}')

        layout = torch.zeros((num_heads, num_blocks, num_blocks), dtype=torch.int64)
        heads = num_heads if different_layout_per_head else 1

        for h in range(0, heads):
            layout = DeepSpeedSparseSelfAttention._set_layout_bigbird_rnd(layout, h, num_blocks, num_rnd_blocks)
            layout = DeepSpeedSparseSelfAttention._set_layout_bigbird_sliding_window(layout, h, num_blocks, num_global_blocks)
            layout = DeepSpeedSparseSelfAttention._set_layout_bigbird_global_itc(layout, h, num_blocks, num_global_blocks)

        if not different_layout_per_head:
            layout[1:num_heads, :, :] = layout[0, :, :]
        return layout

    ops = dict()

    layoutCreators = {
        "dense": _make_layout_dense.__func__,
        "fixed": _make_layout_fixed.__func__,
        "bigbird": _make_layout_bigbird.__func__
    }

    @staticmethod
    def _make_layout(num_heads, num_blocks, mode, block_stride, attention, numverts, vertsize, rndblocks, different_layout_per_head = True):
        if not mode in SparsityConfig.Modes:
            raise NotImplementedError(f'only supported modes are: {SparsityConfig.Modes}')
        layout = DeepSpeedSparseSelfAttention.layoutCreators[mode](num_heads, num_blocks, block_stride, attention, numverts, vertsize, rndblocks, different_layout_per_head)
        return layout

    # add to cache
    def get_ops(self, H, L):
        import sys
        if L not in DeepSpeedSparseSelfAttention.ops:
            spConfig = self.sparsity_config

            num_blocks = L // spConfig.block
            if num_blocks != L / spConfig.block:
                raise ValueError(f'Sequence length {L} must be dividable by block size {spConfig.block}')

            block_stride = spConfig.stride // spConfig.block
            if block_stride != spConfig.stride // spConfig.block:
                raise ValueError(f'Stride {spConfig.stride} must be dividable by block size {spConfig.block}')

            layout = DeepSpeedSparseSelfAttention._make_layout(H,
                    num_blocks,
                    spConfig.mode,
                    block_stride,
                    spConfig.attention,
                    spConfig.numverts, 
                    spConfig.vertsize,
                    spConfig.rndblocks)

            sparse_dot_sdd_nt = torch_blocksparse.MatMul(layout,
                    spConfig.block,
                    'sdd',
                    trans_a=False,
                    trans_b=True)

            sparse_dot_dsd_nn = torch_blocksparse.MatMul(layout,
                    spConfig.block,
                    'dsd',
                    trans_a=False,
                    trans_b=False)

            sparse_softmax = torch_blocksparse.Softmax(layout, spConfig.block)

            DeepSpeedSparseSelfAttention.ops[L] = (sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax)
        return DeepSpeedSparseSelfAttention.ops[L]

    # constructor
    def __init__(self, sparsity_config=SparsityConfig(), key_padding_mask_mode='add', attn_mask_mode='mul'):
        super().__init__()

        # sparsity information
        self.sparsity_config = sparsity_config
       
        # mask modes
        self.key_padding_mask_mode = key_padding_mask_mode
        self.attn_mask_mode = attn_mask_mode

    def transpose_key_for_scores(self, x, L):
        bsz, num_heads, seq_len, head_dim = x.size()
        if seq_len != L:
            return x.permute(0, 1, 3, 2)
        return x

    def transpose_mask_for_sparse(self, qtype, x, is_key_padding_mask=False):
        x = x.type(qtype)
        if is_key_padding_mask:
            xdim = x.dim()
            for d in range(xdim - 1, 0, -1):
                x = x.squeeze(dim=d)
            return x
        return x.squeeze()

    # forward pass
    def forward(self, query, key, value, rpe=None, key_padding_mask=None, attn_mask=None):
        bsz, num_heads, tgt_len, head_dim = query.size()
        
        # transpose back key if it is already transposed
        key = self.transpose_key_for_scores(key, tgt_len)

        # check that operation is supported
        if query.shape != key.shape or key.shape != value.shape:
            raise NotImplementedError('only self-attention is supported for now')

        # squeeze key_padding_mask if it is given
        if key_padding_mask is not None:
            key_padding_mask = self.transpose_mask_for_sparse(query.dtype, key_padding_mask, is_key_padding_mask=True)


        # squeeze attn_mask if it is given
        if attn_mask is not None:
            attn_mask = self.transpose_mask_for_sparse(query.dtype, attn_mask)

        # cache look-up table computations etc
        sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = self.get_ops(num_heads, tgt_len)

        scaling = float(head_dim) ** -0.5

        # attention scores
        attn_output_weights = sparse_dot_sdd_nt(query, key)
        attn_output_weights = sparse_softmax(attn_output_weights, scale=scaling, rpe=rpe, key_padding_mask=key_padding_mask, attn_mask=attn_mask,
                key_padding_mask_mode=self.key_padding_mask_mode, attn_mask_mode=self.attn_mask_mode)

        # outputs
        attn_output = sparse_dot_dsd_nn(attn_output_weights, value)
        return attn_output
