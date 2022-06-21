# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Modified from /home/alta/pronunciation/tools/fairseq/fairseq/data/add_target_dataset.py
# Modified: __init__, __getitem__, collater
# Mengjie Qian, 2022-05-27

from cProfile import label
import torch
import numpy as np

from . import BaseWrapperDataset, data_utils
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel


class AddTargetDatasetExtended(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        labels,
        pad,
        eos,
        batch_targets,
        process_label=None,
        label_len_fn=None,
        add_to_input=False,
        text_compression_level=TextCompressionLevel.none,
        start_boundary_idx=None,
        end_boundary_idx=None,
        targets=None,
        pad_ext=None
    ):
        super().__init__(dataset)
        self.labels = labels
        self.batch_targets = batch_targets
        self.pad = pad
        self.eos = eos
        self.process_label = process_label
        self.label_len_fn = label_len_fn
        self.add_to_input = add_to_input
        self.text_compressor = TextCompressor(level=text_compression_level)
        self.start_boundary_idx=start_boundary_idx
        self.end_boundary_idx=end_boundary_idx
        self.targets=targets
        self.pad_ext = pad_ext      # used to pad start_idx, end_idx, target; in __init__(), non-default follows default

    def get_label(self, index, process_fn=None):
        lbl = self.labels[index]
        lbl = self.text_compressor.decompress(lbl)
        return lbl if process_fn is None else process_fn(lbl)

    def __getitem__(self, index):
        item = self.dataset[index]
        item["label"] = self.get_label(index, process_fn=self.process_label)
        item["start_idx"] = torch.Tensor(self.start_boundary_idx[index])
        item["end_idx"] = torch.Tensor(self.end_boundary_idx[index])
        item["target"] = torch.Tensor(self.targets[index])
        # item["padding_mask"] = torch.Tensor(self.targets[padding_mask])
        # item["padding_mask"] = torch.Tensor([True if item == self.pad_ext else False for item in self.targets[index]])
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = self.label_len_fn(self.get_label(index))
        return sz, own_sz

    def get_padding_mask(self, input, pad_idx=-1):
        B, T = input.shape
        mask = torch.zeros((B,T), dtype=torch.bool).to(input.device)
        for b in range(B):
            for t in range(T):
                if input[b, t] != pad_idx:
                    mask[b, t] = False
                else:
                    mask[b, t] = True
        return mask

    def collater(self, samples):
        # print('mq227: AddTargetDatasetExtended.collater, checking collated ...')
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        label = [s["label"] for s in samples if s["id"] in indices]             # label is the old target, before we have only [target], now [target, start_idx, end_idx, label]
        start_idx = [s["start_idx"] for s in samples if s["id"] in indices]
        end_idx = [s["end_idx"] for s in samples if s["id"] in indices]
        target = [s["target"] for s in samples if s["id"] in indices]

        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
            label = data_utils.collate_tokens(label, pad_idx=self.pad, left_pad=False)      # padding target, e.g. [67, 62] -> [67, 67]
            collated["ntokens"] = collated["target_lengths"].sum().item()                   # the real num_tokens
            # padding start_idx, end_idx, target, 
            start_idx = data_utils.collate_tokens(start_idx, pad_idx=self.pad_ext, left_pad=False)    # pad_idx for start_idx, end_idx and target
            end_idx = data_utils.collate_tokens(end_idx, pad_idx=self.pad_ext, left_pad=False)
            target = data_utils.collate_tokens(target, pad_idx=self.pad_ext, left_pad=False)
            padding_mask = self.get_padding_mask(start_idx, pad_idx=self.pad_ext)
        else:
            collated["ntokens"] = sum([len(t) for t in target])

        collated["label"] = label
        collated["start_idx"] = start_idx
        collated["end_idx"] = end_idx
        collated["target"] = target
        collated["padding_mask"] = padding_mask

        if self.add_to_input:
            eos = target.new_full((target.size(0), 1), self.eos)
            collated["target"] = torch.cat([target, eos], dim=-1).long()                    # add an eos to the end of each sample, [[x11,x12, ...],[x21,x22,...,-1]] -> [[x11,x12, ...,2],[x21,x22,...,-1,2]] 
            collated["start_idx"] = torch.cat([start_idx, eos], dim=-1).long() 
            collated["end_idx"] = torch.cat([end_idx, eos], dim=-1).long() 
            collated["label"] = torch.cat([label, eos], dim=-1).long() 
            collated["net_input"]["prev_output_tokens"] = torch.cat(
                [eos, target], dim=-1
            ).long()
            collated["ntokens"] += target.size(0)
            collated["padding_mask"] = torch.cat([padding_mask, eos], dim=-1).long() 
        # print('mq227: AddTargetDatasetExtended.collater, passed check')
        return collated

    def filter_indices_by_size(self, indices, max_sizes):
        indices, ignored = data_utils._filter_by_size_dynamic(
            indices, self.size, max_sizes
        )
        return indices, ignored
