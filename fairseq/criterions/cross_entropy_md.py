# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import torch


@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("cross_entropy_md", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyCriterionMispronunciationDectection(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # import pdb;pdb.set_trace()
        # net_output = model(phones_start_time=sample["start_idx"], phones_end_time=sample["end_idx"], phones=sample['label'], **sample["net_input"])
        # loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        net_output = model(sample)
        loss = self.compute_loss_md(net_output, sample, reduce=reduce)
        # import pdb;pdb.set_trace()
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss_md(self, net_output, sample, reduce=True):
        """Cross entropy loss for mispronunciation detection
        Loss function: L = -(tlog(y)+(1-t)log(1-y))
        t=0: correct pronunciation, t=1: mispronunciation, t=-1: padded (changed to 0 to calculate the loss)
        net_output: B x T, target: B x T
        how to deal with padded numbers (y=1e-10, t=-1) -> (y=1e-10, t=0)
        
        2 methods to use the BCE loss: 
        (1) loss = torch.nn.BCELoss(reduction="sum"); loss = loss(net_output, target)
        (2) loss = torch.nn.functional.binary_cross_entropy(net_output, target, weight=None, reduction="sum" if reduce else "none")
        """
        target = sample["target"]
        B, T = target.shape
        assert B == net_output.size(0) and T == net_output.size(1)
        # process the -1 in target
        for b in range(B):
            for i in range(T):
                if target[b, i] == -1:
                    target[b, i] = 0
        # import pdb;pdb.set_trace()
        loss = F.binary_cross_entropy(net_output, target, weight=None, reduction="sum" if reduce else "none")
        # import pdb;pdb.set_trace()
        return loss

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
