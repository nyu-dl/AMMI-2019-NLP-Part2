import torch
import numpy
import editdistance
from collections import namedtuple
from operator import attrgetter
from global_variables import (
    SOS_IDX,
    SOS_TOKEN,
    EOS_IDX,
    EOS_TOKEN,
    UNK_IDX,
    UNK_TOKEN,
    PAD_IDX,
    PAD_TOKEN,
    SEP_IDX,
    SEP_TOKEN,
    NEAR_INF,
)

import math


class Beam(object):
    def __init__(
        self,
        beam_size=10,
        min_length=3,
        padding_token=PAD_IDX,
        bos_token=SOS_IDX,
        eos_token=EOS_IDX,
        min_n_best=3,
        device="cpu",
        block_ngram=1,
        expand_beam=1,
    ):
        self.beam_size = beam_size
        self.min_length = min_length
        self.min_n_best = min_n_best
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        self.device = device

        self.scores = torch.Tensor(self.beam_size).float().zero_().to(self.device)
        self.all_scores = [torch.Tensor([0.0] * beam_size).to(self.device)]
        self.bookkeep = []
        self.outputs = [
            torch.Tensor(self.beam_size).long().fill_(self.bos).to(self.device)
        ]

        self.finished = []
        self.HypothesisTail = namedtuple(
            "HypothesisTail", ["timestep", "hypid", "score", "tokenid"]
        )
        self.eos_top = False
        self.eos_top_ts = None
        self.n_best_counter = 0
        self.expand_beam = expand_beam
        self.block_ngram = block_ngram
        self.partial_hyps = numpy.array([self.bos for i in range(beam_size)], dtype=numpy.int32).reshape((beam_size, 1))

    def get_edits(self, array1, array2):
        ar1size = array1.shape[0]
        ar2size = array2.shape[0]

        result = numpy.zeros(shape=(ar1size, ar2size), dtype=numpy.int)
        for i in range(ar1size):
            for j in range(ar2size):
                result[i, j] = editdistance.eval(array1[i], array2[j])

        return result

    def do_block_ngram(self, i, n):
        if n > 0:
            for n_gram in range(2, n + 1):
                current_hypo = self.partial_hyps[i, 1:]
                ngrams = set()
                gram = []
                for l in range(len(current_hypo)):
                    gram = (gram + [current_hypo[l].item()])[-n_gram:]
                    if tuple(gram) in ngrams:
                        return True
                    else:
                        ngrams.add(tuple(gram))
        return False

    def get_output_from_current_step(self):
        """Get the outputput at the current step."""
        return self.outputs[-1]

    def get_backtrack_from_current_step(self):
        """Get the backtrack at the current step."""
        return self.bookkeep[-1]

    def advance(self, softmax_probs):
        voc_size = softmax_probs.size(-1)
        current_length = len(self.all_scores) - 1
        if current_length < self.min_length:
            # penalize all eos probs to make it decode longer
            for hyp_id in range(softmax_probs.size(0)):
                softmax_probs[hyp_id][self.eos] = -NEAR_INF

        if len(self.bookkeep) == 0:
            # the first step we take only the first hypo into account since all
            # hypos are the same initially
            beam_scores = softmax_probs[0]
        else:
            beam_scores = softmax_probs + self.scores.unsqueeze(1).expand_as(
                softmax_probs
            )

        for i in range(self.outputs[-1].size(0)):
            if self.do_block_ngram(i, self.block_ngram):
                beam_scores[i] = -NEAR_INF

        flatten_beam_scores = beam_scores.view(-1)  # [beam_size * voc_size]
        if self.expand_beam == 1:
            with torch.no_grad():
                best_scores, best_idxs = torch.topk(
                    flatten_beam_scores, self.beam_size, dim=-1
                )
        elif self.expand_beam > 1:
            with torch.no_grad():
                # first stage
                best_scores, best_idxs = torch.topk(
                    flatten_beam_scores, self.beam_size * self.expand_beam, dim=-1
                )
            # get the backtracking hypothesis id as a multiple of full voc_sizes
            large_hyp_ids = numpy.array(best_idxs.cpu() / voc_size, dtype=numpy.int32)
            # get the actual word id from residual of the same division
            large_tok_ids = numpy.array(best_idxs.cpu() % voc_size, dtype=numpy.int32)

            prefixes = self.partial_hyps.take(large_hyp_ids, axis=0)
            partials = numpy.concatenate([prefixes, large_tok_ids[:, None]], axis=1)
            original_ids = numpy.arange(0, large_hyp_ids.size, dtype=numpy.int32)

            remove_mask = numpy.ones(large_hyp_ids.shape, dtype=bool)  #
            remove_mask[0] = False

            chosen_partials = partials[
                numpy.logical_not(remove_mask)
            ]  # we start from the best partial w.r.t. the model
            chosen_hyp_ids = large_hyp_ids[numpy.logical_not(remove_mask)]
            chosen_original_ids = original_ids[numpy.logical_not(remove_mask)]

            left_partials = partials[remove_mask]
            left_hyp_ids = large_hyp_ids[remove_mask]
            left_original_ids = original_ids[remove_mask]

            for i in range(self.beam_size - 1):
                # edit_matrix = self.get_cross_edit_distance(left_partials, chosen_partials)
                edit_matrix = self.get_edits(
                    left_partials,
                    chosen_partials
                ).reshape([left_partials.shape[0], chosen_partials.shape[0]])
                avg_edits = edit_matrix.mean(
                    1
                )  # avg dist for each left w.r.t all chosen
                most_distant_hyp_ids = left_original_ids[numpy.argmax(avg_edits)]
                remove_mask[most_distant_hyp_ids] = False

                chosen_partials = partials[
                    numpy.logical_not(remove_mask)
                ]  # we start from the best partial w.r.t. the model
                chosen_hyp_ids = large_hyp_ids[numpy.logical_not(remove_mask)]
                chosen_original_ids = original_ids[numpy.logical_not(remove_mask)]

                left_partials = partials[remove_mask]
                left_hyp_ids = large_hyp_ids[remove_mask]
                left_original_ids = original_ids[remove_mask]

            best_scores = best_scores[chosen_original_ids]
            best_idxs = best_idxs[chosen_original_ids]

        self.scores = best_scores
        self.all_scores.append(self.scores)

        # get the backtracking hypothesis id as a multiple of full voc_sizes
        hyp_ids = best_idxs / voc_size
        # get the actual word id from residual of the same division
        tok_ids = best_idxs % voc_size

        self.outputs.append(tok_ids)
        self.bookkeep.append(hyp_ids)

        reorder_hyps = self.partial_hyps.take(hyp_ids.cpu().numpy(), axis=0)
        self.partial_hyps = numpy.concatenate([reorder_hyps, tok_ids.cpu().int().numpy()[:,None]], axis=1)

        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                #  this is finished hypo, adding to finished
                eostail = self.HypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=hypid,
                    score=self.scores[hypid],
                    tokenid=self.eos,
                )
                self.finished.append(eostail)
                self.n_best_counter += 1

        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) - 1

    def done(self):
        """Return whether beam search is complete."""
        return self.eos_top and self.n_best_counter >= self.min_n_best

    def get_top_hyp(self):
        """Get single best hypothesis.
        :return: hypothesis sequence and the final score
        """
        top_hypothesis_tail = self.get_rescored_finished(n_best=1)[0]
        return (
            self.get_hyp_from_finished(top_hypothesis_tail),
            top_hypothesis_tail.score,
        )

    def get_hyp_from_finished(self, hypothesis_tail):
        """Extract hypothesis ending with EOS at timestep with hyp_id.
        :param timestep: timestep with range up to len(self.outputs)-1
        :param hyp_id: id with range up to beam_size-1
        :return: hypothesis sequence
        """
        assert self.outputs[hypothesis_tail.timestep][hypothesis_tail.hypid] == self.eos
        assert hypothesis_tail.tokenid == self.eos
        hyp_idx = []
        endback = hypothesis_tail.hypid
        for i in range(hypothesis_tail.timestep, -1, -1):
            hyp_idx.append(
                self.HypothesisTail(
                    timestep=i,
                    hypid=endback,
                    score=self.all_scores[i][endback],
                    tokenid=self.outputs[i][endback],
                )
            )
            endback = self.bookkeep[i - 1][endback]

        return hyp_idx

    @staticmethod
    def get_length_penalty(length):
        return math.pow((1 + length) / 6, 0.65)

    @staticmethod
    def get_pretty_hypothesis(list_of_hypotails):
        """Return prettier version of the hypotheses."""
        hypothesis = []
        for i in list_of_hypotails:
            hypothesis.append(i.tokenid)

        hypothesis = torch.stack(list(reversed(hypothesis)))
        return hypothesis

    def get_rescored_finished(self, n_best=None):
        """Return finished hypotheses in rescored order.
        :param n_best: how many n best hypothesis to return
        :return: list with hypothesis
        """
        rescored_finished = []
        for finished_item in self.finished:
            current_length = finished_item.timestep
            # these weights are from Google NMT paper
            length_penalty = Beam.get_length_penalty(current_length)
            rescored_finished.append(
                self.HypothesisTail(
                    timestep=finished_item.timestep,
                    hypid=finished_item.hypid,
                    score=finished_item.score / length_penalty,
                    tokenid=finished_item.tokenid,
                )
            )

        srted = sorted(rescored_finished, key=attrgetter("score"), reverse=True)

        if n_best is not None:
            srted = srted[:n_best]

        return srted

    def check_finished(self):
        """Check if self.finished is empty and add hyptail in that case.
        This will be suboptimal hypothesis since the model did not get any EOS
        :returns: None
        """
        if len(self.finished) == 0:
            # we change output because we want outputs to have eos
            # it is ok since empty self.finished
            # means junk prediction anyway
            self.outputs[-1][0] = self.eos
            hyptail = self.HypothesisTail(
                timestep=len(self.outputs) - 1,
                hypid=0,
                score=self.all_scores[-1][0],
                tokenid=self.outputs[-1][0],
            )

            self.finished.append(hyptail)

    def get_beam_dot(self, dictionary=None, n_best=None):
        """Create pydot graph representation of the beam.

        :param outputs: self.outputs from the beam
        :param dictionary: tok 2 word dict to save words in the tree nodes
        :returns: pydot graph
        """
        try:
            import pydot
        except ImportError:
            print("Please install pydot package to dump beam visualization")

        graph = pydot.Dot(graph_type="digraph", rankdir="LR")
        outputs = [i.tolist() for i in self.outputs]
        bookkeep = [i.tolist() for i in self.bookkeep]
        all_scores = [i.tolist() for i in self.all_scores]
        if n_best is None:
            n_best = int(self.beam_size / 2)

        # get top nbest hyp
        top_hyp_idx_n_best = []
        n_best_colors = ["aquamarine", "chocolate1", "deepskyblue", "green2", "tan"]
        sorted_finished = self.get_rescored_finished(n_best=n_best)
        for hyptail in sorted_finished:
            # do not include EOS since it has rescored score not from original
            # self.all_scores, we color EOS with black
            top_hyp_idx_n_best.append(self.get_hyp_from_finished(hyptail))

        # create nodes
        for tstep, lis in enumerate(outputs):
            for hypid, token in enumerate(lis):
                if tstep == 0:
                    hypid = 0  # collapse all __NULL__ nodes
                node_tail = self.HypothesisTail(
                    timestep=tstep,
                    hypid=hypid,
                    score=all_scores[tstep][hypid],
                    tokenid=token,
                )
                color = "white"
                rank = None
                for i, hypseq in enumerate(top_hyp_idx_n_best):
                    if node_tail in hypseq:
                        if n_best <= 5:  # color nodes only if <=5
                            color = n_best_colors[i]
                        rank = i
                        break
                label = (
                    "<{}".format(
                        dictionary.ind2word[token].replace("<", "").replace(">", "")
                        if dictionary is not None
                        else token
                    )
                    + " : "
                    + "{:.{prec}f}>".format(all_scores[tstep][hypid], prec=3)
                )

                graph.add_node(
                    pydot.Node(
                        node_tail.__repr__(),
                        label=label,
                        fillcolor=color,
                        style="filled",
                        xlabel="{}".format(rank) if rank is not None else "",
                    )
                )

        # create edges
        for revtstep, lis in reversed(list(enumerate(bookkeep))):
            for i, prev_id in enumerate(lis):
                from_node = graph.get_node(
                    '"{}"'.format(
                        self.HypothesisTail(
                            timestep=revtstep,
                            hypid=prev_id,
                            score=all_scores[revtstep][prev_id],
                            tokenid=outputs[revtstep][prev_id],
                        ).__repr__()
                    )
                )[0]
                to_node = graph.get_node(
                    '"{}"'.format(
                        self.HypothesisTail(
                            timestep=revtstep + 1,
                            hypid=i,
                            score=all_scores[revtstep + 1][i],
                            tokenid=outputs[revtstep + 1][i],
                        ).__repr__()
                    )
                )[0]
                newedge = pydot.Edge(from_node.get_name(), to_node.get_name())
                graph.add_edge(newedge)

        return graph
