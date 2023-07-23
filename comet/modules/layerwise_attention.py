# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Layer-Wise Attention Mechanism
================================
    Computes a parameterised scalar mixture of N tensors,
        `mixture = gamma * sum(s_k * tensor_k)`
    where `s = softmax(w)`, with `w` and `gamma` scalar parameters.
    If `layer_norm=True` then apply layer normalization.
    If `dropout > 0`, then for each scalar weight, adjust its softmax
    weight mass to 0 with the dropout probability (i.e., setting the
    unnormalized weight to -inf). This effectively should redistribute
    dropped probability mass to all other weights.
    Original implementation:
        - https://github.com/Hyperparticle/udify
"""
from typing import List, Optional

import torch
from torch.nn import Parameter, ParameterList


class LayerwiseAttention(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        layer_norm: bool = False,
        layer_weights: Optional[List[int]] = None,
        dropout: float = None,
        layer_transformation: str = "softmax",
        batch_level: bool = False,
        bias: bool = False,
    ) -> None:
        super(LayerwiseAttention, self).__init__()
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        self.dropout = dropout
        self.batch_level = batch_level
        self.bias = bias

        self.transform_fn = torch.softmax
        if layer_transformation == "sparsemax":
            from entmax import sparsemax
            self.transform_fn = sparsemax

        if layer_weights is None:
            layer_weights = [0.0] * num_layers
        elif len(layer_weights) != num_layers:
            raise Exception(
                "Length of layer_weights {} differs \
                from num_layers {}".format(
                    layer_weights, num_layers
                )
            )

        self.scalar_parameters = ParameterList(
            [
                Parameter(
                    torch.FloatTensor([layer_weights[i]]),
                    requires_grad=True,
                )
                for i in range(num_layers)
            ]
        )

        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        if self.bias:
            self.beta = Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        else:
            self.beta = None

        if self.dropout:
            dropout_mask = torch.zeros(len(self.scalar_parameters))
            dropout_fill = torch.empty(len(self.scalar_parameters)).fill_(-1e20)
            self.register_buffer("dropout_mask", dropout_mask)
            self.register_buffer("dropout_fill", dropout_fill)

    def forward(
        self,
        tensors: List[torch.Tensor],  # pylint: disable=arguments-differ
        mask: torch.Tensor = None,
    ) -> torch.Tensor:

        if len(tensors) != self.num_layers:
            raise Exception(
                "{} tensors were passed, but the module was initialized to \
                mix {} tensors.".format(
                    len(tensors), self.num_layers
                )
            )

        def _layer_norm(tensor, broadcast_mask, mask):
            tensor_masked = tensor * broadcast_mask
            batch_size, _, input_dim = tensors[0].size()
            num_elements_not_masked = torch.tensor(
                [mask[i].sum() * input_dim for i in range(batch_size)],
                device=tensor.device,
            )
            # mean for each sentence
            mean = torch.sum(torch.sum(tensor_masked, dim=2), dim=1)
            mean = mean / num_elements_not_masked

            variance = torch.vstack(
                [
                    torch.sum(((tensor_masked[i] - mean[i]) * broadcast_mask[i]) ** 2)
                    / num_elements_not_masked[i]
                    for i in range(batch_size)
                ]
            )
            normalized_tensor = torch.vstack(
                [
                    ((tensor[i] - mean[i]) / torch.sqrt(variance[i] + 1e-12)).unsqueeze(
                        0
                    )
                    for i in range(batch_size)
                ]
            )
            return normalized_tensor

        def _layer_norm_batch(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = (
                torch.sum(((tensor_masked - mean) * broadcast_mask) ** 2)
                / num_elements_not_masked
            )
            return (tensor - mean) / torch.sqrt(variance + 1e-12)

        # BUG: Pytorch bug fix when Parameters are not well copied across GPUs
        # https://github.com/pytorch/pytorch/issues/36035
        if len([parameter for parameter in self.scalar_parameters]) != self.num_layers:
            weights = torch.tensor(self.weights, device=tensors[0].device)
            gamma = torch.tensor(self.gamma_value, device=tensors[0].device)
        else:
            weights = torch.cat([parameter for parameter in self.scalar_parameters])
            gamma = self.gamma
            beta = self.beta

        if self.training and self.dropout:
            weights = torch.where(
                self.dropout_mask.uniform_() > self.dropout, weights, self.dropout_fill
            )

        normed_weights = self.transform_fn(weights, dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        if not self.layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * tensor)
            return gamma * sum(pieces)

        else:
            mask_float = mask.float()
            broadcast_mask = mask_float.unsqueeze(-1)
            if self.batch_level:
                input_dim = tensors[0].size(-1)
                num_elements_not_masked = torch.sum(mask_float) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                if self.batch_level:
                    pieces.append(
                        weight
                        * _layer_norm_batch(tensor, broadcast_mask, num_elements_not_masked)
                    )
                else:
                    pieces.append(weight * _layer_norm(tensor, broadcast_mask, mask_float))
            if beta is not None:
                return gamma * sum(pieces) + beta
            else:
                return gamma * sum(pieces)