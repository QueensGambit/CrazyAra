import abc
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from abc import ABCMeta


class PommerModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, nb_input_channels, board_width, board_height, is_stateful, state_batch_dim):
        """
        Create a PommerModel.

        :param is_stateful: Whether the model is stateful (additionally receives and produces a state)
        :param state_batch_dim: The batch dimension of the state (required for flattening if model is stateful)
        """
        super().__init__()
        self.is_stateful = is_stateful
        self.state_batch_dim = state_batch_dim

        self.sequence_length = None
        self.has_state_input = None
        self.nb_input_channels = nb_input_channels
        self.board_width = board_width
        self.board_height = board_height

    def get_init_state_bf_flat(self, batch_size: int, device):
        """
        Get an initial state for the given batch size with the batch dimension first and flattened
        (only required if the model is stateful).

        :param batch_size: The batch size
        :returns: The shape of a state, as required by the used stateful module
        """
        bf_shape = self.transpose_state_shape(self.get_state_shape(batch_size))
        bf_shape_flat = (batch_size, np.prod(bf_shape[1:]).item())
        return torch.zeros(bf_shape_flat, requires_grad=False).to(device)

    @abc.abstractmethod
    def get_state_shape(self, batch_size: int) -> Tuple[int]:
        """
        Gets the (original) shape of a state (only required if the model is stateful).

        :param batch_size: The batch size
        :returns: The shape of a state, as required by the used stateful module
        """
        ...

    def unflatten(self, flat_batches):
        #return flat_batches, None

        assert self.has_state_input is not None, \
            "You first have to set the input dimensions before you can unflatten the input."

        batch_size = flat_batches.shape[0]

        nb_x_elem_in_sequence = 1 if self.sequence_length is None else self.sequence_length
        nb_single_x_elem = self.nb_input_channels * self.board_width * self.board_height
        nb_all_x_elem = nb_single_x_elem * nb_x_elem_in_sequence

        if self.has_state_input:
            x, state_bf = torch.split(flat_batches, nb_all_x_elem, dim=-1)
        else:
            x = flat_batches
            state_bf = None

        if self.sequence_length is None:
            # no sequence dimension
            x = x.view(batch_size, self.nb_input_channels, self.board_height, self.board_width)
        else:
            # with sequence dimension (for training)
            x = x.view(batch_size, self.sequence_length, self.nb_input_channels, self.board_height,
                       self.board_width)

        if state_bf is not None:
            state_bf = state_bf.view(*self.transpose_state_shape(self.get_state_shape(batch_size)))

        return x, state_bf

    def set_input_options(self, sequence_length: Optional[int], has_state_input: bool):
        self.sequence_length = sequence_length
        self.has_state_input = has_state_input

    def transpose_state(self, state):
        """
        A transpose operation which transforms a batch_first (bf) state to the dims expected by the used stateful
        Module and vice versa.
        """
        # just switch state and batch dim
        return torch.transpose(state, self.state_batch_dim, 0).contiguous()

    def transpose_state_shape(self, state_shape):
        transposed_shape = list(state_shape)
        transposed_shape[self.state_batch_dim] = state_shape[0]
        transposed_shape[0] = state_shape[self.state_batch_dim]

        return transposed_shape

    def flatten(self, x: torch.Tensor, state_bf_flat: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Flattens and concatenates model input and state tensors.

        :param x: The (spacial) input of the model.
        :param state_bf_flat: The current state (batch-dim first and flattened).
        :returns: A flattened concatenation of the given tensors
        """
        batch_size = x.shape[0]
        x_batches_flat = x.view(batch_size, -1)

        if state_bf_flat is None:
            return x_batches_flat

        assert state_bf_flat.shape[0] == batch_size

        # concatenate the batches
        return torch.cat((x_batches_flat, state_bf_flat), dim=1)
