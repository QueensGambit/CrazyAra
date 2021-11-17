import torch
import torch.nn as nn

from nn.PommerModel import PommerModel
from nn.a0_resnet import ResidualBlock
from nn.builder_util import _Stem, get_act, TimeDistributed


class SimpleLSTM(PommerModel):
    """
    A simple test architecture using lstm.
    """
    def __init__(self, channels=64, nb_input_channels=18, num_res_blocks=2, act_type='relu', embedding_size=256,
                 hidden_size=128, value_hidden_size=128, policy_hidden_size=128, n_labels=6, bn_mom=0.9,
                 board_height=11, board_width=11, lstm_num_layers=1):

        super().__init__(is_stateful=True)

        self.nb_flatten = board_height * board_width * channels

        self.body = TimeDistributed(nn.Sequential(
            _Stem(channels=channels, bn_mom=bn_mom, act_type=act_type, nb_input_channels=nb_input_channels),
            *[ResidualBlock(channels, bn_mom, act_type) for _ in range(0, num_res_blocks)],
            nn.Flatten(),
            nn.Linear(self.nb_flatten, embedding_size),
            # Batch norm?
            # Dropout?
            get_act(act_type),
        ), input_dims=4)

        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True, num_layers=lstm_num_layers)

        self.value_head = TimeDistributed(nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=value_hidden_size),
            get_act(act_type),
            nn.Linear(in_features=value_hidden_size, out_features=1),
            get_act("tanh")
        ), input_dims=4)

        self.policy_head = TimeDistributed(nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=policy_hidden_size),
            get_act(act_type),
            nn.Linear(in_features=policy_hidden_size, out_features=n_labels)
        ), input_dims=4)

    def get_init_state(self, batch_size: int, device):
        # (h0, c0) as single tensor
        return torch.zeros(self.get_state_shape(batch_size), requires_grad=False).to(device)

    def get_state_shape(self, batch_size: int):
        return 2, self.lstm.num_layers, batch_size, self.lstm.hidden_size

    def forward(self, x: torch.Tensor, hidden_state: torch.Tensor = None):
        # batch without seq: 4, batch & seq: 5
        single_input = len(x.shape) == 4
        if single_input:
            # unsqueeze single input (add sequence dimension) => process a batch of 1-element sequences
            x = x.unsqueeze(1)

        embedding = self.body(x)

        if hidden_state is None:
            output, next_hidden_state_pair = self.lstm(embedding)
        else:
            output, next_hidden_state_pair = self.lstm(embedding, (hidden_state[0], hidden_state[1]))

        next_h = next_hidden_state_pair[0].unsqueeze(0)
        next_c = next_hidden_state_pair[1].unsqueeze(0)
        next_hidden_state = torch.cat((next_h, next_c), dim=0)

        value = self.value_head(output)
        policy = self.policy_head(output)

        if single_input:
            value = value.squeeze(1)
            policy = policy.squeeze(1)

        return value, policy, next_hidden_state
