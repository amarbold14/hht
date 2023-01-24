import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class OneLayerLSTM(nn.Module):
    def __init__(self, data_loader, train_config,
                 num_layers=1,
                 batch_first=True,
                 bidirectional=False):
        """
        :param num_layers: number of LSTMs stacked on top of each other (default 1)
        :param batch_first: (batch, seq, feature)
        """
        super(OneLayerLSTM, self).__init__()

        # Parse dimensions
        input_dim = data_loader.input_dimension
        hidden_size = train_config.LSTM_hidden_size
        sequence_len = data_loader.sequence_len
        output_dim = data_loader.output_dimension
        self.input_size = (sequence_len, input_dim)

        self.H_in = input_dim
        self.L = sequence_len
        self.H_hidden = hidden_size
        self.H_out = output_dim

        self.lstm = nn.LSTM(input_size=self.H_in, hidden_size=self.H_hidden, num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=batch_first)
        self.linear = nn.Linear(in_features=self.H_hidden * self.L, out_features=self.H_out)
        self.output_layer = nn.Softmax(dim=1)  # along axis

    def forward(self, x):
        x_lstm, _ = self.lstm(x)  # (N, L, H_hidden)
        x_lstm_activation = F.relu(x_lstm)  # (N, L, H_hidden)
        x_lstm_activation_flatten = torch.reshape(x_lstm_activation, (-1, self.L * self.H_hidden))  # (N, LxH_hidden)
        x_linear = self.linear(x_lstm_activation_flatten)
        out = self.output_layer(x_linear)  # (N, H_out)
        return out


class BinaryOneLayerLSTM(OneLayerLSTM):
    def __init__(self, data_loader, train_config,
                 num_layers=1,
                 batch_first=True,
                 bidirectional=False):
        super(BinaryOneLayerLSTM, self).__init__(data_loader=data_loader,
                                                 train_config=train_config,
                                                 num_layers=num_layers,
                                                 batch_first=batch_first,
                                                 bidirectional=bidirectional)
        self.output_layer = nn.Sigmoid()  # overwrite OneLayerLSTM


class TwoLayerLSTM(nn.Module):
    """
    NEW: hidden_size, example
        hidden_size = [30, 30]
    """
    def __init__(self, data_loader, train_config,
                 num_layers=1,
                 batch_first=True,
                 bidirectional=False):
        super(TwoLayerLSTM, self).__init__()

        # Parse dimensions
        input_dim = data_loader.input_dimension
        hidden_size = train_config.LSTM_hidden_size
        sequence_len = data_loader.sequence_len
        output_dim = data_loader.output_dimension
        self.input_size = (sequence_len, input_dim)

        assert len(hidden_size) == 2, f'hidden_size should be a list, or consider using OneLayerLSTM!'

        self.H_in = input_dim
        self.L = sequence_len
        self.H_out = output_dim

        self.H_hidden_1 = hidden_size[0]
        self.H_hidden_2 = hidden_size[1]

        self.lstm = nn.LSTM(input_size=self.H_in, hidden_size=self.H_hidden_1, num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=batch_first)
        self.linear_1 = nn.Linear(in_features=self.H_hidden_1 * self.L, out_features=self.H_hidden_2)
        self.linear_2 = nn.Linear(in_features=self.H_hidden_2, out_features=self.H_out)
        self.output_layer = nn.Softmax(dim=1)  # along axis

    def forward(self, x):
        x_lstm, _ = self.lstm(x)  # (N, L, H_hidden)
        x_lstm_activation = F.relu(x_lstm)  # (N, L, H_hidden)
        x_lstm_activation_flatten = torch.reshape(x_lstm_activation,
                                                  (-1, self.L * self.H_hidden_1))  # (N, LxH_hidden_1)
        x_linear_1 = self.linear_1(x_lstm_activation_flatten)
        x_linear_2 = self.linear_2(x_linear_1)
        out = self.output_layer(x_linear_2)  # (N, H_out)
        return out


class BinaryTwoLayerLSTM(TwoLayerLSTM):
    def __init__(self, train_config, data_loader,
                 num_layers=1,
                 batch_first=True,
                 bidirectional=False):
        super(BinaryTwoLayerLSTM, self).__init__(train_config=train_config,
                                                 data_loader=data_loader,
                                                 num_layers=num_layers,
                                                 batch_first=batch_first,
                                                 bidirectional=bidirectional)
        self.output_layer = nn.Sigmoid()  # overwrite TwoLayerLSTM
