import torch
import torch.nn as nn
import torch.nn.init as initializer
from torch.cuda.amp import autocast
import copy

import src.config as config


class Embedding(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, bidirectional=False, num_layers=1):
        super(Embedding, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                           bidirectional=bidirectional)
        self.num_class = num_class
        if bidirectional:
            hidden_size = hidden_size * 2

        self.embedding = nn.Sequential(nn.Dropout(0.5), nn.Linear(hidden_size, num_class), )

        initializer.xavier_uniform_(self.rnn.weight_hh_l0)
        initializer.xavier_uniform_(self.rnn.weight_ih_l0)
        initializer.xavier_uniform_(self.embedding[1].weight)

    @autocast()
    def forward(self, x):
        recurrent, _ = self.rnn(x)

        batch_size, sequence_size, hidden_size = recurrent.size()
        t_rec = recurrent.reshape(sequence_size * batch_size, hidden_size)
        embedding = self.embedding(t_rec)
        result = embedding.reshape(batch_size, sequence_size, self.num_class).log_softmax(2)
        return result


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=1):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout2d(0.25))

    @autocast()
    def forward(self, x):
        return self.block(x)


class CRNN(nn.Module):
    def __init__(self, image_h, num_class, num_layers, is_lstm_bidirectional):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(

            ConvBlock(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2, 2),

            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2, 2),

            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0), )

        self.squeeze = nn.Sequential(nn.Linear(512 * 14, 256), nn.ReLU(inplace=True), nn.Dropout(0.5))

        initializer.xavier_uniform_(self.squeeze[0].weight)

        self.rnn = Embedding(256, 256, num_class, bidirectional=is_lstm_bidirectional, num_layers=num_layers)

    @autocast()
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 512 * 14, 38).permute(0, 2, 1).contiguous()
        x = self.squeeze(x)
        return self.rnn(x)


class CRNN_2(nn.Module):
    def __init__(self, image_h, num_class, num_layers, is_lstm_bidirectional, num_regions):
        super(CRNN_2, self).__init__()

        self.cnn = nn.Sequential(

            ConvBlock(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2, 2),

            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2, 2),

            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0), )

        self.squeeze = nn.Sequential(nn.Linear(512 * 14, 256), nn.ReLU(inplace=True), nn.Dropout(0.5))

        self.classificator = nn.Sequential(nn.Linear(9728, num_regions))

        initializer.xavier_uniform_(self.squeeze[0].weight)

        self.rnn = Embedding(256, 256, num_class, bidirectional=is_lstm_bidirectional, num_layers=num_layers)

    @autocast()
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 512 * 14, 38).permute(0, 2, 1).contiguous()

        x = self.squeeze(x)
        cls = x.view(x.size(0), -1)
        return self.rnn(x), self.classificator(cls)


def load_model_CRNN(model_path):
    model = CRNN(image_h=config.img_h, num_class=config.num_class_crossed, num_layers=config.model_lstm_layers,
                   is_lstm_bidirectional=config.model_lsrm_is_bidirectional).cuda()
    state = torch.load(model_path)
    state_dict = state['state_dict']
    new_state_dict = copy.deepcopy(state_dict)

    for key in state_dict:
        new_state_dict[key.replace('module.', '')] = new_state_dict.pop(key)

    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    return model


def load_model_CRNN2(model_path):
    model = CRNN_2(image_h=config.img_h, num_class=config.num_class, num_layers=config.model_lstm_layers,
                   is_lstm_bidirectional=config.model_lsrm_is_bidirectional, num_regions=config.num_regions).cuda()
    state = torch.load(model_path)
    state_dict = state['state_dict']
    new_state_dict = copy.deepcopy(state_dict)

    for key in state_dict:
        new_state_dict[key.replace('module.', '')] = new_state_dict.pop(key)

    model.load_state_dict(new_state_dict)
    model.cuda()
    model.eval()

    return model
