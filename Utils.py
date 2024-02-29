import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from captcha.image import ImageCaptcha
from tqdm import tqdm
import string
import random
import numpy as np
from collections import OrderedDict


class CaptchaDataset(Dataset):
    def __init__(self, characters, length, width, height, input_length, label_length):
        super(CaptchaDataset, self).__init__()
        self.characters = characters
        self.length = length
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        random_str = "".join(
            [random.choice(self.characters[1:]) for j in range(self.label_length)]
        )
        image = to_tensor(self.generator.generate_image(random_str))
        target = torch.tensor(
            [self.characters.find(x) for x in random_str], dtype=torch.long
        )
        input_length = torch.full(
            size=(1,), fill_value=self.input_length, dtype=torch.long
        )
        target_length = torch.full(
            size=(1,), fill_value=self.label_length, dtype=torch.long
        )
        return image, target, input_length, target_length


class Model(nn.Module):
    def __init__(self, n_classes, input_shape=(3, 64, 128)):
        super(Model, self).__init__()
        self.input_shape = input_shape
        channels = [32, 64, 128, 256, 256]
        layers = [2, 2, 2, 2, 2]
        kernels = [3, 3, 3, 3, 3]
        pools = [2, 2, 2, 2, (2, 1)]
        modules = OrderedDict()

        def cba(name, in_channels, out_channels, kernel_size):
            modules[f"conv{name}"] = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=(1, 1) if kernel_size == 3 else 0,
            )
            modules[f"bn{name}"] = nn.BatchNorm2d(out_channels)
            modules[f"relu{name}"] = nn.ReLU(inplace=True)

        last_channel = 3
        for block, (n_channel, n_layer, n_kernel, k_pool) in enumerate(
            zip(channels, layers, kernels, pools)
        ):
            for layer in range(1, n_layer + 1):
                cba(f"{block+1}{layer}", last_channel, n_channel, n_kernel)
                last_channel = n_channel
            modules[f"pool{block+1}"] = nn.MaxPool2d(k_pool)
        modules[f"dropout"] = nn.Dropout(0.25, inplace=True)
        self.cnn = nn.Sequential(modules)
        self.lstm = nn.LSTM(
            input_size=self.infer_features(),
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
        )
        self.fc = nn.Linear(in_features=256, out_features=n_classes)

    def infer_features(self):
        x = torch.zeros((1,) + self.input_shape)
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x.shape[1]

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x


def decode(sequence, characters):
    a = "".join([characters[x] for x in sequence])
    s = "".join(
        [x for j, x in enumerate(a[:-1]) if x != characters[0] and x != a[j + 1]]
    )
    if len(s) == 0:
        return ""
    if a[-1] != characters[0] and s[-1] != a[-1]:
        s += a[-1]
    return s


def decode_target(sequence, characters):
    return "".join([characters[x] for x in sequence]).replace(" ", "")


def calc_acc(target, output, characters):
    output_argmax = output.detach().permute(1, 0, 2).argmax(dim=-1)
    target = target.cpu().numpy()
    output_argmax = output_argmax.cpu().numpy()
    a = np.array(
        [
            decode_target(true, characters) == decode(pred, characters)
            for true, pred in zip(target, output_argmax)
        ]
    )
    return a.mean()


def train(model, optimizer, epoch, dataloader, characters):
    model.train()
    loss_mean = 0
    acc_mean = 0
    with tqdm(dataloader) as pbar:
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(
            pbar
        ):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)

            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            acc = calc_acc(target, output, characters)

            if batch_index == 0:
                loss_mean = loss
                acc_mean = acc
            loss_mean = 0.1 * loss + 0.9 * loss_mean
            acc_mean = 0.1 * acc + 0.9 * acc_mean
            pbar.set_description(
                f"Epoch: {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} "
            )


def valid(model, optimizer, epoch, dataloader, characters):
    model.eval()
    with tqdm(dataloader) as pbar, torch.no_grad():
        loss_sum = 0
        acc_sum = 0
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(
            pbar
        ):
            data, target = data.cuda(), target.cuda()

            output = model(data)
            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = F.ctc_loss(
                output_log_softmax, target, input_lengths, target_lengths
            )
            loss = loss.item()
            acc = calc_acc(target, output, characters)
            loss_sum += loss
            acc_sum += acc
            loss_mean = loss_sum / (batch_index + 1)
            acc_mean = acc_sum / (batch_index + 1)
            pbar.set_description(
                f"Test : {epoch} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} "
            )
