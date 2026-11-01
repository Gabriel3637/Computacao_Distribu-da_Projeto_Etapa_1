"""Arquitetura extraída da célula 26 do UNet.ipynb fornecido."""
import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bloco = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.bloco(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes=2,
                 channels=(64, 128, 256, 512, 1024)):
        super().__init__()
        if len(channels) != 5 or any(c <= 0 for c in channels):
            raise ValueError('Informe cinco quantidades positivas de canais.')
        self.in_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder = nn.ModuleList()
        anterior = in_channels
        for canal in channels[:-1]:
            self.encoder.append(DoubleConv(anterior, canal))
            anterior = canal
        self.bottleneck = DoubleConv(channels[-2], channels[-1])

        self.upsampling = nn.ModuleList()
        self.decoder = nn.ModuleList()
        anterior = channels[-1]
        for canal in reversed(channels[:-1]):
            self.upsampling.append(nn.ConvTranspose2d(
                anterior, canal, kernel_size=2, stride=2,
            ))
            self.decoder.append(DoubleConv(2 * canal, canal))
            anterior = canal
        self.classificador = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels or min(x.shape[-2:]) < 32:
            raise ValueError(
                'Entrada esperada: [lote, bandas, altura>=32, largura>=32].'
            )
        skips = []
        for bloco in self.encoder:
            x = bloco(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, bloco, skip in zip(self.upsampling, self.decoder, reversed(skips)):
            x = up(x)
            # Para dimensões ímpares, alinha a resolução antes de concatenar.
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = bloco(torch.cat([skip, x], dim=1))
        return self.classificador(x)