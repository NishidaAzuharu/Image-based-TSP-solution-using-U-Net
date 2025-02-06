
import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, first_layer_filter_count):
        super(UNet, self).__init__()

        # パラメータ定義
        self.CONCATENATE_AXIS = 1  # チャンネルファースト (B, C, H, W)
        self.CONV_FILTER_SIZE = 4
        self.CONV_STRIDE = 2
        self.CONV_PADDING = 1
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2

        # エンコーダーの作成
        self.enc1 = self._add_encoding_layer(1, first_layer_filter_count)
        self.enc2 = self._add_encoding_layer(first_layer_filter_count, first_layer_filter_count * 2)
        self.enc3 = self._add_encoding_layer(first_layer_filter_count * 2, first_layer_filter_count * 4)
        self.enc4 = self._add_encoding_layer(first_layer_filter_count * 4, first_layer_filter_count * 8)
        self.enc5 = self._add_encoding_layer(first_layer_filter_count * 8, first_layer_filter_count * 16)
        self.enc6 = self._add_encoding_layer(first_layer_filter_count * 16, first_layer_filter_count * 16)
        self.enc7 = self._add_encoding_layer(first_layer_filter_count * 16, first_layer_filter_count * 16)

        # デコーダーの作成
        self.dec1 = self._add_decoding_layer(first_layer_filter_count * 16, first_layer_filter_count * 16, True)
        self.dec2 = self._add_decoding_layer(first_layer_filter_count * 16 * 2, first_layer_filter_count * 16, True)
        self.dec3 = self._add_decoding_layer(first_layer_filter_count * 16 * 2, first_layer_filter_count * 8, True)
        self.dec4 = self._add_decoding_layer(first_layer_filter_count * 8 * 2, first_layer_filter_count * 4, False)
        self.dec5 = self._add_decoding_layer(first_layer_filter_count * 4 * 2, first_layer_filter_count * 2, False)
        self.dec6 = self._add_decoding_layer(first_layer_filter_count * 2 * 2, first_layer_filter_count, False)
        self.dec7 = self._add_decoding_layer(first_layer_filter_count * 2, 1, False)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)

        dec1 = self.dec1(enc7)
        dec2 = self.dec2(torch.cat((dec1, enc6), dim=self.CONCATENATE_AXIS))
        dec3 = self.dec3(torch.cat((dec2, enc5), dim=self.CONCATENATE_AXIS))
        dec4 = self.dec4(torch.cat((dec3, enc4), dim=self.CONCATENATE_AXIS))
        dec5 = self.dec5(torch.cat((dec4, enc3), dim=self.CONCATENATE_AXIS))
        dec6 = self.dec6(torch.cat((dec5, enc2), dim=self.CONCATENATE_AXIS))
        dec7 = self.dec7(torch.cat((dec6, enc1), dim=self.CONCATENATE_AXIS))

        return dec7

    def _add_encoding_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, self.CONV_FILTER_SIZE, self.CONV_STRIDE, self.CONV_PADDING),
            nn.LeakyReLU(0.2, inplace=False)
        )

    def _add_decoding_layer(self, in_channels, out_channels, add_drop_layer=False):
        layers = [
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(in_channels, out_channels, self.DECONV_FILTER_SIZE, self.DECONV_STRIDE),
            nn.BatchNorm2d(out_channels)
        ]
        if add_drop_layer:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)