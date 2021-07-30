import torch.nn as nn


class BaseBlock(nn.Module):
    def __init__(self, c_in, c_out, spatial_downsample):
        super(BaseBlock, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.spatial_downsample = spatial_downsample
        self.relu = nn.ReLU()

        if self.spatial_downsample:
            self.base_block = nn.Sequential(
                self.conv_block(
                    c_in=self.c_in, c_out=self.c_out, kernel_size=3, stride=1, padding=1
                ),
                nn.ReLU(),
                self.conv_block(
                    c_in=self.c_out,
                    c_out=self.c_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        else:
            self.base_block = nn.Sequential(
                self.conv_block(
                    c_in=self.c_in, c_out=self.c_out, kernel_size=3, stride=1, padding=1
                ),
                nn.ReLU(),
                self.conv_block(
                    c_in=self.c_out,
                    c_out=self.c_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )

        if self.c_in != self.c_out:
            self.pointwise_conv = nn.Conv2d(
                in_channels=self.c_in,
                out_channels=self.c_out,
                kernel_size=1,
                stride=2,
                padding=0,
            )

    def forward(self, x):
        identity_layer = x
        x = self.base_block(x)

        if self.c_in == self.c_out:
            x += identity_layer
        else:
            identity_layer = self.pointwise_conv(identity_layer)
            x += identity_layer

        x = self.relu(x)

        return x

    def conv_block(self, c_in, c_out, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, bias=False, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
        )

        return seq_block


class BaseLayer(nn.Module):
    def __init__(self, c_in, c_out):
        super(BaseLayer, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.c_in,
                out_channels=self.c_out,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(self.c_out),
            nn.ReLU(),
        )

        self.res_block = BaseBlock(
            c_in=self.c_out, c_out=self.c_out, spatial_downsample=False,
        )

    def forward(self, x):
        x_id = self.conv(x)
        x = self.res_block(x_id)
        x = x + x_id

        return x


class ResNet(nn.Module):
    def __init__(self, num_input_channels, num_classes):
        super(ResNet, self).__init__()

        self.num_input_channels = num_input_channels
        self.num_classes = num_classes

        self.layer_0 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.num_input_channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
        )
        self.layer_1 = BaseLayer(c_in=64, c_out=128)

        self.layer_2 = self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1,
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.layer_3 = BaseLayer(c_in=256, c_out=512)

        self.maxpool = nn.MaxPool2d(4, 4)
        self.fc = nn.Conv2d(
            in_channels=512, out_channels=self.num_classes, kernel_size=1, stride=1
        )

    def forward(self, x):
        x = self.layer_0(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.maxpool(x)
        x = self.fc(x)

        return x.squeeze()


if __name__ == "__main__":
    from torchinfo import summary

    print("Model Summary: \n")
    model = ResNet(num_input_channels=3, num_classes=10)
    summary(model, input_size=(2, 3, 32, 32))
