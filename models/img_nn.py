import sys
sys.path.append('..')
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return nn.ReLU()(x)

class Img_2D_CNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        embedding_dim = args.cb_dim

        self.img_2d_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            ResidualStack(in_channels=256, num_hiddens=256, num_residual_layers=6, num_residual_hiddens=256),
            nn.Conv2d(in_channels=256, out_channels=embedding_dim, kernel_size=1, stride=1))

        self.img_2d_decoder = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim, out_channels=256, kernel_size=3, stride=1, padding=1),
            ResidualStack(in_channels=256, num_hiddens=256, num_residual_layers=6, num_residual_hiddens=256),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1))

    def forward(self, x, z):
        if z is None:
            return self.img_2d_encoder(x)
        else:
            assert self.img_2d_decoder(z).size(-1) == self.args.img_size
            return self.img_2d_decoder(z)