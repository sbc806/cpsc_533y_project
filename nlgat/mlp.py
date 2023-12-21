class Mlps(nn.Module):
    """Mlps implemented as (dxd) convolution."""

    def __init__(self, inc, outc_list=[128], kernel_size=1, stride=1, padding=0, last_bn_norm=True):
        """Initialize network with hyperparameters.

        Args:
            inc (int): number of channels in the input.
            outc_list (List[]): list of dimensions of hidden layers.
            last_bn_norm (boolean): determine if bn and norm layer is added into the output layer.
        """
        assert len(outc_list) > 0
        super(Mlps, self).__init__()

        self.layers = nn.Sequential()

        # We compose MLPs according to the list of out_channel (`outc_list`).
        # Additionally, we use the flag `last_bn_norm` to
        # determine if we want to add norm and activation layers
        # at last layer.
        for i, outc in enumerate(outc_list):
            self.layers.add_module(f"Linear-{i}", nn.Conv2d(inc, outc, kernel_size, stride=stride, padding=padding))
            if i + 1 < len(outc_list) or last_bn_norm:
                self.layers.add_module(f"BN-{i}", nn.BatchNorm2d(outc))
                self.layers.add_module(f"ReLU-{i}", nn.ReLU(inplace=True))
            inc = outc
        if config.gsn_init == "he_1":
            self.apply(self._init_weights_he_1)

    def _init_weights_he_1(self, module):
        if isinstance(module, torch.nn.Conv2d):
            print('initializing He weights in {}'.format(module.__class__.__name__))
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.01)

    def forward(self, x, format="BCNM"):
        """Forward pass.

        Args:
            x (torch.tensor): input tensor.
            format (str): format of point tensor.
                Options include 'BCNM', 'BNC', 'BCN'
        """
        assert format in ["BNC", "BCNM", "BCN"]

        # Re-formate tensor into "BCNM".
        if format == "BNC":
            x = x.transpose(2, 1).unsqueeze(-1)
        elif format == "BCN":
            x = x.unsqueeze(-1)
        # print("x",x.shape)
        # We use the tensor of the "BCNM" format.
        x = self.layers(x)

        # Re-formate tensor back input format.
        if format == "BNC":
            x = x.squeeze(-1).transpose(2, 1)
        elif format == "BCN":
            x = x.squeeze(-1)

        return x