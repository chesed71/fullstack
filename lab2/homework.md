default : {'test_acc': 0.8036044836044312}

1. Edit the CNN and ConvBlock architecture in text_recognizers/models/cnn.py in some ways.
In particular, edit the ConvBlock module to be more like a ResNet block, as shown in the following image:
![alt resblock](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs/raw/main/lab2/resblock.png)
    ```python
    class ConvBlock(nn.Module):
        """
        Simple 3x3 conv with padding size 1 (to leave the input size unchanged), followed by a ReLU.
        """

        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x
                of dimensions (B, C, H, W)

            Returns
            -------
            torch.Tensor
                of dimensions (B, C, H, W)
            """
            c = self.conv1(x)
            r = self.relu(c)
            c = self.conv2(r)
            r = c + x
            r = self.relu(r)
            return r

    ```
    {'test_acc': 0.8170334100723267}

2. Try adding more of the ResNet secret sauce, such as BatchNorm. 


    ```python
    class ConvBlock(nn.Module):
    """
        Simple 3x3 conv with padding size 1 (to leave the input size unchanged), followed by a ReLU.
        """

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, kernel_size=3, stride=1, padding=1
        )
        self.relu = nn.ReLU()
        self.norm_layer1 = nn.BatchNorm2d(output_channels)
        self.norm_layer2 = nn.BatchNorm2d(output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Parameters
            ----------
            x
                of dimensions (B, C, H, W)

            Returns
            -------
            torch.Tensor
                of dimensions (B, C, H, W)
            """
        c = self.conv1(x)
        l = self.norm_layer1(c)
        r = self.relu(c)
        c = self.conv2(r)
        l = self.norm_layer1(c)
        r = l + x
        r = self.relu(r)
        return r
    ```
    {'test_acc': 0.8118841052055359}

3. Remove MaxPool2D, perhaps using a strided convolution instead.

    ```python
    class ConvBlock(nn.Module):
        """
        Simple 3x3 conv with padding size 1 (to leave the input size unchanged), followed by a ReLU.
        """

        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__()
            self.conv = nn.Conv2d(
                input_channels, output_channels, kernel_size=3, stride=2, padding=1
            )
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x
                of dimensions (B, C, H, W)

            Returns
            -------
            torch.Tensor
                of dimensions (B, C, H, W)
            """
            c = self.conv(x)
            r = self.relu(c)
            return r
    ```

    {'test_acc': 0.7943617105484009}

4. Add some command-line arguments to make trying things a quicker process.

    ```
    precision 16 : 172.61s 
    precision 32 : 164.37s 
    gpu=2 accelerator=ddp : 96.42s

    ```

5. A good argument to add would be for the number of ConvBlocks to run the input through.
--conv_num=4
    ```
    class CNN(nn.Module):
        """Simple CNN for recognizing characters in a square image."""

        def __init__(
            self, data_config: Dict[str, Any], args: argparse.Namespace = None
        ) -> None:
            super().__init__()
            self.args = vars(args) if args is not None else {}

            input_dims = data_config["input_dims"]
            num_classes = len(data_config["mapping"])

            conv_dim = self.args.get("conv_dim", CONV_DIM)
            fc_dim = self.args.get("fc_dim", FC_DIM)

            conv_num = self.args.get("conv_num", 2)

            conv_list = []
            for i in range(conv_num):
                if i == 0:
                    conv_list.append(ConvBlock(input_dims[0], conv_dim))
                else:
                    conv_list.append(ConvBlock(conv_dim, conv_dim))

            self.convs = nn.ModuleList(conv_list)

            self.dropout = nn.Dropout(0.25)
            self.max_pool = nn.MaxPool2d(2)

            # Because our 3x3 convs have padding size 1, they leave the input size unchanged.
            # The 2x2 max-pool divides the input size by 2. Flattening squares it.
            conv_output_size = IMAGE_SIZE // 2
            fc_input_dim = int(conv_output_size * conv_output_size * conv_dim)
            self.fc1 = nn.Linear(fc_input_dim, fc_dim)
            self.fc2 = nn.Linear(fc_dim, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
            x
                (B, C, H, W) tensor, where H and W must equal IMAGE_SIZE

            Returns
            -------
            torch.Tensor
                (B, C) tensor
            """
            _B, _C, H, W = x.shape
            assert H == W == IMAGE_SIZE
            for conv_block in self.convs:
                x = conv_block(x)
            
            x = self.max_pool(x)
            x = self.dropout(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return x
    ```