* Default
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0 --model_class=LineCNNSimple --window_width=28 --window_stride=28
    ```
    {'test_acc': 0.9010781049728394}
    ```

* Changing window_stride
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0 --model_class=LineCNNSimple --window_width=28 --window_stride=20 --limit_output_length
    ```
    {'test_acc': 0.5277031064033508}
    ```

* Changing overlap
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0.25 --max_overlap=0.25 --model_class=LineCNNSimple --window_width=28 --window_stride=20 --limit_output_length
    ```
    {'test_acc': 0.8266875147819519}
    ```

* Variable-length overlap
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0.33 --model_class=LineCNNSimple --window_width=28 --window_stride=20 --limit_output_length

    ```
    {'test_acc': 0.6136093735694885}
    ```

* LineCNN
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0.25 --max_overlap=0.25 --model_class=LineCNN --window_width=28 --window_stride=20 --limit_output_length
    ```
    {'test_acc': 0.5952031016349792}
    ```

* CTC Loss
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0.25 --max_overlap=0.25 --model_class=LineCNN --window_width=28 --window_stride=20 --loss=ctc
    ```
    {'test_acc': 0.903124988079071, 'test_cer': 0.07401753962039948}
    ```

* CTC variable overlap
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0.33 --model_class=LineCNN --window_width=28 --window_stride=18 --loss=ctc
    ```
    {'test_acc': 0.9005781412124634, 'test_cer': 0.0796314924955368}
    ```

* Add LSTM
python training/run_experiment.py --max_epochs=10 --gpus=1 --num_workers=4 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0.33 --model_class=LineCNNLSTM --window_width=28 --window_stride=18 --loss=ctc
    ```
    {'test_acc': 0.9326093792915344, 'test_cer': 0.060667239129543304}
    ```

1. Play around with the hyperparameters of the CNN (window_width, window_stride, conv_dim, fc_dim) and/or the LSTM (lstm_dim, lstm_layers). 
    ```
    --window_width=30 --window_stride=20
    ```
    {'test_acc': 0.9158750176429749, 'test_cer': 0.06678462028503418}
    ```
    --conv_dim=128 --fc_dim=256
    ```
    {'test_acc': 0.941031277179718, 'test_cer': 0.055211391299963}
    ```
    --lstm_dim=1024 --lstm_layers=2 --lstm_dropout=0.5
    ```
    {'test_acc': 0.9623281359672546, 'test_cer': 0.04397038742899895}

2. Better yet, edit LineCNN to use residual connections and other CNN tricks, or just change its architecture in some ways, like you did for Lab 2.
{'test_acc': 0.9527968764305115, 'test_cer': 0.05197448655962944}
    ```python
    class ConvBlock(nn.Module):
    """
    Simple 3x3 conv with padding size 1 (to leave the input size unchanged), followed by a ReLU.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: Param2D = 3,
        stride: Param2D = 1,
        padding: Param2D = 1,
        downsample: bool = False
    ) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size = kernel_size, 
                               stride = stride, padding = padding, bias = False)
        self.bn1 = nn.BatchNorm2d(output_channels)
        
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size = kernel_size, 
                               stride = 1, padding = padding, bias = False)
        self.bn2 = nn.BatchNorm2d(output_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(input_channels, output_channels, kernel_size = 1, 
                             stride = stride,  bias = False)
            bn = nn.BatchNorm2d(output_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
        
        self.downsample = downsample

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
        if self.stride != 1 or self.stride != 2:
            x = self.conv1(x)
        else : 
            i = x
            
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            
            x = self.conv2(x)
            x = self.bn2(x)
            
            if self.downsample is not None:
                i = self.downsample(i)
            x += i

        x = self.relu(x)
        return x

    class LineCNN(nn.Module):
    """
    Model that uses a simple CNN to process an image of a line of characters with a window, outputs a sequence of logits
    """

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.args = vars(args) if args is not None else {}
        self.num_classes = len(data_config["mapping"])
        self.output_length = data_config["output_dims"][0]

        _C, H, _W = data_config["input_dims"]
        conv_dim = self.args.get("conv_dim", CONV_DIM)
        fc_dim = self.args.get("fc_dim", FC_DIM)
        self.WW = self.args.get("window_width", WINDOW_WIDTH)
        self.WS = self.args.get("window_stride", WINDOW_STRIDE)
        self.limit_output_length = self.args.get("limit_output_length", False)

        # Input is (1, H, W)
        self.convs = nn.Sequential(
            ConvBlock(1, conv_dim),
            ConvBlock(conv_dim, conv_dim),
            ConvBlock(conv_dim, conv_dim, stride=2, downsample=True),
            ConvBlock(conv_dim, conv_dim),
            ConvBlock(conv_dim, conv_dim * 2, stride=2, downsample=True),
            ConvBlock(conv_dim * 2, conv_dim * 2),
            ConvBlock(conv_dim * 2, conv_dim * 4, stride=2, downsample=True),
            ConvBlock(conv_dim * 4, conv_dim * 4),
            ConvBlock(
                conv_dim * 4, fc_dim, kernel_size=(H // 8, self.WW // 8), stride=(H // 8, self.WS // 8), padding=0
            ),
        )
        self.fc1 = nn.Linear(fc_dim, fc_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(fc_dim, self.num_classes)

        self._init_weights()

    ```

3. Feel free to edit LineCNNLSTM as well, get crazy with LSTM stuff!

    `1번에서 이미 함`


4. In your own words, explain how the CharacterErrorRate metric and the greedy_decode method work.

    `CharacterErrorRate` : 레벤슈타인 거리를 이용해서 각 문자에 대한 에러율 표시

    `greedy_decode` : 각각 확률 중에서 가장 큰 것을 다음 step의 입력으로 사용
