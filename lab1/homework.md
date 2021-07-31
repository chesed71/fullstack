1. Try training/run_experiment.py with different MLP hyper-parameters (e.g. --fc1=128 --fc2=64).

+ `python training/run_experiment.py --model_class=MLP --data_class=MNIST --max_epochs=5 --gpus=0`

     {'test_acc': 0.9776999950408936}

+ `python training/run_experiment.py --model_class=MLP --data_class=MNIST --max_epochs=5 --gpus=0 --fc1=128 --fc2=64`

     {'test_acc': 0.9642000198364258}

2. Try editing the MLP architecture in text_recognizers/models/mlp.py

    ```
        fc1_dim = self.args.get("fc1", FC1_DIM)
        fc2_dim = self.args.get("fc2", FC2_DIM)

        self.dropout = nn.Dropout(0.2)
        self.bn1 = torch.nn.BatchNorm1d(fc1_dim)
        self.bn2 = torch.nn.BatchNorm1d(fc2_dim)

        self.fc1 = nn.Linear(input_dim, fc1_dim)

        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    ```
    {'test_acc': 0.9782999753952026}