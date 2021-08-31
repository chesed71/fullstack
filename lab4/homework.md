```
python training/run_experiment.py --max_epochs=40 --gpus=1 --num_workers=16 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0.33 --model_class=LineCNNTransformer --window_width=20 --window_stride=12 --loss=transformer
```
default : {'test_acc': 0.4105468690395355, 'test_cer': 0.5889826416969299}

1. Standard stuff: try training with some different hyperparameters, explain what you tried.
```
이미 line_cnn_transormer.py에 구현되어 있음


python training/run_experiment.py --max_epochs=40 --gpus=1 --num_workers=16 --data_class=EMNISTLines --min_overlap=0 --max_overlap=0.33 --model_class=LineCNNTransformer --window_width=20 --window_stride=12 --loss=transformer --tf_dim=512 --tf_fc_dim=2048 --tf_dropout=0.5 --tf_layers=6 --tf_nhead=8
```
{'test_acc': 0.021968750283122063, 'test_cer': 0.9756916165351868}



2. There is also an opportunity to speed up the predict method that you could try.
```
batch로 되어 있는 데 과연 가능한가?  
```