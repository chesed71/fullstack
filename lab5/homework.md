```
python training/run_experiment.py --wandb --max_epochs=10 --gpus='0,' --num_workers=4 --data_class=EMNISTLines2 --model_class=LineCNNTransformer --loss=transformer

```
default : {'test_acc': 0.4105468690395355, 'test_cer': 0.5889826416969299}

1. Try to find a settings of hyperparameters for LineCNNTransformer (don't forget -- it includes LineCNN hyperparams) that trains fastest while reaching low CER

2. Perhaps do that by running a sweep!

test_cer : 0.03288
batch size : 128, conv_dim : 64, fc_dim : 512, lr: 0.0003, max_length : 43, max_overlap : 0.5, min_overlap : 0.2, precision : 16, 
tf_dim : 256, tf_dropout : 0.4, tf_fc_dim : 1024, tf_layer : 4, tf_nhead : 4, 
window_stride : 8, window_width : 16

3. Try some experiments with LineCNNLSTM if you want
