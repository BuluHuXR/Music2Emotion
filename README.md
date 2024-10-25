# State-of-the-art Music Tagging Models
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PyTorch implementation of state-of-the-art music tagging models :notes:

[Demo and Docker image on Replicate](https://replicate.ai/minzwon/sota-music-tagging-models)

## Requirements
```
conda create -n YOUR_ENV_NAME python=3.7
conda activate YOUR_ENV_NAME
pip install -r requirements.txt
```

## Preprocessing
STFT will be done on-the-fly. You only need to read and resample audio files into `.npy` files. 

`cd preprocessing/`

`python -u mtat_read.py run YOUR_DATA_PATH`

## Training

`cd training/`

`python -u main.py --data_path YOUR_DATA_PATH`

Options

```
'--num_workers', type=int, default=0
'--dataset', type=str, default='mtat', choices=['mtat', 'msd', 'jamendo']
'--model_type', type=str, default='fcn',
				choices=['fcn', 'musicnn', 'crnn', 'sample', 'se', 'short', 'short_res', 'attention', 'hcnn']
'--n_epochs', type=int, default=200
'--batch_size', type=int, default=16
'--lr', type=float, default=1e-4
'--use_tensorboard', type=int, default=1
'--model_save_path', type=str, default='./../models'
'--model_load_path', type=str, default='.'
'--data_path', type=str, default='./data'
'--log_step', type=int, default=20
```

## Evaluation
`cd training/`

`python -u eval.py --data_path YOUR_DATA_PATH`

Options

```
'--num_workers', type=int, default=0
'--dataset', type=str, default='mtat', choices=['mtat', 'msd', 'jamendo']
'--model_type', type=str, default='fcn',
                choices=['fcn', 'musicnn', 'crnn', 'sample', 'se', 'short', 'short_res', 'attention', 'hcnn']
'--batch_size', type=int, default=16
'--model_load_path', type=str, default='.'
'--data_path', type=str, default='./data'
```



## Bulu 

### Preprocessing
1. 執行random_splitdata.py生成train.npy、test.npy、val.npy 以及刪除被整理出來的wavfile
```commandline
python random_splitdata.py
```
2. 到preprocessing執行`mtat_read.py`（這樣有音檔在的資料夾就會有npy資料夾）
```commandline
python -u mtat_read.py run ../training/labelstudio
```

### Training
1. 進training資料夾內`main.py`，修改下面的參數即可訓練。
```commandline
python -u main.py
```

### Evaluation
1.執行
```commandline
python eval_bulu.py
```