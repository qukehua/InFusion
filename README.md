## Visualization Results 
![Visualization Results](images/pred_act1.gif)
![Visualization Results](images/pred_act2.gif)
![Visualization Results](images/pred_act5.gif)
## 🛠 Setup

### 1. Python/Conda Environment

The following code is tested on Linux-64 in a cluster environment as well as on Windows 11. If you are using Linux-64 in a cluster environment, please use `source activate transfusion` instead of `conda activate transfusion`.

```
conda create -n infusion python=3.8
conda activate infusion
python -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r requirement.txt
```

### 2. Datasets

```
data
├── data_3d_amass.npz
├── data_3d_amass_test.npz
├── data_3d_h36m.npz
├── data_3d_h36m_test.npz
├── data_3d_humaneva15.npz
├── data_3d_humaneva15_test.npz
├── data_multi_modal
│   ├── data_candi_t_his25_t_pred100_skiprate20.npz
│   └── t_his25_1_thre0.500_t_pred100_thre0.100_filtered_dlow.npz
└── humaneva_multi_modal
    ├── data_candi_t_his15_t_pred60_skiprate15.npz
    └── t_his15_1_thre0.500_t_pred60_thre0.010_index_filterd.npz
```


## ⏳ To Training

Training on HARPER:

python main.py --cfg harper3d_30hz --mode train --exp_name harper3d_30hz

python main.py --cfg harper3d_120hz --mode train --exp_name harper3d_120hz

Training on CHICO:

python main.py --cfg chico --mode train --exp_name chico

Training on CoMad:

python main.py --cfg comad --mode train --exp_name comad


### 🔎 To Evaluation

Evaluate on HARPER:

python main.py --cfg harper3d_30hz --mode eval --ckpt ./results/harper3d_30hz/models/best_ema.pt

python main.py --cfg harper3d_120hz --mode eval --ckpt ./results/harper3d_120hz/models/best_ema.pt

Evaluate on CHICO:

python main.py --cfg chico --mode eval --ckpt ./results/chico/models/best_ema.pt

Evaluate on CoMad:

python main.py --cfg comad --mode eval --ckpt ./results/comad/models/best_ema.pt

## 🎥 To Visualization
Run the following scripts for visualization purpose:

python main.py --cfg harper3d_30hz --mode pred --vis_row 3 --vis_col 10 --ckpt ./results/harper3d_30hz/models/best_ema.pt

python main.py --cfg harper3d_120hz --mode pred --vis_row 3 --vis_col 10 --ckpt ./results/harper3d_120hz/models/best_ema.pt






## 🌹 Acknowledgment
Project structure is borrowed from [TransFusion](https://github.com/sibotian96/TransFusion). We would like to thank the authors for making their code publicly available.


