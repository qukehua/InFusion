
## 🛠 Setup

### 1. Python/Conda Environment

The following code is tested on Linux-64 in a cluster environment as well as on Windows 11. If you are using Linux-64 in a cluster environment, please use `source activate transfusion` instead of `conda activate transfusion`.

```
mkdir ./checkpoints
mkdir ./data
mkdir ./inference
mkdir ./results
conda create -n transfusion python=3.8
conda activate transfusion
python -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -r requirement.txt
```

### 2. Datasets


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




### 🔎 Evaluation
```
Evaluate on Human3.6M:


python main.py --cfg harper3d_30hz --mode eval --ckpt ./results/harper3d_30hz/models/best_ema.pt


**Note**: We change the random seed to 6 instead of 0 for AMASS dataset to fairly compared with [BeLFusion](https://github.com/BarqueroGerman/BeLFusion). GPU is required for evaluation.

## ⏳ Training
For training TransFusion from scratch for all three datasets, run the following scripts:

python main.py --cfg h36m --mode train

python main.py --cfg humaneva --mode train

python main.py --cfg amass --mode train --multimodal_threshold 0.4 --seed 6 --milestone [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800]

## 🎥 Visualization
Run the following scripts for visualization purpose:

python main.py --cfg h36m --mode pred --vis_row 3 --vis_col 10 --ckpt ./checkpoints/h36m_ckpt.pt

python main.py --cfg humaneva --mode pred --vis_row 3 --vis_col 10 --ckpt ./checkpoints/humaneva_ckpt.pt

python main.py --cfg amass --mode pred --vis_row 3 --vis_col 10 --ckpt ./checkpoints/amass_ckpt.pt



```
## 🌹 Acknowledgment
Project structure is borrowed from [HumanMAC](https://github.com/LinghaoChan/HumanMAC). We would like to thank the authors for making their code publicly available.


## 📚 License
The software in this repository is freely available for free non-commercial use (see [license](https://github.com/sibotian96/TransFusion/blob/main/LICENSE) for further details).
