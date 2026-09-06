# BA-MSAT-Net Implementation

```

This repository contains the official PyTorch implementation of BA‑MSAT‑Net as described in the paper:

\*\*BA‑MSAT‑Net: Boundary‑Aware Multi‑Scale Attention Transformer Network for CBCT Tooth and Alveolar Bone Segmentation\*\*.



\## Table of Contents

\- \[Environment Requirements](#environment-requirements)

\- \[Installation](#installation)

\- \[Dataset Preparation](#dataset-preparation)

\- \[Quick Start](#quick-start)

\- \[Project Directory Structure](#project-directory-structure)

\- \[Citation](#citation)

\- \[License](#license)



\## Environment Requirements

\- Python >= 3.9

\- PyTorch >= 1.10

\- Recommended GPU environment: CUDA 11.3 / 11.7



Main dependencies: `torch`, `torchvision`, `nibabel`, `numpy`, `tensorboard`, `pyyaml`, `SimpleITK`, `MONAI`, `scipy`, `matplotlib`, `tqdm`



\## Installation

Clone this repository and install all dependencies.

```bash

\# Clone repository

git clone https://github.com/your-org/BA-MSAT-Net-master.git

cd BA-MSAT-Net-master



\# Install python dependencies

pip install -r requirements.txt

```



> 

> \*\*Note for PyTorch CUDA\*\*:

> If you encounter CUDA runtime mismatch, install PyTorch matching your local CUDA toolkit from \[PyTorch official website](https://pytorch.org/get-started/locally/).

> Example for CUDA 11.7:

> 

> 

> ```

> pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu117

> ```



\## Dataset Preparation



The CBCT tooth and alveolar bone dataset is hosted on Zenodo. All data are stored in NIfTI (`\*.nii.gz`) format.



All the data is available through Zenodo (DOI:10.5281/zenodo.22220948) and Github(https://github.com/xjguochao/BA-MSAT-NET/).

\### Step 2: Folder Layout



Unzip dataset and place under `./datasets/cbct`. The expected directory structure is strictly as follows:



```

BA-MSAT-Net-master/

└── datasets/

&#x20;   └── cbct/

&#x20;       ├── imagesTr/      # Training CBCT volumes (\*.nii.gz)

&#x20;       ├── labelsTr/      # Training segmentation ground truth (\*.nii.gz)

&#x20;       ├── imagesTs/      # Test CBCT volumes (\*.nii.gz)

&#x20;       └── labelsTs/      # Test ground truth (optional, for metrics calculation)

```



> 

> Important:

> 

> 

> - Dataset root path is defined inside `dataloaders/cbct\_dataset.py`. Modify this file if you change dataset storage path.

> - All input volumes must be NIfTI‑gz format. Other formats are not supported.



\## Quick Start



\### Inference



Run inference using checkpoint obtained from your own training process:



```

python inference.py \\

&#x20; --weights ./logs/train\_exp01/checkpoints/your\_trained\_checkpoint.pth \\

&#x20; --data\_root ./datasets/cbct \\

&#x20; --output ./output/inference\_results

```



Parameters explanation:



\- `--weights`: Path to your locally trained model checkpoint

\- `--data\_root`: Root directory of CBCT dataset

\- `--output`: Output folder for predicted segmentation NIfTI files



Predicted segmentation masks will be saved to `./output/inference\_results`.



> 

> Segmentation metrics (Dice, IoU, HD95) are implemented in `utils/metrics.py`. You can invoke this module to compute quantitative results against test‑set ground‑truth.



\### Training



Train BA‑MSAT‑Net from scratch on your CBCT dataset:



```

python train.py \\

&#x20; --data\_root ./datasets/cbct \\

&#x20; --logdir ./logs/train\_exp01

```



\- Tensorboard training logs are saved in `./logs/train\_exp01`

\- Model checkpoints will be auto‑saved to `./logs/train\_exp01/checkpoints/`

\- Launch tensorboard to monitor training:



```

tensorboard --logdir ./logs/train\_exp01

```



\## Project Directory Structure



```

BA‑MSAT‑Net‑master

├── dataloaders/

│   ├── \_\_init\_\_.py

│   └── cbct\_dataset.py      # CBCT dataset reader

├── datasets/                 # Dataset storage directory (place cbct data here)

│   └── \_\_init\_\_.py

├── detection/                # auxiliary detection modules

├── models/

│   ├── \_\_init\_\_.py

│   ├── ba\_msat\_net.py        # Main BA‑MSAT‑Net architecture

│   └── modules.py            # Attention \& boundary‑aware blocks

├── utils/

│   ├── \_\_init\_\_.py

│   ├── losses.py             # Custom loss functions

│   └── metrics.py            # Segmentation metrics(Dice,IoU,HD95 etc.)

├── logs/                     # Training logs \& checkpoints (auto‑generated)

├── output/                   # Inference outputs (auto‑generated)

├── inference.py              # Inference entry script

├── train.py                  # Training entry script

├── README.md

└── requirements.txt

```



\## Citation



If you find this repository useful for your research, please cite our paper:



```

@article{bamsatnet202x,

&#x20; title={BA‑MSAT‑Net: Boundary‑Aware Multi‑Scale Attention Transformer Network for CBCT Tooth and Alveolar Bone Segmentation},

&#x20; author={ },

&#x20; journal={},

&#x20; year={}

}

```



\## License



This project is released under the MIT License.

