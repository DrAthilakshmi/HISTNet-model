# HISTNet – Implementation Guide

### Overview
Full implementation of **HISTNet**: A Hybrid InceptionResnetv2-DCNN and Swin Transformer Network for Breast Cancer Histopathology Classification. This repository contains the code to train and evaluate the model using the 2-class BreaKHis dataset and the 4-class BACH dataset. 

HISTNet operates via a dual-stream architecture:
1. **Local Feature Stream:** Utilizes InceptionResNet-v2 extended with a custom deep CNN block to extract fine-grained nuclear feature patterns.
2. **Global Feature Stream:** Employs a Swin Transformer to model long-range spatial relationships.
The outputs are integrated using an attention-based fusion block for highly accurate classification.

### Quantitative Results
The proposed HISTNet outperforms standalone models. Below is the overall performance of our dual-stream fusion approach (derived from 5-fold cross-validation):

**Table 1: Overall Performance of HISTNet**
| Dataset | Classes | Accuracy (%) | Precision (%) | F1-Score (%) |
| :--- | :--- | :--- | :--- | :--- |
| **BreaKHis** | 2-Class | **95.88 ± 0.5** | 95.8 ± 0.6 | 95.0 ± 0.5 |
| **BACH** | 4-Class | **91.0 ± 3.2** | 91.3 ± 3.3 | 91.0 ± 3.1 |

### Ablation Study Results
To demonstrate the effectiveness of our hybrid approach, we compared HISTNet against its standalone local and global feature extraction streams:

**Table 2: Ablation Study of HISTNet Components**
| Dataset & Metric | CNN Stream (Local Texture) | Swin Stream (Global Context) | **HISTNet (Dual-Stream Fusion)** |
| :--- | :--- | :--- | :--- |
| **BreaKHis (Binary Class)** | | | |
| Accuracy (%) | 94.5 ± 1.2 | 94.1 ± 1.14 | **95.88 ± 0.5\*** |
| Precision (%) | 94.2 ± 1.5 | 94.5 ± 1.13 | **95.8 ± 0.6** |
| F1-Score (%) | 94.8 ± 1.3 | 94.8 ± 1.3 | **95.0 ± 0.5** |
| **BACH (Multi Class)** | | | |
| Accuracy (%) | 87.5 ± 1.8 | 88.2 ± 1.1 | **91.0 ± 3.2\*** |
| Precision (%) | 87.5 ± 1.0 | 87.9 ± 1.2 | **91.3 ± 3.3** |
| F1-Score (%) | 87.8 ± 1.0 | 87.2 ± 2.1 | **91.0 ± 3.1** |

---

### Installation & Running the Pipeline
```bash
# Clone the repository
git clone [https://github.com/DrAthilakshmi/HISTNet.git](https://github.com/YOUR-USERNAME/HISTNet.git)
cd HISTNet

# Install required dependencies
pip install -r requirements.txt

# Run the training script (Example)
python train_histnet.py --dataset breakhis --data_dir ./data/breakhis/ --epochs 50
````

### Dataset Requirements & Links

This code is designed to work with two publicly available datasets:

**1. BreaKHis Dataset (2-Class)**

  * **Link:** [Download BreaKHis on Kaggle](https://www.kaggle.com/datasets/ambarish/breakhis)
  * **Classes:** Benign vs. Malignant
  * **Magnifications:** 400X
  * **Structure:** Ensure images are organized into subfolders by class and magnification level.

**2. BACH Dataset (4-Class)**

  * **Link:** [Download BACH Challenge on Kaggle](https://www.google.com/search?q=https://www.kaggle.com/datasets/nabeelsajid917/iciar-2018-bach-challenge)
  * **Classes:** Normal, Benign, In situ carcinoma, Invasive carcinoma
  * **Structure:** Ensure images are sorted into the four respective class directories.

**Automated Download via Kaggle API:**

```bash
# Download and unzip the BreaKHis dataset
kaggle datasets download -d ambarish/breakhis
unzip breakhis.zip -d data/breakhis/

# Download and unzip the BACH dataset
kaggle datasets download -d nabeelsajid917/iciar-2018-bach-challenge
unzip iciar-2018-bach-challenge.zip -d data/bach/
```

### Arguments

| Argument | Description |
| :--- | :--- |
| `--dataset` | Choose the target dataset: `breakhis` (2-class) or `bach` (4-class) |
| `--data_dir` | Path to the root directory of the downloaded dataset |
| `--epochs` | Number of training epochs (default: 50) |
| `--batch_size` | Batch size for training (default: 32) |
| `--out` | Output directory for saved models, logs, and plots (default: `./results`) |

### Outputs Generated

| File | Description |
| :--- | :--- |
| `best_histnet_model.pth` | Saved weights for the best performing model |
| `training_log.csv` | Epoch-by-epoch loss and accuracy tracking |
| `confusion_matrix.png` | Visual matrix of predicted vs actual classes |
| `attention_maps.png` | Attention map visualizations highlighting salient tissue regions |

