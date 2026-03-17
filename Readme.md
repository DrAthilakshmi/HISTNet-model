# HISTNet – Implementation Guide

### Overview
Full implementation of **HISTNet**: A Hybrid InceptionResnetv2-DCNN and Swin Transformer Network for Breast Cancer Histopathology Classification. This repository contains the code to train and evaluate the model using the 2-class BreaKHis dataset and the 4-class BACH dataset. 

HISTNet operates via a dual-stream architecture:
1. **Local Feature Stream:** Utilizes InceptionResNet-v2 extended with a custom deep CNN block to extract fine-grained nuclear feature patterns.
2. **Global Feature Stream:** Employs a Swin Transformer to model long-range spatial relationships.
The outputs are integrated using an attention-based fusion block for highly accurate classification.

### Installation & Running the Pipeline
```bash
# Clone the repository
git clone [https://github.com/DrAthilakshmi/HISTNet-model/HISTNet.git]
cd HISTNet

# Install required dependencies
pip install -r requirements.txt

# Run the training script (Example)
python train_histnet.py --dataset breakhis --data_dir ./data/breakhis/ --epochs 50
