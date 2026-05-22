# SL-S4Wave: Self-Supervised Learning of Physiological Waveforms with Structured State Space Models

> Official PyTorch implementation of the paper:  
> **"SL-S4Wave: Self-Supervised Learning of Physiological Waveforms with Structured State Space Models"**  
> *Transactions on Machine Learning Research, 2026*

---

## Overview

SL-S4Wave is a self-supervised framework for physiological waveform analysis. It leverages **Structured State Space Models (S4)** within a WaveNet-style architecture to capture long-range temporal dependencies in multi-channel biosignals. The model is pre-trained via contrastive learning (NT-Xent loss) and fine-tuned for cardiac alarm classification.

**Key features:**
- S4-based sequence modeling for long-range biosignal dependencies
- Self-supervised pre-training with contrastive loss — no labels required at pre-training stage
- Supports 4-channel physiological signals (ECG leads + PLETH)
- Evaluated on VTaC, MIMIC, and PhysioNet/CinC Challenge 2015 datasets
- Multi-node distributed training support via SLURM

---

## Requirements

```
Python >= 3.10
PyTorch >= 2.0
CUDA 
```

Install dependencies:

```bash
conda create -n sls4wave python=3.10 -y
conda activate sls4wave
pip install -r requirements.txt
```

---

## Data Preparation

### VTaC Dataset
The VTaC dataset is publicly available on [PhysioNet](https://physionet.org/content/vtac/1.0/). The original data format is WFDB, you need to follow the preprocessing steps provided in the [VTaC GitHub repository](https://github.com/ML-Health/VTaC) to generate the required `.pt` files.

The expected directory structure after preprocessing:

```
/data/vtac/
├── train.pt      # (samples, groundtruth, names)
├── val.pt        # (samples, groundtruth, record_ids)
└── test.pt       # (samples, groundtruth, record_ids)
```

Each sample tensor has shape `(N, 4, 75000)` — N samples, 4 channels, 75000 time steps (at 250 Hz, ~5 minutes).

### MIMIC Dataset
Access the MIMIC dataset via [Physionet MIMIC III wdb](https://physionet.org/content/mimic3wdb/1.0/). You can use the script `Data Process/Data_process.py` to convert the raw MIMIC waveform data (WFDB Format) into the required `.pt` format. 



### PhysioNet/CinC Challenge 2015 Dataset
Download from [PhysioNet Challenge 2015](https://physionet.org/content/challenge-2015/1.0.0/). The dataset includes alarm types: **VT (Ventricular Tachycardia), VFib, Asystole, Tachycardia, Bradycardia**. You can use the script `Data Process/Data_process.py` to convert the raw waveform data (WFDB Format) into the required `.pt` format. 

### Pre-training Data
Pre-training uses a large unlabeled waveform collection. You can download our preprocessed pre-training dataset from [Coming Soon]().

Place the file at:
```
datasets/pre-train.pt
```

---

## Pre-training

```bash
# Pre-train from scratch
python pre-train.py \
    --learning_rate 1e-4 \
    --batch_size 16 \
    --max_epoch 50 \
    --data_length 1250

# Resume from a checkpoint
python pre-train.py \
    --checkpoint models/<timestamp>_pretrain/checkpoint_epoch_10.pt
```

Checkpoints are saved every epoch under `models/<timestamp>_pretrain/`. A full model snapshot is saved every 10 epochs.

---

## Fine-tuning & Evaluation

### VTaC / MIMIC

```bash
# Fine-tune from scratch
python run.py \
    --dataset VTac \
    --data_dir /data/vtac \
    --batch_size 16 \
    --max_epoch 60 \
    --data_length 1250

# Fine-tune from pre-trained checkpoint
python run.py \
    --dataset VTac \
    --data_dir /data/vtac \
    --batch_size 16 \
    --max_epoch 60 \
    --data_length 1250 \
    --pretrain \
    --pretrained_path models/<timestamp>_pretrain/model_pretrain.pt
```

Replace `--dataset` with `mimic` and `--data_dir` with the MIMIC data path as needed.

### PhysioNet/CinC Challenge 2015

```bash
# Fine-tune from scratch
python run_multi.py \
    --dataset 2015 \
    --data_dir /data/2015 \
    --batch_size 16 \
    --max_epoch 50 \
    --data_length 1250

# Fine-tune from pre-trained checkpoint
python run_multi.py \
    --dataset 2015 \
    --data_dir /data/2015 \
    --batch_size 16 \
    --max_epoch 50 \
    --data_length 1250 \
    --pretrain \
    --pretrained_path models/<timestamp>_pretrain/model_pretrain.pt
```

`run_multi.py` also supports individual alarm sub-type training (e.g., ASY, VFIB, AFIB) via `--sub_dataset`.

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `VTac` | Dataset name: `VTac`, `mimic`, or `2015` |
| `--data_dir` | — | Path to directory containing `train.pt`, `val.pt`, `test.pt` |
| `--batch_size` | `16` | Training batch size |
| `--max_epoch` | `60` | Number of training epochs |
| `--data_length` | `1250` | Input sequence length (crop length) |
| `--learning_rate` | `2e-6` | AdamW learning rate |
| `--pretrain` | `False` | Load pre-trained weights if set |
| `--pretrained_path` | — | Path to pre-trained model `.pt` file |
| `--seed` | `0` | Random seed for reproducibility |
| `--save_dir` | `models/` | Directory for saving models and logs |



---

## Evaluation Metrics

The model is evaluated using the following metrics:

- **Score**: `(TP + TN) / (TP + TN + FP + 5·FN)` — the primary metric; penalizes missed alarms (false negatives) 5× more heavily than false positives, reflecting clinical priority
- **TPR** (Sensitivity / Recall)
- **TNR** (Specificity)
- **F1 Score**
- **PPV** (Precision)
- **AUC-ROC**

---

## Pre-trained Models

You can find our pre-trained model checkpoints on [Coming Soon]().

---

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{SL-S4Wave2025,
  title={SL-S4Wave: Self-Supervised Learning of Physiological Waveforms with Structured State Space Models},
  author={...},
  booktitle={Transactions on Machine Learning Research},
  year={2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

- The S4 layer implementation builds on [Structured State Spaces](https://github.com/HazyResearch/state-spaces).
- The contrastive pre-training loss is adapted from [NT-Xent](https://arxiv.org/abs/2002.05709).
- VTaC dataset processing follows the pipeline from [VTaC GitHub](https://github.com/ML-Health/VTaC).
