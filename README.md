
# **UTR k-mer Embedding and Machine Learning Pipeline**

This repository contains a Python-based pipeline for:
- Generating **k-mer embeddings** from UTR sequences.
- Training machine learning models (**CNN** and **Random Forest**) to classify sequences based on a gene list.
- Visualising results using **t-SNE** plots and **ROC curves**.
- Extracting the most **informative k-mers** using **Grad-CAM** and feature importance scores.

## **Table of Contents**
1. [Installation](#installation)
2. [File Structure](#file-structure)
3. [Script Overview](#script-overview)
4. [How to Use](#how-to-use)
5. [Functions Overview](#functions-overview)
6. [Example Commands](#example-commands)
7. [Output Files](#output-files)

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/utr-kmer-project.git
   cd utr-kmer-project
   ```

2. Create a conda environment and install dependencies:
   ```bash
   conda create -n utr_kmer_env python=3.9
   conda activate utr_kmer_env
   pip install -r requirements.txt
   ```

3. Dependencies:
   - `Bio` (Biopython) for parsing FASTA files.
   - `numpy`, `pandas` for data processing.
   - `matplotlib` for plots.
   - `torch` and `torchvision` for CNNs.
   - `scikit-learn` for Random Forest and metrics.
   - `captum` for Grad-CAM visualisation.

## **File Structure**
```
/utr-kmer-project/
├── inputs/                # Input sequences and gene lists (excluded by .gitignore)
├── outputs/               # Results and plots (excluded by .gitignore)
├── utr_kmer_embeddings.py # Main script
├── .gitignore             # Ignored files
└── README.md              # Documentation
```

## **Script Overview**
`utr_kmer_embeddings.py` contains the following key steps:
1. **Generate k-mer embeddings:** Convert sequences to feature vectors of k-mers.
2. **Train Models:** Train a **CNN** and **Random Forest** for sequence classification.
3. **Extract Important k-mers:** Save the top 100 k-mers based on Grad-CAM and feature importance.
4. **Visualise Outputs:** Create **t-SNE plot** and **ROC curve**.

## **How to Use**
### **Input Requirements**
- **FASTA file**: UTR sequences in FASTA format (`--fasta_file`).
- **Gene list**: A text file containing a list of genes of interest, one per line (`--gene_list_file`).

### **Run the Script**
```bash
python utr_kmer_embeddings.py --fasta_file inputs/utr_sequences.fasta --gene_list_file inputs/gene_list.txt --tsne --num_epochs 10 --batch_size 32 --max_sequences 2000 --verbose
```

### **Arguments**
| Argument             | Description                                      | Default     |
|----------------------|--------------------------------------------------|-------------|
| `--fasta_file`        | Path to the UTR FASTA file                       | Required    |
| `--gene_list_file`    | Path to the gene list file                       | Required    |
| `--k`                | k-mer size                                        | `6`         |
| `--num_epochs`        | Number of CNN training epochs                    | `10`        |
| `--batch_size`        | Batch size for CNN training                      | `32`        |
| `--learning_rate`     | Learning rate for CNN training                   | `0.001`     |
| `--max_sequences`     | Maximum number of sequences to load (for testing)| `None`      |
| `--tsne`              | Enable t-SNE visualisation                       | `False`     |
| `--perplexity`        | t-SNE perplexity                                 | `30`        |
| `--learning_rate_tsne`| t-SNE learning rate                              | `200`       |
| `--verbose`           | Enable verbose mode for detailed output          | `False`     |

## **Functions Overview**
### **1. `generate_embeddings()`**
- **Purpose:** Converts UTR sequences into k-mer embeddings.
- **Inputs:** FASTA file, k-mer size.
- **Outputs:** DataFrame of k-mer counts for each sequence.

### **2. `train_cnn_model()`**
- **Purpose:** Trains a CNN for binary sequence classification.
- **Inputs:** k-mer embeddings, gene list, training parameters.
- **Outputs:** ROC AUC, predictions, and Grad-CAM attributions.
- **Saves:** Top 100 k-mers to `outputs/cnn_top_features.txt`.

### **3. `train_rf_model()`**
- **Purpose:** Trains a Random Forest for binary sequence classification.
- **Inputs:** k-mer embeddings, gene list.
- **Outputs:** ROC AUC, predictions.
- **Saves:** Top 100 k-mers to `outputs/rf_top_features.txt`.

### **4. `plot_tsne()`**
- **Purpose:** Creates a t-SNE plot to visualise embeddings in 2D space.
- **Outputs:** Saves the t-SNE plot as `outputs/tsne_plot.png`.

### **5. `plot_roc_curve()`**
- **Purpose:** Plots and saves the ROC curve comparing CNN and Random Forest performance.
- **Outputs:** Saves the ROC curve as `outputs/roc_curve.png`.

## **Example Commands**
1. **Run with t-SNE and CNN for 2000 sequences:**
   ```bash
   python utr_kmer_embeddings.py --fasta_file inputs/utr_sequences.fasta --gene_list_file inputs/gene_list.txt --tsne --max_sequences 2000
   ```

2. **Run only with Random Forest without visualisation:**
   ```bash
   python utr_kmer_embeddings.py --fasta_file inputs/utr_sequences.fasta --gene_list_file inputs/gene_list.txt --num_epochs 0
   ```

## **Output Files**
| File                       | Description                                    |
|----------------------------|------------------------------------------------|
| `outputs/tsne_plot.png`     | t-SNE plot of k-mer embeddings                 |
| `outputs/roc_curve.png`     | ROC curve comparing CNN and Random Forest      |
| `outputs/cnn_top_features.txt` | Top 100 k-mers ranked by CNN (Grad-CAM)      |
| `outputs/rf_top_features.txt`  | Top 100 k-mers ranked by Random Forest       |

## **Contributions and Issues**
Feel free to open an issue or contribute to this repository if you have suggestions or improvements!
