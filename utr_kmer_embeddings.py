# Import necessary libraries
from Bio import SeqIO
from collections import Counter
import numpy as np
import pandas as pd
import argparse
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam
import os

# Create output directory if it doesn't exist
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CNN model definition
class CNNKmerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(CNNKmerClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * (input_dim // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Function to load sequences
def load_sequences(fasta_file, max_sequences=None, verbose=False):
    utr_sequences = {}
    for i, record in enumerate(SeqIO.parse(fasta_file, "fasta")):
        if max_sequences and i >= max_sequences:
            break
        utr_sequences[record.id] = str(record.seq)
    if verbose:
        print(f"Loaded {len(utr_sequences)} UTR sequences")
    return utr_sequences

# Function to generate k-mer embeddings
def generate_embeddings(fasta_file, k=6, max_sequences=None, verbose=False):
    utr_sequences = load_sequences(fasta_file, max_sequences, verbose)
    vocab = set()
    for seq in utr_sequences.values():
        vocab.update([seq[i:i + k] for i in range(len(seq) - k + 1)])
    vocab = list(vocab)

    utr_embeddings = {}
    for seq_id, seq in utr_sequences.items():
        kmer_counts = Counter([seq[i:i + k] for i in range(len(seq) - k + 1)])
        utr_embeddings[seq_id] = [kmer_counts.get(kmer, 0) for kmer in vocab]

    df = pd.DataFrame.from_dict(utr_embeddings, orient='index', columns=vocab)
    df.index.name = "Gene"
    df.fillna(0, inplace=True)
    if verbose:
        print(f"Generated k-mer embeddings with shape {df.shape}")
    return df, vocab

# Function to perform t-SNE visualisation
def plot_tsne(embedding_df, label_col="Label", perplexity=30, learning_rate=200, output_plot=f"{OUTPUT_DIR}/tsne_plot.png", verbose=False):
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    reduced = tsne.fit_transform(embedding_df.drop(columns=[label_col], errors='ignore').values)

    if verbose:
        print(f"t-SNE result shape: {reduced.shape}")
        print(f"First 5 t-SNE points:\n{reduced[:5]}")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=embedding_df[label_col], cmap="viridis", alpha=0.7)
    plt.title(f"t-SNE of UTR k-mer Embeddings (coloured by {label_col})")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.colorbar(scatter, label=label_col)
    plt.savefig(output_plot)
    plt.show()
    print(f"t-SNE plot saved as {output_plot}")

# Function to plot ROC curve
def plot_roc_curve(cnn_fpr, cnn_tpr, cnn_auc, rf_fpr, rf_tpr, rf_auc, output_file=f"{OUTPUT_DIR}/roc_curve.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(cnn_fpr, cnn_tpr, label=f"CNN (AUC = {cnn_auc:.4f})", linewidth=2)
    plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.savefig(output_file)
    plt.show()
    print(f"ROC curve saved as {output_file}")

# Function to train CNN model
def train_cnn_model(embedding_df, gene_list, num_epochs=10, batch_size=32, lr=0.001, verbose=False):
    embedding_df["Label"] = embedding_df.index.isin(gene_list).astype(int)
    X = embedding_df.drop(columns=["Label"]).values
    y = embedding_df["Label"].values

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = CNNKmerClassifier(input_dim=X_tensor.shape[2])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    cnn_probs = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = torch.exp(model(batch_X))
            cnn_probs.extend(outputs[:, 1].cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    cnn_fpr, cnn_tpr, _ = roc_curve(all_labels, cnn_probs)
    cnn_auc = auc(cnn_fpr, cnn_tpr)
    return cnn_fpr, cnn_tpr, cnn_auc

# Function to train Random Forest model
def train_rf_model(embedding_df, gene_list, verbose=False):
    embedding_df["Label"] = embedding_df.index.isin(gene_list).astype(int)
    X = embedding_df.drop(columns=["Label"])
    y = embedding_df["Label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    rf_fpr, rf_tpr, _ = roc_curve(y_test, y_proba)
    rf_auc = auc(rf_fpr, rf_tpr)

    if verbose:
        print(f"Random Forest AUC: {rf_auc:.4f}")
    return rf_fpr, rf_tpr, rf_auc

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN and Random Forest, extract k-mer relationships, and plot ROC curves.")
    parser.add_argument("--fasta_file", type=str, required=True, help="Path to the UTR FASTA file")
    parser.add_argument("--gene_list_file", type=str, required=True, help="Path to a file containing the list of target genes")
    parser.add_argument("--k", type=int, default=6, help="k-mer size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of CNN training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for CNN training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for CNN")
    parser.add_argument("--max_sequences", type=int, default=None, help="Maximum number of sequences to load for testing")
    parser.add_argument("--tsne", action="store_true", help="Enable t-SNE visualisation")
    parser.add_argument("--perplexity", type=int, default=30, help="t-SNE perplexity")
    parser.add_argument("--learning_rate_tsne", type=int, default=200, help="t-SNE learning rate")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode for debugging")

    args = parser.parse_args()

    with open(args.gene_list_file, "r") as f:
        gene_list = [line.strip() for line in f.readlines()]

    embedding_df, _ = generate_embeddings(args.fasta_file, args.k, max_sequences=args.max_sequences, verbose=args.verbose)

    # Train CNN
    cnn_fpr, cnn_tpr, cnn_auc = train_cnn_model(embedding_df, gene_list, num_epochs=args.num_epochs, batch_size=args.batch_size, lr=args.learning_rate, verbose=args.verbose)
    print(f"CNN AUC: {cnn_auc:.4f}")

    # Train Random Forest
    rf_fpr, rf_tpr, rf_auc = train_rf_model(embedding_df, gene_list, verbose=args.verbose)
    print(f"Random Forest AUC: {rf_auc:.4f}")

    # Plot ROC curve
    plot_roc_curve(cnn_fpr, cnn_tpr, cnn_auc, rf_fpr, rf_tpr, rf_auc)

    # Perform t-SNE visualisation if requested
    if args.tsne:
        plot_tsne(embedding_df, label_col="Label", perplexity=args.perplexity, learning_rate=args.learning_rate_tsne, output_plot=f"{OUTPUT_DIR}/tsne_plot.png", verbose=args.verbose)