#!/usr/bin/env python3
"""Extract embeddings from ProteinDataset (one-hot-segment), run PCA+UMAP, and save plots.

Usage examples:
  poetry run python scripts/embeddings_umap_onehot_segment.py --sample-size 2000

This script is conservative by default (samples up to 2000 points). It saves:
  - outputs/onehot_segment_umap.png
  - outputs/onehot_segment_umap.html
  - outputs/onehot_segment_embeddings.npz

  command:
  python scripts/embeddings_umap_onehot_segment.py --sample-size 2000

  to view interactive HTML:
  open outputs/onehot_segment_umap.html

"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Ensure the repository root is on sys.path so `import data` works when running
# this file as a script (python scripts/...). When executed this way, sys.path[0]
# is the scripts/ folder which prevents imports of sibling packages.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from data.dataset import ProteinDataset


def extract_embeddings(dataset, batch_size=256, sample_size=None, device='cpu'):
    if sample_size is not None and sample_size < len(dataset):
        indices = np.random.choice(len(dataset), size=sample_size, replace=False)
        dataset = Subset(dataset, indices)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embs = []
    scores = []
    ids = []
    for batch in loader:
        X, y = batch
        # X is flattened one-hot + PE
        X = X.detach().cpu().numpy()
        embs.append(X)
        scores.append(y.detach().cpu().numpy())

    embs = np.vstack(embs)
    scores = np.concatenate(scores)
    return embs, scores


def reduce_and_plot(embs, scores, outdir: Path, pca_dim=50, umap_n=15, umap_min_dist=0.1, random_state=42):
    outdir.mkdir(parents=True, exist_ok=True)

    # PCA first if highdim
    if embs.shape[1] > pca_dim:
        pca = PCA(n_components=pca_dim, random_state=random_state)
        embs_p = pca.fit_transform(embs)
    else:
        embs_p = embs

    reducer = umap.UMAP(n_neighbors=umap_n, min_dist=umap_min_dist, metric='cosine', random_state=random_state)
    emb2 = reducer.fit_transform(embs_p)

    # static plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=emb2[:, 0], y=emb2[:, 1], hue=scores, palette='viridis', s=8, legend=False)
    plt.title('One-hot-segment embeddings (UMAP)')
    png_path = outdir / 'onehot_segment_umap.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()

    # interactive
    df = pd.DataFrame({'x': emb2[:, 0], 'y': emb2[:, 1], 'score': scores})
    fig = px.scatter(df, x='x', y='y', color='score', color_continuous_scale='viridis', hover_data=['score'])
    html_path = outdir / 'onehot_segment_umap.html'
    fig.write_html(str(html_path))

    # save embeddings and reduced arrays
    np.savez_compressed(outdir / 'onehot_segment_embeddings.npz', embs=embs, embs_p=embs_p, emb2=emb2, scores=scores)

    return png_path, html_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--split', type=str, default='train')
    p.add_argument('--sample-size', type=int, default=2000, help='Number of points to sample (None = all)')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--pca-dim', type=int, default=50)
    p.add_argument('--umap-n', type=int, default=15)
    p.add_argument('--umap-min-dist', type=float, default=0.1)
    p.add_argument('--outdir', type=Path, default=Path('outputs'))
    args = p.parse_args()

    variants = pd.read_csv(args.data_dir / f"{args.split}.csv")
    Sequences = pd.read_csv(args.data_dir / 'Sequences.csv', index_col='ensp')

    ds = ProteinDataset(split=args.split, variants=variants, Sequences=Sequences, encoding='one-hot-segment')

    sample_size = args.sample_size if args.sample_size is not None and args.sample_size > 0 else None
    embs, scores = extract_embeddings(ds, batch_size=args.batch_size, sample_size=sample_size)

    png, html = reduce_and_plot(embs, scores, args.outdir, pca_dim=args.pca_dim, umap_n=args.umap_n, umap_min_dist=args.umap_min_dist)

    print('Saved:', png, html)


if __name__ == '__main__':
    main()
