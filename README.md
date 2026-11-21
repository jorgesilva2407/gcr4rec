# gcr4rec: Graph Collaborative Reasoning for Recommendation

A from-scratch PyTorch reimplementation of **Graph Collaborative Reasoning** (GCR; Chen et al., WSDM 2022), written because no working public code was available. It adds two extensions: **FuzzGCR**, which swaps GCR's learned logic modules for fixed fuzzy-logic operators, and **FA-GCR**, which reasons over user and item attributes instead of IDs alone.
This is my BSc capstone project (Projeto Orientado em Computação II, UFMG, 2025/2), advised by Prof. Rodrygo Santos.

## Main results

NDCG@5 on the test set. Each test item is ranked against 99 sampled negatives (see [Evaluation caveats](#evaluation-caveats)).

| Dataset | BiasedSVD | GCR | best FuzzGCR¹ | FA-GCR (w/ ID) | FA-GCR (w/o ID) |
|---|---|---|---|---|---|
| ML-100K | 0.073 | 0.368 | 0.175 | 0.501 | **0.546** |
| ML-1M | 0.213 | 0.455 | 0.197 | 0.460 | **0.483** |
| Amazon Beauty | 0.147 | **0.304** | 0.115 | — | — |
| Amazon Clothing | 0.111 | **0.330** | 0.107 | — | — |

¹ The best of the product, Gödel and smooth variants on each dataset: product on ML-100K, Gödel on ML-1M and Clothing, smooth on Beauty.
FA-GCR was run only on MovieLens, because the Amazon datasets have no user features.

- **FA-GCR without ID embeddings scores highest on both MovieLens datasets.** NDCG@5 is 48% higher than GCR on ML-100K and 6% higher on ML-1M. The variant without IDs beats the one with them on both datasets.
- **GCR beats BiasedSVD on NDCG@5 on all four datasets.** The margin ranges from about 2× (ML-1M, Beauty) to 5× (ML-100K).
- **FuzzGCR is a negative result.** It falls below GCR on every dataset, and below BiasedSVD on everything except ML-100K. The score is a long disjunction, `OR(¬n₁, …, ¬nₖ, target)`, which by De Morgan's law equals `¬(n₁ ∧ … ∧ nₖ ∧ ¬target)`. Chaining a t-norm over many terms pushes the truth value of the conjunction toward zero, so the disjunction saturates at one. The gradient that reaches any single term shrinks as the chain grows: under the product t-norm it is scaled by the product of all the other terms, and under Gödel `max`, only one term receives a gradient at all. With up to 2k+1 terms per prediction, the event embeddings get almost no learning signal. GCR's learned modules avoid this because they are not tied to bounded truth values.

Hyperparameters were tuned with Optuna (TPE sampler): about 430 runs across 24 model–dataset combinations, on Colab GPUs.

## Models

**GCR.** Each user–item pair (u, i) is encoded by an MLP as an *event* embedding. To score a target pair, GCR samples up to k items the user interacted with and up to k users who interacted with the item. It then builds the Horn clause "(u likes these neighbors) ∧ (these neighbors like i) → u likes i" and evaluates it as `OR(NOT(e₁), …, NOT(e₂ₖ), e_target)`. The score is the cosine similarity between the result and a fixed TRUE anchor vector. NOT and OR are small MLPs, kept close to logical behavior by regularizers on negation, double negation, identity, annihilation, idempotence and complement. The order of the terms is shuffled during training. ([`src/models/gcr.py`](src/models/gcr.py), [`src/models/base_gcr.py`](src/models/base_gcr.py))

**FuzzGCR.** The reasoning graph is the same as in GCR, but events live in `[0,1]^d` (sigmoid output), NOT is `1 − x`, and OR is a fixed t-conorm: product (`x + y − xy`), Gödel (`max`), smooth (log-sum-exp with a stiffness parameter) or Łukasiewicz (`min(x + y, 1)`). Łukasiewicz is implemented but was not tuned. The fixed operators have no parameters to learn, and they satisfy the logical laws by construction, so no logic regularizers are needed. ([`src/models/fuzzy_gcr.py`](src/models/fuzzy_gcr.py), [`src/models/logical_modules/fuzzy.py`](src/models/logical_modules/fuzzy.py))

**FA-GCR.** This replaces the ID lookup tables with a feature-aware encoder. Optional ID embeddings, embeddings of categorical attributes and raw numeric or boolean attributes are concatenated, then projected by a linear layer with ReLU to a fixed size (`--encoder-output-dim`) before entering the GCR reasoning step. On MovieLens, the user features are gender, age, occupation and zip code, and the item features are release year and genres. The two variants differ only in whether ID embeddings are included. ([`src/models/feature_aware_gcr.py`](src/models/feature_aware_gcr.py), [`src/models/encoders/feature_aware_encoder.py`](src/models/encoders/feature_aware_encoder.py))

**BiasedSVD** is the matrix-factorization baseline: a dot product of user and item embeddings plus user and item biases.

## Data and protocol

| Dataset | Users | Items | Interactions | Density |
|---|---:|---:|---:|---:|
| MovieLens 100K | 943 | 1,682 | 100,000 | 6.30% |
| MovieLens 1M | 6,040 | 3,706 | 1,000,209 | 4.47% |
| Amazon Beauty (2014, 5-core) | 22,363 | 12,101 | 198,502 | 0.073% |
| Amazon Clothing, Shoes & Jewelry (2014, 5-core) | 39,387 | 23,033 | 278,677 | 0.031% |

Ratings are treated as implicit feedback. The processed data is committed under `data/processed/`.

- **Split:** leave-one-out by timestamp. Each user's last interaction is the test item, the second-to-last is the validation item, and the rest are for training.
- **Evaluation:** each positive is ranked against 99 items the user never interacted with. Test negatives are precomputed once with seed 42. Validation negatives are resampled each time, on a fixed random 10% subset of validation users. Ties count against the positive.
- **Metrics:** NDCG@5 and NDCG@10 during training. HR@1/5/10 are computed from the saved test predictions in `ResultAnalysis.ipynb`.
- **Loss:** BPR with one sampled negative per positive. Scores are scaled by an inverse temperature: 10 for GCR and FA-GCR, 20 for FuzzGCR, 1 for BiasedSVD. GCR and FA-GCR add the weighted logic regularizers.
- **Training:** Adam (learning rate and weight decay are tuned) for up to 50 epochs. Each epoch draws one (positive, negative) pair per user. Early stopping watches validation NDCG@5, with patience 7 after 10 warm-up epochs, and the best checkpoint is restored before testing.

## Evaluation caveats

- **These numbers are not comparable to the original GCR paper.** The protocols differ. Under this one, with 1 positive and 99 negatives, a random ranker already scores HR@10 = 0.10. The paper reports values below that, so it almost certainly ranked against more candidates. Nothing here should be read as matching or beating the paper.
- **Sampled metrics can rank models differently from full ranking.** Metrics computed over a small sample of negatives are not always consistent with metrics over the full catalog (Krichene & Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020). The comparisons above hold only under this protocol.
- **Model selection used the test set.** The Optuna objective, and the choice of the best run per model, use test NDCG@5, not validation NDCG@5. The reported numbers are therefore optimistic for every model. Within-protocol comparisons are less affected than absolute values.
- **The metric changes the ordering.** GCR and FA-GCR place most of their hits at rank 1, so their advantage is largest at small cutoffs. At HR@10, GCR roughly ties BiasedSVD on ML-1M (0.471 vs. 0.470), and FuzzGCR-product is slightly ahead of GCR on ML-100K (0.403 vs. 0.399).
- **Small validation set on ML-100K.** The 10% validation subset is only 94 users, so early stopping on ML-100K is noisy.

## How to run

### Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                # includes the dev group (scikit-learn, Jupyter), needed by the data scripts
```

Or with Docker (CUDA 12.2 base image; the entry point is `main.py`):

```bash
docker build -t gcr4rec .
docker run --gpus all -v "$PWD/data:/workspace/data" -v "$PWD/runs:/workspace/runs" gcr4rec <main.py arguments>
```

The image contains only `main.py` and `src/`, so the data has to be mounted.

### Data preparation

Optional, because `data/processed/` is already in the repo. To rebuild it, put the raw files under `data/raw/` and run these commands from the repo root (ML-100K shown):

```bash
D=data/processed/ml-100k
uv run python scripts/1_read_ml100k_ratings.py -i data/raw/ml-100k/u.data -o $D
uv run python scripts/2_leave_one_out_split.py -d $D
uv run python -m scripts.3_precompute_test_neg_samples -d $D        # 99 negatives, seed 42
uv run python scripts/4_read_ml100k_features.py \
  --user-data-path data/raw/ml-100k/u.user --item-data-path data/raw/ml-100k/u.item \
  --user-encoder-path $D/user_encoder.pkl --item-encoder-path $D/item_encoder.pkl \
  --output-path $D/features
```

For ML-1M, use `1_read_ml1m_ratings.py` (on `ratings.dat`) and `4_read_ml1m_features.py` (on `users.dat` and `movies.dat`). For Amazon, use `1_read_amazon_ratings.py` on the 2014 5-core review JSON (for example `reviews_Beauty_5.json`). Amazon has no step 4. Step 3 imports from `src/`, so it has to be run with `-m`.

### Training

`main.py` trains one configuration, restores the best checkpoint by validation NDCG@5, and evaluates it on the test set. Example with GCR on ML-100K:

```bash
D=data/processed/ml-100k
uv run python main.py --model GCR \
  --user-item-embedding-dim 64 --event-embedding-dim 64 --hidden-dim 64 \
  --num-neighbors 5 --reg-weight 1e-2 --dropout-rate 0.2 \
  --learning-rate 1e-3 --weight-decay 1e-5 --batch-size 64 \
  --all-interactions-csv $D/all_interactions.csv \
  --train-interactions-csv $D/split/train.csv \
  --validation-interactions-csv $D/split/val.csv \
  --test-interactions-csv $D/split/test_neg_samples.csv \
  --artifacts-saver Local --local-artifacts-path runs/artifacts \
  --tensorboard-log-dir runs/tensorboard
```

| `--model` | Model-specific flags (plus `--event-embedding-dim --hidden-dim --num-neighbors --dropout-rate` for all GCR variants) |
|---|---|
| `BiasedSVD` | `--embedding-dim` |
| `GCR` | `--user-item-embedding-dim --reg-weight` |
| `FuzzyProductGCR`, `FuzzyGodelGCR`, `FuzzyLukasiewiczGCR` | `--user-item-embedding-dim` |
| `FuzzySmoothGCR` | `--user-item-embedding-dim --stiffness` |
| `FeatureAwareWithIdEmbedGCR`, `FeatureAwareWithoutIdEmbedGCR` | `--reg-weight --categorical-embedding-dim --encoder-output-dim --user-feature-csv $D/features/users.csv --item-feature-csv $D/features/items.csv --feature-metadata-json $D/features/metadata.json` |

Each run writes `metrics.json`, `model_weights.pth`, `test_predictions.csv` and per-user metrics to `<local-artifacts-path>/<model-and-hparams>/`, and TensorBoard curves to `--tensorboard-log-dir`. Training uses CUDA, then MPS, then CPU, whichever is available first. The best checkpoint is written to a fixed path, `/tmp/best_model_checkpoint.pt`, so don't run several trainings at once on the same machine.

### Tuning

The tuning runs are the notebooks in `notebooks/Colab/`, one per model–dataset combination, all built from `BaseOptunaNotebook.ipynb`. Each notebook clones the repo, installs Optuna, reads data from Google Drive, and calls `main.py` with `--experiment-tracker Optuna`. It runs 20 TPE trials (seed 42) over categorical search spaces and stores the study in SQLite, so an interrupted study can be resumed. `ResultAnalysis.ipynb` picks the best run per model and computes the final HR/NDCG table. The notebooks expect Colab and Drive paths, so running them elsewhere means editing the path constants.

`hpo_jobs/`, the `VertexAI` experiment tracker, the `GoogleCloud` artifact saver and `scripts/build_and_push.sh` / `scripts/sync-data.sh` make up an alternative Vertex AI setup. **This code is untested and was not used for any result here.** It has at least one known argument mismatch: `hpo_jobs/vertex_ai_job.py` passes `--artifact-saver`, but `main.py` expects `--artifacts-saver`.

## Repository layout

```
main.py                     CLI entry point: builds model, data loaders, trainer
src/
  models/
    base_gcr.py             shared GCR machinery: neighbor sampling, event encoder, logic chain
    gcr.py                  GCR (neural NOT/OR + logic regularizers)
    fuzzy_gcr.py            FuzzGCR (product / Gödel / Łukasiewicz / smooth)
    feature_aware_gcr.py    FA-GCR, with and without ID embeddings
    biased_svd.py           baseline
    encoders/               ID lookup and feature-aware encoders
    logical_modules/        neural and fuzzy operators, logic regularizers
  trainer.py                training loop, early stopping, evaluation
  datasets/                 negative sampling (train/val), precomputed test negatives
  losses/  metrics/         BPR, NDCG@k
  loggers/  artifacts_savers/  experiment_trackers/   TensorBoard, local/GCS output, Optuna/Vertex hooks
scripts/                    data preparation (steps 1–4), GCP helper scripts
data/processed/             processed interactions, splits, test negatives, MovieLens features
notebooks/Colab/            Optuna tuning notebooks, result analysis, TensorBoard viewer
hpo_jobs/                   Vertex AI tuning jobs (untested)
Dockerfile  pyproject.toml  uv.lock
```

## Citation

This repository reimplements:

```bibtex
@inproceedings{chen2022graph,
  title     = {Graph Collaborative Reasoning},
  author    = {Chen, Hanxiong and Li, Yunqi and Shi, Shaoyun and Liu, Shuchang and Zhu, He and Zhang, Yongfeng},
  booktitle = {Proceedings of the Fifteenth ACM International Conference on Web Search and Data Mining (WSDM '22)},
  pages     = {75--84},
  year      = {2022},
  doi       = {10.1145/3488560.3498410}
}
```
