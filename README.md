# STRSM: Code and Data

Code, data, and results for the paper on herbal prescription recommendation with STRSM — a set-aware self-supervised heterogeneous graph network — evaluated on two public datasets:

- **PTM** (33,765 prescriptions, 390 symptoms, 811 herbs) — six-model comparison (Table 5 in the paper) and SSL ablation (Table 7)
- **TCM-SFT** (14,567 prescriptions, 3,830 symptoms, 566 herbs) — generalization on a dataset with a 10× larger symptom vocabulary (Table 6 in the paper)

## Repository layout

```
code/                  six model variants (PTM experiments)
code_tcm_sft/          STRSM model + dataset construction + feature pipelines (TCM-SFT)
data/                  PTM inputs (prescriptions, vocabularies, split indices, herb attributes)
data_TCM/              TCM-SFT inputs (prescriptions, vocabularies, split indices, features, Pharmacopoeia tables)
processing_data/       PTM co-occurrence graphs and symptom/herb features
results/               multi-seed results: main table, per-seed details, significance tests, raw logs
```

## Models (code/)

All six variants are re-trained under a unified configuration (200 epochs with cosine annealing, patience 50, dropout 0.1 in graph attention layers and 0.3 in the prediction layer, batch size 32) and evaluated over five random seeds (42–46).

| # | Variant | Script | Launch command |
|---|---|---|---|
| 1 | **STRSM (full)** | `strsm_full.py` | `SEEDS=42,43,44,45,46 python "code/strsm_full.py"` |
| 2 | ST-WORD | `strsm_word.py` | `SEEDS=42,43,44,45,46 python "code/strsm_word.py"` |
| 3 | ST-GAT | `strsm_gat.py` | `SEEDS=42,43,44,45,46 python "code/strsm_gat.py"` |
| 4 | ST-RGAT | `strsm_rgat.py` | `SEEDS=42,43,44,45,46 python "code/strsm_rgat.py"` |
| 5 | w/o Align | `strsm_w_o_align.py` | `SEEDS=42,43,44,45,46 python "code/strsm_w_o_align.py"` |
| 6 | w/o Recon | `strsm_w_o_recon.py` | `SEEDS=42,43,44,45,46 RECON_SCALE=0 python "code/strsm_w_o_recon.py"` |

Main results (F1@10, mean ± std over seeds 42–46): STRSM (full) **0.2637 ± 0.0013** > w/o Align 0.2629 ± 0.0008 > w/o Recon 0.2623 ± 0.0011 > ST-RGAT 0.2618 ± 0.0011 > ST-GAT 0.2590 ± 0.0019 > ST-WORD 0.2550 ± 0.0023.

## TCM-SFT experiments (code_tcm_sft/)

| File | Description |
|---|---|
| `strsm_tcm_sft.py` | **STRSM (full)** on TCM-SFT — main result **0.4459 ± 0.0022** (F1@10), vs. the strongest baseline TCMPR 0.4273 ± 0.0013 |
| `prepare_data_TCM.py` | Dataset construction: symptom/herb extraction from the structured clinical records, herb-name normalization, group-aware 6:2:2 split (seed 42) |
| `symptom_features_tcm.py` | Symptom features: two-stage Word2Vec (Skip-Gram pre-training on classical TCM texts → fine-tuning on training prescriptions only) |
| `herb_features_tcm23.py` | Herb features: 23-dimensional graded attribute vectors from the Pharmacopoeia database (894 herbs, four-step name matching) |

Launch: `SEEDS=42,43,44,45,46 python "code_tcm_sft/strsm_tcm_sft.py"` (defaults reproduce the paper configuration). The script reads `../data_TCM/` by relative path.

## Data

**data/** (PTM): `pre_symptoms.txt`, `pre_herbs.txt` (prescriptions; line number = sample id), `symptom_contains.txt`, `herbs_contains.txt` (vocabularies), `split_indices.npz` (group-aware 6:2:2 split, seed 42), `herbs_info.json` (herb attribute source), `external_corpus/` (Word2Vec pre-training corpus).

**processing_data/** (PTM): three co-occurrence graphs (`*_graph_train_622m.npy`, built from train+val only), `cold_nodes_622m.npz`, `symptom_features.npy`, `herb_features.npy`.

**data_TCM/** (TCM-SFT): `pre_symptoms.txt`, `pre_herbs.txt`, `split_indices.npz` (group-aware 6:2:2 = 8741/2913/2913, seed 42), `symptom_features_w2v.npy` (classically pre-trained 100-d symptom embeddings), `herb_attr23.npy` (23-d graded herb attributes), `herbs_info.json`, vocabularies.

**Note on Chinese literals:** the processing scripts contain Chinese string tables (section labels, herb-name alias mappings, and the graded property tables). These are lookup tables required to parse the Chinese Pharmacopoeia and the Chinese clinical source text, and are integral to the pipeline.

## Leakage prevention

The three co-occurrence graphs (symptom–symptom, symptom–herb, herb–herb) are constructed **exclusively from the training and validation prescriptions**; the test partition contributes no edges and is evaluated exactly once. The symptom Word2Vec embeddings are fine-tuned on training prescriptions only. A group-aware split guarantees that identical prescriptions never span subsets.

## Results

`results/table6_main.csv` — six variants × nine metrics (mean ± std over seeds 42–46); `table6_perseed.csv` — per-seed details; `table6_significance.csv` — paired t-tests (STRSM full vs. each variant); `results/raw/` — raw multi-seed logs.
