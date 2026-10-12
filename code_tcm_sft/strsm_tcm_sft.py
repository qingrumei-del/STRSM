#
#

import os
import csv
import math
import random
import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import roc_auc_score

# ============ Config ============
BINARY_GRAPH = os.environ.get("BINARY_GRAPH", "1") == "1"
GRAPH_MODE = os.environ.get("GRAPH_MODE", "raw_bin")
FREEZE_FEAT = os.environ.get("FREEZE_FEAT", "1") == "1"

def _maybe_binarize(adj: torch.Tensor) -> torch.Tensor:
    if not BINARY_GRAPH:
        return adj
    return (adj > 0).float()

if GRAPH_MODE == "selfloop":
    _base = "Ours-aug-selfloop"
elif GRAPH_MODE == "knn_bin":
    _base = "Ours-aug-bin"
else:
    _base = "Ours-aug-bin-raw"
DEFAULT_MODEL_NAME = _base + ("-frz" if FREEZE_FEAT else "")

class Config:
    num_symptoms = 3830
    num_herbs = 566
    symptom_feat_dim = 100
    herb_feat_dim = 23

    gat_out_dim = 100
    gat_num_heads = 4
    gat_dropout = 0.1

    rgat_layers = 2

    default_num_relations = 3

    st_num_inds = 32
    st_dim_hidden = 128
    st_num_heads = 4
    st_ln = True
    st_dropout = 0.1

    batch_size = 16
    learning_rate = 3e-4
    weight_decay = 1e-4
    num_epochs = 200
    train_ratio = 0.7
    dropout_rate = 0.3
    patience = 40
    pos_weight = 3.0
    sym_drop = 0.15

    # SSL weights
    ssl_weight = 0.01  # global multiplier for SSL loss (tune this)
    align_weight = 0.1  # weight for alignment component inside SSL
    recon_scale = 1.0  # scale for reconstruction terms

    symptom_feature_file = "../data_TCM/symptom_features_w2v.npy"
    herb_feature_file = "../data_TCM/herb_attr23.npy"
    pre_symptoms_file = "../data_TCM/pre_symptoms.txt"
    pre_herbs_file = "../data_TCM/pre_herbs.txt"
    split_file = "../data_TCM/split_indices.npz"          # group-aware 6:2:2 = 8741/2913/2913

# ============ Helpers ============
def normalize_adj(adj: torch.Tensor):
    adj = adj.clone().float()
    n = adj.size(0)
    adj = adj + torch.eye(n, device=adj.device)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D = torch.diag(deg_inv_sqrt)
    return D @ adj @ D

def scale_adj(adj: torch.Tensor):
    a = (adj.float() / adj.max().clamp(min=1.0)).clone()
    a.fill_diagonal_(1.0)
    return a

# ============ GAT / Relation GAT (same as before) ============
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, concat=True):
        super().__init__()
        assert out_dim % num_heads == 0
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * (out_dim // num_heads)))
        nn.init.xavier_uniform_(self.a)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.concat = concat

    def forward(self, h, adj_mask):
        N = h.size(0)
        Wh = self.W(h)
        Wh_heads = Wh.view(N, self.num_heads, self.head_dim)
        a_left = self.a[:, :self.head_dim]
        a_right = self.a[:, self.head_dim:]
        left = torch.einsum('nhd,hd->nh', Wh_heads, a_left)
        right = torch.einsum('nhd,hd->nh', Wh_heads, a_right)
        scores = left.unsqueeze(1) + right.unsqueeze(0)
        scores = self.leaky_relu(scores)
        mask = (adj_mask > 0).unsqueeze(-1)
        neg_inf_val = -1e4 if scores.dtype == torch.float16 else -1e9
        neg_inf = torch.tensor(neg_inf_val, device=scores.device, dtype=scores.dtype)
        scores = scores.masked_fill(~mask, neg_inf)
        scores = scores + torch.log1p(adj_mask).unsqueeze(-1)
        attn = torch.softmax(scores, dim=1)
        attn = self.dropout(attn)
        attn = attn.permute(2, 0, 1)
        Wh_heads = Wh_heads.permute(1, 0, 2)
        out_heads = torch.bmm(attn, Wh_heads)
        out_heads = out_heads.permute(1, 0, 2)
        out = out_heads.reshape(N, -1) if self.concat else out_heads.mean(dim=1)
        return out

class ResidualMultiGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        if in_dim != out_dim:
            self.input_proj = nn.Linear(in_dim, out_dim)
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.input_proj = nn.Identity()
            self.res_proj = nn.Identity()
        layers = [GATLayer(out_dim, out_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        self.layers = nn.ModuleList(layers)
        self.final_linear = nn.Linear(out_dim, out_dim)
        self.act = nn.Tanh()

    def forward(self, x, adj_norm):
        h = self.input_proj(x)
        h_orig = x
        for i, layer in enumerate(self.layers):
            h_in = h
            h = layer(h, adj_norm)
            if i == 0 and not isinstance(self.res_proj, nn.Identity):
                res = self.res_proj(h_orig)
            else:
                res = h_in
            h = self.act(self.final_linear(h + res))
        return h

class RelationGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, num_relations, dropout=0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_relations = num_relations
        self.head_dim = out_dim // num_heads

        self.W_shared = nn.Linear(in_dim, out_dim * 3, bias=False)

        self.rel_bias = nn.Parameter(torch.zeros(num_relations, 1, 1, num_heads))
        nn.init.normal_(self.rel_bias, mean=0.0, std=0.01)

        self.W_res = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.rel_importance = nn.Parameter(torch.ones(num_relations))
        self.temperature = 1.0

    def forward(self, x, adj_rel):
        single = False
        if x.dim() == 2:
            x = x.unsqueeze(0);
            single = True
        B, N, _ = x.shape

        projected = self.W_shared(x).view(B, N, 3, self.num_heads, self.head_dim)
        Q, K, V = projected[:, :, 0], projected[:, :, 1], projected[:, :, 2]

        outputs = []
        weights = []
        for r in range(min(adj_rel.size(0), self.rel_bias.size(0))):
            adj = adj_rel[r]
            if float(adj.sum().item()) == 0.0:
                continue

            scores = torch.einsum('bihd,bjhd->bijh', Q, K) / math.sqrt(self.head_dim)
            scores = scores + self.rel_bias[r]

            mask = adj.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, self.num_heads).bool()
            neg_inf_val = -1e4 if scores.dtype == torch.float16 else -1e9
            neg_inf = torch.tensor(neg_inf_val, device=scores.device, dtype=scores.dtype)
            scores = scores.masked_fill(~mask, neg_inf)
            scores = scores + torch.log1p(adj).unsqueeze(0).unsqueeze(-1)

            attn = F.softmax(scores, dim=2)
            attn = self.dropout(attn)

            out = torch.einsum('bijh,bjhd->bihd', attn, V)
            out = out.reshape(B, N, self.out_dim)
            row_has_edge = (adj.sum(dim=-1) > 0).to(out.dtype)  # [N]
            out = out * row_has_edge.unsqueeze(0).unsqueeze(-1)
            outputs.append(out)
            weights.append(self.rel_importance[r])

        if outputs:
            w_t = torch.stack([w for w in weights]).to(x.device)
            w_soft = F.softmax(w_t / max(1e-6, self.temperature), dim=0)
            combined = sum(w * o for w, o in zip(w_soft, outputs))
        else:
            combined = torch.zeros_like(x)

        residual = self.W_res(x)
        out = self.layer_norm(combined + residual)
        return self.activation(out) if not single else self.activation(out).squeeze(0)

class SH_RGAT(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, num_relations, num_layers=1, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(RelationGATLayer(in_dim if i == 0 else out_dim, out_dim, num_heads, num_relations, dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, adj_rel):
        h = x
        for layer in self.layers:
            h = layer(h, adj_rel)
        return h

# ============ Set Transformer components (MAB / ISAB / PMA / Encoder / Decoder) ============
class MAB(nn.Module):
    """
    Multihead Attention Block used in Set Transformer.
    - dim_Q: input dim for Q
    - dim_K: input dim for K (and V)
    - dim_V: output feature dimension (will be split across heads)
    - num_heads: number of attention heads
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=True, dropout=0.1):
        super().__init__()
        assert dim_V % num_heads == 0, "dim_V must be divisible by num_heads"
        self.num_heads = num_heads
        self.dim_V = dim_V
        d = dim_V // num_heads

        # linear projections
        self.fc_q = nn.Linear(dim_Q, dim_V)  # projects Q -> dim_V
        self.fc_k = nn.Linear(dim_K, dim_V)  # projects K -> dim_V
        self.fc_v = nn.Linear(dim_K, dim_V)  # projects V -> dim_V

        # output projection (after heads concatenation)
        self.fc_o = nn.Linear(dim_V, dim_V)

        self.dropout = nn.Dropout(dropout)
        self.ln0 = nn.LayerNorm(dim_V) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(dim_V) if ln else nn.Identity()

    def forward(self, Q, K, mask=None):
        """
        Q: [B, Nq, dim_Q]
        K: [B, Nk, dim_K]
        mask: [B, Nk] with 1 for valid tokens, 0 for padding (mask applied on Keys)
        """
        B, Nq, _ = Q.size()
        Nk = K.size(1)
        head_dim = self.dim_V // self.num_heads

        # 1) linear projections (keep the full-dim versions for residual)
        Q_proj_full = self.fc_q(Q)  # [B, Nq, dim_V]
        K_proj_full = self.fc_k(K)  # [B, Nk, dim_V]
        V_proj_full = self.fc_v(K)  # [B, Nk, dim_V]

        # 2) reshape for multi-head: [B, num_heads, N, head_dim]
        Qh = Q_proj_full.view(B, Nq, self.num_heads, head_dim).transpose(1, 2)  # [B, h, Nq, d]
        Kh = K_proj_full.view(B, Nk, self.num_heads, head_dim).transpose(1, 2)  # [B, h, Nk, d]
        Vh = V_proj_full.view(B, Nk, self.num_heads, head_dim).transpose(1, 2)  # [B, h, Nk, d]

        # 3) scaled dot-product attention
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (head_dim ** 0.5)  # [B, h, Nq, Nk]

        # 4) apply mask (mask over keys). mask: [B, Nk] -> expand -> [B, 1, 1, Nk]
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            mask_bool = (mask > 0)
            mask_len = mask_bool.size(1)
            if mask_len == Nk:
                mK = mask_bool.unsqueeze(1).unsqueeze(1)
            elif mask_len == Nq:
                mK = torch.ones((B, 1, 1, Nk), device=mask_bool.device, dtype=torch.bool)
            else:
                try:
                    mK = mask_bool.unsqueeze(1).unsqueeze(1).expand(-1, 1, 1, Nk)
                except Exception:
                    mK = torch.ones((B, 1, 1, Nk), device=mask_bool.device, dtype=torch.bool)
            neg_inf_val = -1e4 if scores.dtype == torch.float16 else -1e9
            neg_inf = torch.tensor(neg_inf_val, device=scores.device, dtype=scores.dtype)
            scores = scores.masked_fill(~mK, neg_inf)

        A = torch.softmax(scores, dim=-1)  # [B, h, Nq, Nk]
        A = self.dropout(A)

        # 5) attention output
        O = torch.matmul(A, Vh)  # [B, h, Nq, d]
        O = O.transpose(1, 2).contiguous().view(B, Nq, self.dim_V)  # [B, Nq, dim_V]

        # 6) residual + layernorm: use projected Q (Q_proj_full) to match dimension dim_V
        H = self.ln0(Q_proj_full + O)  # both are [B, Nq, dim_V]
        H = self.ln1(H + self.fc_o(F.relu(H)))  # feed-forward-ish block
        return H

class ISAB(nn.Module):
    """
    Induced Set Attention Block
    - dim_in: input feature dim of elements in X
    - dim_out: output dim for induced points / hidden dim
    - num_heads: attention heads
    - num_inds: number of inducing points (I)
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=True, dropout=0.1):
        super().__init__()
        # learnable inducing points I: shape [1, num_inds, dim_out]
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)

        # MABs:
        # mab0: Q from I (dim_out), K from X (dim_in), outputs dim_out
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln, dropout)
        # mab1: Q from X (dim_in), K from H (dim_out), outputs dim_out
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln, dropout)

    def forward(self, X, mask=None):
        """
        X: [B, L, dim_in]
        mask: [B, L]  (1 for valid, 0 for pad)
        """
        # 1) Induce: Q = I.repeat(B,1,1), K = X -> mask applies to Keys (X)
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask)  # H: [B, num_inds, dim_out]

        # 2) Attend: Q = X, K = H. H are inducing slots (no padding) -> no mask needed here
        return self.mab1(X, H, None)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, dropout=0.1):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln, dropout)

    def forward(self, X, mask=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, mask)

class SetTransformerEncoder(nn.Module):
    def __init__(self, dim_input, num_inds, dim_hidden, num_heads, ln=True, dropout=0.1):
        super().__init__()
        self.isab0 = ISAB(dim_input, dim_hidden, num_heads, num_inds, ln, dropout)
        self.isab1 = ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln, dropout)

    def forward(self, X, mask=None):
        h = self.isab0(X, mask)
        h = self.isab1(h, mask)
        return h

class SetTransformerDecoder(nn.Module):
    def __init__(self, dim_hidden, num_heads, num_outputs=1, ln=True, dropout=0.1):
        super().__init__()
        self.pma = PMA(dim_hidden, num_heads, num_outputs, ln, dropout)
        self.sab = MAB(dim_hidden, dim_hidden, dim_hidden, num_heads, ln, dropout)
        self.out_linear = nn.Linear(dim_hidden, dim_hidden)

    def forward(self, H, mask=None):
        out = self.pma(H, mask)
        out = self.sab(out, out, mask=None)
        out = self.out_linear(out)
        return out

# ============ Full Model with SSL (modified for double-layer SH and LayerNorm) ============
class RGAT_to_SetTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.gat_out_dim
        self.sym_in_proj = nn.Linear(cfg.symptom_feat_dim, D)
        self.sym_embedding = nn.Embedding(cfg.num_symptoms, D)
        nn.init.xavier_uniform_(self.sym_embedding.weight)
        self.herb_in_proj = nn.Linear(cfg.herb_feat_dim, D)
        self.herb_embedding = nn.Embedding(cfg.num_herbs, D)
        nn.init.xavier_uniform_(self.herb_embedding.weight)

        # Modules A & B: ResidualMultiGAT (they already use num_layers=2 by default)
        self.S_module = ResidualMultiGAT(D, D, num_layers=2, num_heads=cfg.gat_num_heads, dropout=cfg.gat_dropout)
        self.H_module = ResidualMultiGAT(D, D, num_layers=2, num_heads=cfg.gat_num_heads, dropout=cfg.gat_dropout)

        self.SH_module = SH_RGAT(D, D, cfg.gat_num_heads, cfg.default_num_relations,
                                 num_layers=cfg.rgat_layers, dropout=cfg.gat_dropout)

        self.gate_s = nn.Sequential(nn.Linear(D * 2, D // 2), nn.ReLU(), nn.Linear(D // 2, D))
        self.gate_h = nn.Sequential(nn.Linear(D * 2, D // 2), nn.ReLU(), nn.Linear(D // 2, D))

        self.layernorm_sh_s = nn.LayerNorm(D)
        self.layernorm_sh_h = nn.LayerNorm(D)
        self.layernorm_s = nn.LayerNorm(D)
        self.layernorm_h = nn.LayerNorm(D)

        # Set transformer encoder/decoder
        self.encoder = SetTransformerEncoder(dim_input=D, num_inds=cfg.st_num_inds,
                                             dim_hidden=cfg.st_dim_hidden, num_heads=cfg.st_num_heads,
                                             ln=cfg.st_ln, dropout=cfg.st_dropout)
        self.decoder = SetTransformerDecoder(dim_hidden=cfg.st_dim_hidden, num_heads=cfg.st_num_heads,
                                             num_outputs=1, ln=cfg.st_ln, dropout=cfg.st_dropout)
        self.dec_to_D = nn.Linear(cfg.st_dim_hidden, D)
        self.dropout = nn.Dropout(cfg.dropout_rate)
        self.sym_recon = nn.Linear(D, cfg.symptom_feat_dim)
        self.herb_recon = nn.Linear(D, cfg.herb_feat_dim)

        self.unk_sym = nn.Parameter(torch.zeros(D))
        self.unk_herb = nn.Parameter(torch.zeros(D))
        self.register_buffer("iso_sym_mask", torch.zeros(cfg.num_symptoms))
        self.register_buffer("iso_herb_mask", torch.zeros(cfg.num_herbs))

    def _build_relations(self, adj_ss, adj_hh, adj_sh):
        Ns = self.cfg.num_symptoms;
        Nh = self.cfg.num_herbs;
        device = adj_ss.device
        sh_mats = []
        if adj_sh is not None and adj_sh.ndim == 2 and adj_sh.shape[0] == Ns + Nh:
            full = adj_sh
            ss_block = full[:Ns, :Ns]
            hh_block = full[Ns:, Ns:]
            sh_block = full[:Ns, Ns:]
            ss_block = ss_block + torch.eye(Ns, device=device)
            hh_block = hh_block + torch.eye(Nh, device=device)
            ss_n = scale_adj(ss_block)
            hh_n = scale_adj(hh_block)
            sh_mats.append(scale_adj(sh_block))
        else:
            ss_n = scale_adj(adj_ss)
            hh_n = scale_adj(adj_hh)
            if adj_sh is None:
                sh_mats = []
            else:
                if adj_sh.ndim == 3:
                    for r in range(adj_sh.shape[0]):
                        sh_mats.append(scale_adj(adj_sh[r]))
                else:
                    sh_mats.append(scale_adj(adj_sh))
        R_total = 2 + len(sh_mats)
        N = Ns + Nh
        adj_rel = torch.zeros((R_total, N, N), device=device, dtype=torch.float32)
        adj_rel[0, :Ns, :Ns] = ss_n.to(device)
        adj_rel[1, Ns:, Ns:] = hh_n.to(device)
        for idx, sh in enumerate(sh_mats):
            r_idx = 2 + idx
            adj_rel[r_idx, :Ns, Ns:] = sh.to(device)
            adj_rel[r_idx, Ns:, :Ns] = sh.t().to(device)
        return adj_rel

    def forward(self, sym_feat, herb_feat, adj_ss, adj_hh, adj_sh, batch_sym_indices, mask):
        """
        sym_feat: [Ns, feat_s]
        herb_feat: [Nh, feat_h]
        adj_ss, adj_hh: adjacency (square)
        adj_sh: either bipartite or full; _build_relations handles cases
        batch_sym_indices: list of lists of symptom indices per sample
        mask: padding mask (from collate) -> used by SetTransformer
        """
        device = sym_feat.device
        Ns = self.cfg.num_symptoms
        Nh = self.cfg.num_herbs

        s_proj = (self.sym_in_proj(sym_feat) + self.sym_embedding.weight
                  + self.iso_sym_mask.unsqueeze(1) * self.unk_sym)
        h_proj = (self.herb_in_proj(herb_feat) + self.herb_embedding.weight
                  + self.iso_herb_mask.unsqueeze(1) * self.unk_herb)

        # 2) normalize adjacency
        ss_norm = scale_adj(adj_ss.to(device))
        hh_norm = scale_adj(adj_hh.to(device))

        # 3) Module A/B embeddings
        e_s1 = self.S_module(s_proj, ss_norm)  # [Ns, D]
        e_h1 = self.H_module(h_proj, hh_norm)  # [Nh, D]

        # 4) Build relation adjacency and run SH (Module C)
        adj_rel = self._build_relations(adj_ss, adj_hh, adj_sh)  # [R, N, N]
        # build full node features for SH: concat symptoms + herbs along node dim
        full_feat = torch.cat([s_proj, h_proj], dim=0)  # [Ns+Nh, D]
        e_full = self.SH_module(full_feat, adj_rel)  # [N, D] (SH is relation-aware)
        # split SH outputs back
        e_s2 = e_full[:Ns]
        e_h2 = e_full[Ns:]

        e_s2 = self.layernorm_sh_s(e_s2)
        e_h2 = self.layernorm_sh_h(e_h2)

        s_cat = torch.cat([e_s1, e_s2], dim=1)  # [Ns, 2D]
        h_cat = torch.cat([e_h1, e_h2], dim=1)  # [Nh, 2D]
        s_gate = torch.sigmoid(self.gate_s(s_cat))
        h_gate = torch.sigmoid(self.gate_h(h_cat))
        e_s_final = s_gate * e_s1 + (1 - s_gate) * e_s2
        e_h_final = h_gate * e_h1 + (1 - h_gate) * e_h2

        e_s_final = self.layernorm_s(e_s_final)
        e_h_final = self.layernorm_h(e_h_final)

        e_s_final = self.dropout(e_s_final)
        e_h_final = self.dropout(e_h_final)

        batch = [e_s_final[idx] for idx in batch_sym_indices]  # list of [len_i, D]
        padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)  # [B, Lmax, D]

        # mask is provided from collate_fn: shape [B, Lmax] with 1 for real, 0 for pad
        # 9) SetTransformer encoder/decoder
        H = self.encoder(padded, mask)  # [B, Lmax, hidden]
        out = self.decoder(H, mask)  # [B, num_outputs, hidden]
        out = out.squeeze(1)  # [B, hidden]
        out = self.dec_to_D(out)  # [B, D]
        # compute similarity logits against all herbs: simple dot product
        logits = torch.matmul(out, e_h_final.t())  # [B, Nh]
        sym_rec = self.sym_recon(e_s_final)
        herb_rec = self.herb_recon(e_h_final)

        return logits, e_s_final, e_h_final, e_s1, e_s2, e_h1, e_h2, sym_rec, herb_rec

# ============ Dataset, metrics, loops (kept consistent with original) ============
class PrescriptionDataset(Dataset):
    def __init__(self, config: Config):
        self.cfg = config
        self.sym_idx = self.load_idx(config.pre_symptoms_file)
        self.herb_idx = self.load_idx(config.pre_herbs_file)
        # load features (npy)
        if os.path.exists(config.symptom_feature_file):
            self.sym_feat = torch.from_numpy(np.load(config.symptom_feature_file)).float()
        else:
            self.sym_feat = torch.zeros((config.num_symptoms, config.symptom_feat_dim)).float()
        if os.path.exists(config.herb_feature_file):
            self.herb_feat = torch.from_numpy(np.load(config.herb_feature_file)).float()
        else:
            self.herb_feat = torch.zeros((config.num_herbs, config.herb_feat_dim)).float()

        Ns, Nh = config.num_symptoms, config.num_herbs
        sp = np.load(config.split_file)
        tv_rows = sorted(set(sp['train_idx'].tolist()) | set(sp['val_idx'].tolist()))
        ss = self._count_graph([self.sym_idx[r] for r in tv_rows], Ns)
        hh = self._count_graph([self.herb_idx[r] for r in tv_rows], Nh)
        sh = np.zeros((Ns + Nh, Ns + Nh), dtype=np.float32)
        for row in sp['train_idx'].tolist():
            s = np.asarray([i for i in self.sym_idx[row] if 0 <= i < Ns], dtype=np.int64)
            h = np.asarray([i for i in self.herb_idx[row] if 0 <= i < Nh], dtype=np.int64) + Ns
            if s.size and h.size:
                sh[np.ix_(s, h)] += 1.0
                sh[np.ix_(h, s)] += 1.0
        self.iso_sym = (ss.sum(axis=1) == 0).astype(np.float32)
        self.iso_herb = (hh.sum(axis=1) == 0).astype(np.float32)
        self.adj_ss = _maybe_binarize(torch.from_numpy(ss).float())
        self.adj_hh = _maybe_binarize(torch.from_numpy(hh).float())
        self.adj_sh = _maybe_binarize(torch.from_numpy(sh).float())

        if GRAPH_MODE == "selfloop":
            self.adj_ss = torch.eye(config.num_symptoms)
            self.adj_hh = torch.eye(config.num_herbs)
            self.adj_sh = torch.eye(config.num_symptoms + config.num_herbs)

        self.targets = [self.build_target(h, config.num_herbs) for h in self.herb_idx]

    @staticmethod
    def _count_graph(sequences, n_nodes):
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for seq in sequences:
            idx = np.asarray(sorted({i for i in seq if 0 <= i < n_nodes}), dtype=np.int64)
            if idx.size < 2:
                continue
            adj[np.ix_(idx, idx)] += 1.0
        np.fill_diagonal(adj, 0.0)
        return adj

    def load_idx(self, f):
        if not os.path.exists(f):
            return [[]]
        return [list(map(int, line.strip().split())) if line.strip() else []
                for line in open(f, encoding='utf-8')]

    def load_graph(self, fpath, n_nodes):
        if not os.path.exists(fpath):
            return torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
        data = np.load(fpath, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape == (n_nodes, n_nodes):
            return _maybe_binarize(torch.from_numpy(data).float())
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] == 2:
            adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
            rows, cols = data[0], data[1]
            adj[rows, cols] = 1
            adj = torch.max(adj, adj.t())
            return adj
        # fallback
        return torch.zeros((n_nodes, n_nodes), dtype=torch.float32)

    def build_target(self, herbs, n):
        t = torch.zeros(n, dtype=torch.float32)
        for h in herbs:
            if 0 <= h < n:
                t[h] = 1.0
        return t

    def __len__(self):
        return len(self.sym_idx)

    def __getitem__(self, idx):
        sym_indices = self.sym_idx[idx]
        sym_features = self.sym_feat[sym_indices]
        mask = torch.ones(len(sym_indices))
        return sym_features, self.targets[idx], mask, len(sym_indices), sym_indices

def collate_fn(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    syms, targets, masks, lens, idxs = zip(*batch)
    # pad sequences
    syms_p = torch.nn.utils.rnn.pad_sequence(syms, batch_first=True)
    masks_p = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)
    targets = torch.stack(targets)
    # mask: 1 for valid, 0 for pad
    return list(idxs), targets, masks_p, syms_p.shape[1], masks_p

def precision_recall_f1_at_k(probs: torch.Tensor, targets: torch.Tensor, K_list=[5, 10, 20]):
    B, C = probs.shape
    res = {}

    trues = (targets > 0.5).bool()
    total_true = trues.sum().item()

    for K in K_list:
        _, topk_indices = torch.topk(probs, K, dim=1)
        pred_mask = torch.zeros(B, C, dtype=torch.bool, device=probs.device)
        pred_mask.scatter_(1, topk_indices, True)

        correct_matrix = pred_mask & trues
        TP = correct_matrix.sum().item()

        total_pred = B * K
        P_micro = TP / total_pred if total_pred > 0 else 0.0
        R_micro = TP / total_true if total_true > 0 else 0.0
        F1_micro = 2 * P_micro * R_micro / (P_micro + R_micro + 1e-12) if (P_micro + R_micro) > 0 else 0.0

        res[f"P@{K}"] = float(P_micro)
        res[f"R@{K}"] = float(R_micro)
        res[f"F1@{K}"] = float(F1_micro)

    return res

def train_one_epoch(model, loader, opt, scaler, device, cfg, sym_feat, herb_feat, adj_ss, adj_hh, adj_sh, criterion):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch_idxs, targets, masks_p, Lmax, masks in loader:
        # batch_idxs: tuple/list of index lists, but collate returned list of idxs
        # we will pass the list of index-lists to model forward as batch_sym_indices
        if getattr(cfg, 'sym_drop', 0) > 0:
            batch_idxs = [[s for s in idxs if random.random() >= cfg.sym_drop] or idxs
                          for idxs in batch_idxs]
            new_lens = [len(x) for x in batch_idxs]
            new_Lmax = max(new_lens)
            masks_p = torch.zeros(len(batch_idxs), new_Lmax)
            for i, L in enumerate(new_lens):
                masks_p[i, :L] = 1.0
        targets = targets.to(device)
        opt.zero_grad()
        with autocast():
            logits, e_s_final, e_h_final, e_s1, e_s2, e_h1, e_h2, sym_rec, herb_rec = model(
                sym_feat.to(device), herb_feat.to(device),
                adj_ss.to(device), adj_hh.to(device), adj_sh.to(device),
                batch_idxs, masks.to(device)
            )

            bce_loss = criterion(logits, targets.to(device))

            s_align = (1.0 - F.cosine_similarity(e_s1, e_s2, dim=1)).mean()
            h_align = (1.0 - F.cosine_similarity(e_h1, e_h2, dim=1)).mean()
            align_loss = 0.5 * (s_align + h_align)

            recon_s = F.mse_loss(sym_rec, sym_feat.to(device))
            recon_h = F.mse_loss(herb_rec, herb_feat.to(device))
            recon_loss = 0.5 * (recon_s + recon_h)

            ssl_comp = cfg.align_weight * align_loss + cfg.recon_scale * recon_loss
            loss = bce_loss + cfg.ssl_weight * ssl_comp
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)

def evaluate(model, loader, device, cfg, sym_feat, herb_feat, adj_ss, adj_hh, adj_sh, criterion, iso_sym=None):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    K_list = [5, 10, 20]

    total_true = 0
    stats = {
        K: {"TP": 0.0, "total_pred": 0.0}
        for K in K_list
    }
    cold_stats = {K: {"TP": 0.0, "total_pred": 0.0} for K in K_list}
    cold_total_true = 0
    n_cold = 0

    with torch.no_grad():
        for batch_idxs, targets, masks_p, Lmax, masks in loader:
            targets = targets.to(device)
            logits, *_, = model(
                sym_feat.to(device), herb_feat.to(device),
                adj_ss.to(device), adj_hh.to(device), adj_sh.to(device),
                batch_idxs, masks.to(device)
            )

            loss = criterion(logits, targets)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits)  # [B, C]
            B, C = probs.shape

            trues = (targets > 0.5).bool()
            total_true += trues.sum().item()

            cold_mask = torch.zeros(B, dtype=torch.bool, device=device)
            if iso_sym is not None:
                for bi, idxs in enumerate(batch_idxs):
                    if idxs and bool(iso_sym[idxs].any()):
                        cold_mask[bi] = True
            n_cold += int(cold_mask.sum().item())
            cold_total_true += int(trues[cold_mask].sum().item())

            for K in K_list:
                _, topk_indices = torch.topk(probs, K, dim=1)
                pred_mask = torch.zeros(B, C, dtype=torch.bool, device=probs.device)
                pred_mask.scatter_(1, topk_indices, True)

                # True Positive
                correct_matrix = pred_mask & trues
                TP = correct_matrix.sum().item()

                stats[K]["TP"] += TP
                stats[K]["total_pred"] += B * K

                if cold_mask.any():
                    cold_stats[K]["TP"] += correct_matrix[cold_mask].sum().item()
                    cold_stats[K]["total_pred"] += int(cold_mask.sum().item()) * K

    avg_loss = total_loss / max(1, n_batches)

    metrics = {}
    for K in K_list:
        TP = stats[K]["TP"]
        total_pred = stats[K]["total_pred"]

        P = TP / total_pred if total_pred > 0 else 0.0
        R = TP / total_true if total_true > 0 else 0.0
        F1 = 2 * P * R / (P + R + 1e-12) if (P + R) > 0 else 0.0

        metrics[f"P@{K}"] = float(P)
        metrics[f"R@{K}"] = float(R)
        metrics[f"F1@{K}"] = float(F1)

    metrics["n_cold"] = float(n_cold)
    for K in K_list:
        TP = cold_stats[K]["TP"]
        total_pred = cold_stats[K]["total_pred"]

        cP = TP / total_pred if total_pred > 0 else 0.0
        cR = TP / cold_total_true if cold_total_true > 0 else 0.0
        cF1 = 2 * cP * cR / (cP + cR + 1e-12) if (cP + cR) > 0 else 0.0

        metrics[f"cold_P@{K}"] = float(cP)
        metrics[f"cold_R@{K}"] = float(cR)
        metrics[f"cold_F1@{K}"] = float(cF1)

    return avg_loss, metrics

# ============ Main ============
def main(seed=None):
    if seed is None:
        seed = int(os.environ.get("SEED", 42))
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    cfg = Config()
    cfg.pos_weight = float(os.environ.get("POS_WEIGHT", cfg.pos_weight))
    cfg.sym_drop = float(os.environ.get("SYM_DROP", cfg.sym_drop))
    cfg.dropout_rate = float(os.environ.get("DROPOUT", cfg.dropout_rate))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = PrescriptionDataset(cfg)
    sp = np.load(cfg.split_file)
    merged_idx = np.concatenate([sp['train_idx'], sp['val_idx']])
    glabels = sp['group_labels'][merged_idx]
    unique_g = np.unique(glabels)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_g))
    n_val_g = int(len(unique_g) * 0.1)
    val_groups = set(unique_g[perm[:n_val_g]].tolist())
    is_val = np.array([g in val_groups for g in glabels])
    tr_ds = torch.utils.data.Subset(ds, merged_idx[~is_val].tolist())
    va_ds = torch.utils.data.Subset(ds, merged_idx[is_val].tolist())
    te_ds = torch.utils.data.Subset(ds, sp['test_idx'].tolist())
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4,
                           pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4,
                           pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4,
                           pin_memory=True)

    sym_feat = ds.sym_feat.to(device);
    herb_feat = ds.herb_feat.to(device)
    adj_ss = ds.adj_ss.to(device);
    adj_hh = ds.adj_hh.to(device);
    adj_sh = ds.adj_sh.to(device)

    model = RGAT_to_SetTransformer(cfg).to(device)
    if FREEZE_FEAT:
        model.sym_embedding.weight.requires_grad_(False)
        model.herb_embedding.weight.requires_grad_(False)
        n_frozen = model.sym_embedding.weight.numel() + model.herb_embedding.weight.numel()
        n_total = sum(p.numel() for p in model.parameters())
    model.iso_sym_mask.copy_(torch.from_numpy(ds.iso_sym))
    model.iso_herb_mask.copy_(torch.from_numpy(ds.iso_herb))
    iso_sym = model.iso_sym_mask > 0.5

    total_params = sum(p.numel() for p in model.parameters());
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    pos_weight = torch.tensor(cfg.pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.num_epochs)
    scaler = GradScaler()
    early = 0;
    best_f1 = 0.0;
    _safe_tag = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME).replace("/", "_").replace(" ", "_")
    best_path = f"best_rgat_v2m_{_safe_tag}_s{seed}_pw{cfg.pos_weight:g}.pth"

    for epoch in range(cfg.num_epochs):
        tr_loss = train_one_epoch(model, tr_loader, opt, scaler, device, cfg, sym_feat, herb_feat, adj_ss, adj_hh,
                                  adj_sh, criterion)
        va_loss, metrics = evaluate(model, va_loader, device, cfg, sym_feat, herb_feat, adj_ss, adj_hh, adj_sh,
                                    criterion, iso_sym=iso_sym)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{cfg.num_epochs}:")
        for k, v in metrics.items():
            if k == "n_cold" or k.startswith("cold_"):
                continue
            print(f"  {k}: {v:.4f}", end="  ")
        print()
        f1_10 = metrics.get("F1@10", 0.0)
        if f1_10 > best_f1 + 1e-6:
            best_f1 = f1_10
            torch.save(model.state_dict(), best_path)
            early = 0
        else:
            early += 1
            if early >= cfg.patience:
                print("Early stopping triggered.")
                break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    final_loss, final_metrics = evaluate(model, te_loader, device, cfg, sym_feat, herb_feat, adj_ss, adj_hh, adj_sh,
                                         criterion, iso_sym=iso_sym)
    final_metrics = {k: v for k, v in final_metrics.items() if not (k == "n_cold" or k.startswith("cold_"))}
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs("results_v2m_tcm_merged", exist_ok=True)
    out_csv = os.path.join("results_v2m_tcm_merged", f"v2m_aug23_seed{seed}.csv")
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['model', 'seed', 'F1@10', 'F1@20', 'F1@5',
                    'P@10', 'P@20', 'P@5', 'R@10', 'R@20', 'R@5'])
        w.writerow(['v2m-aug', seed,
                    f'{final_metrics["F1@10"]:.6f}', f'{final_metrics["F1@20"]:.6f}',
                    f'{final_metrics["F1@5"]:.6f}',
                    f'{final_metrics["P@10"]:.6f}', f'{final_metrics["P@20"]:.6f}',
                    f'{final_metrics["P@5"]:.6f}',
                    f'{final_metrics["R@10"]:.6f}', f'{final_metrics["R@20"]:.6f}',
                    f'{final_metrics["R@5"]:.6f}'])
    return final_metrics

if __name__ == "__main__":
    seeds = [int(s) for s in os.environ.get("SEEDS", "42,43,44,45,46").split(",") if s.strip()]
    all_metrics = []
    for s in seeds:
        print(f"\n========== SEED = {s} ==========")
        all_metrics.append((s, main(seed=s)))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(all_metrics) > 1:
        for k in sorted(all_metrics[0][1].keys()):
            vals = np.array([m[k] for _, m in all_metrics], dtype=np.float64)
            print(f"  {k}: {vals.mean():.4f} ± {vals.std(ddof=1):.4f}")
