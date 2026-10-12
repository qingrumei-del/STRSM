# rgat_to_settransformer_full.py
# Author: generated for user
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

class Config:
    # data sizes - change if needed
    num_symptoms = 390
    num_herbs = 811

    # feature dims (from your provided files)
    symptom_feat_dim = 100
    herb_feat_dim = 23

    # GAT settings (these follow the same formula used in SetTransformer-GAT baseline)
    gat_out_dim = 100
    gat_num_heads = 4
    gat_dropout = 0.1

    # R-GAT relations
    default_num_relations = 3

    # SetTransformer settings (for symptom-set aggregation)
    st_num_inds = 32
    st_dim_hidden = 128
    st_num_heads = 4
    st_ln = True
    st_dropout = 0.1
    st_num_outputs = 1  # produce a pooled vector per sample (we will use PMA with 1 seed)

    # training
    batch_size = 32
    learning_rate = 3e-4
    weight_decay = 1e-4
    num_epochs = 200
    train_ratio = 0.7
    dropout_rate = 0.3
    sym_drop = 0.15
    pos_weight = 3.0
    patience = 50

    # files (change these paths to where your files actually are)
    symptom_feature_file = "../processing_data/symptom_features.npy"
    herb_feature_file = "../processing_data/herb_features.npy"
    ss_graph_file = "../processing_data/symptom_symptom_graph_train_622m.npy"
    hh_graph_file = "../processing_data/herb_herb_graph_train_622m.npy"
    sh_graph_file = "../processing_data/symptom_herb_graph_train_622m.npy"
    pre_symptoms_file = "../data/pre_symptoms.txt"
    pre_herbs_file = "../data/pre_herbs.txt"
    split_file = "../data/split_indices.npz"
    cold_nodes_file = "../processing_data/cold_nodes_622m.npz"

def normalize_adj(adj: torch.Tensor):
    # symmetric normalization with self-loop
    adj = adj.clone().float()
    n = adj.size(0)
    if n == 0:
        return adj
    adj = adj + torch.eye(n, device=adj.device)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt

def scale_adj(adj: torch.Tensor):
    a = (adj.float() / adj.max().clamp(min=1.0)).clone()
    a.fill_diagonal_(1.0)
    return a

# ==================== GAT Layer (matching SetTransformer-GAT style) ====================
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, concat=True):
        super().__init__()
        assert out_dim % num_heads == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.concat = concat

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # attention vector per head: two parts
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor):
        # h: [N, in_dim]
        N = h.size(0)
        Wh = self.W(h)  # [N, out_dim]
        Wh_heads = Wh.view(N, self.num_heads, self.head_dim)  # [N, heads, head_dim]

        a_left = self.a[:, :self.head_dim]   # [heads, head_dim]
        a_right = self.a[:, self.head_dim:]  # [heads, head_dim]

        # compute left/right terms
        left = torch.einsum('nhd,hd->nh', Wh_heads, a_left)   # [N, heads]
        right = torch.einsum('nhd,hd->nh', Wh_heads, a_right) # [N, heads]

        # scores[b_i, b_j, head]
        scores = left.unsqueeze(1) + right.unsqueeze(0)  # [N, N, heads]
        scores = self.leaky_relu(scores)

        # mask: adj_mask: [N,N] boolean-ish
        mask = (adj_mask > 0).unsqueeze(-1)  # [N, N, 1]
        # choose neg inf based on dtype to avoid float16 overflow
        if scores.dtype == torch.float16:
            neg_inf_val = -1e4
        else:
            neg_inf_val = -1e9
        neg_inf = torch.tensor(neg_inf_val, device=scores.device, dtype=scores.dtype)
        scores = scores.masked_fill(~mask, neg_inf)
        scores = scores + torch.log1p(adj_mask).unsqueeze(-1)

        # softmax over neighbor dimension
        attn = torch.softmax(scores, dim=1)  # [N, N, heads]
        attn = self.dropout(attn)

        # prepare for bmm
        attn = attn.permute(2, 0, 1)          # [heads, N, N]
        Wh_heads_t = Wh_heads.permute(1, 0, 2)  # [heads, N, head_dim]

        out_heads = torch.bmm(attn, Wh_heads_t)  # [heads, N, head_dim]
        out_heads = out_heads.permute(1, 0, 2)   # [N, heads, head_dim]

        if self.concat:
            out = out_heads.reshape(N, self.out_dim)  # [N, out_dim]
        else:
            out = out_heads.mean(dim=1)  # [N, head_dim]

        return out

# ==================== ResidualMultiGAT (stacking GATLayer) ====================
class ResidualMultiGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if in_dim != out_dim:
            self.input_proj = nn.Linear(in_dim, out_dim)
            self.res_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.input_proj = nn.Identity()
            self.res_proj = nn.Identity()

        layers = []
        for i in range(num_layers):
            layers.append(GATLayer(out_dim, out_dim, num_heads=num_heads, dropout=dropout, concat=True))
        self.layers = nn.ModuleList(layers)
        self.final_linear = nn.Linear(out_dim, out_dim)
        self.act = nn.Tanh()

    def forward(self, x, adj_norm):
        h = self.input_proj(x)
        h_orig = x
        for i, layer in enumerate(self.layers):
            h_in = h
            h = layer(h, adj_norm)
            if i == 0 and self.in_dim != self.out_dim:
                res = self.res_proj(h_orig)
            else:
                res = h_in
            h = self.act(self.final_linear(h + res))
        return h

# ==================== RelationGATLayer (for R-GAT on SH graph) ====================
class RelationGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, num_relations, dropout=0.1):
        super().__init__()
        assert out_dim % num_heads == 0
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_relations = num_relations
        self.head_dim = out_dim // num_heads

        # shared linear to produce Q,K,V per head
        self.W_shared = nn.Linear(in_dim, out_dim * 3, bias=False)
        self.rel_bias = nn.Parameter(torch.zeros(num_relations, 1, 1, num_heads))
        nn.init.normal_(self.rel_bias, mean=0.0, std=0.01)

        self.W_res = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        # learned relation importance
        self.rel_importance = nn.Parameter(torch.ones(num_relations))
        self.temperature = 1.0

    def forward(self, x, adj_rel):
        # x: [N, in_dim] or [B, N, in_dim]
        single_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            single_batch = True
        B, N, _ = x.shape

        projected = self.W_shared(x).view(B, N, 3, self.num_heads, self.head_dim)
        Q, K, V = projected[:, :, 0], projected[:, :, 1], projected[:, :, 2]  # [B,N,heads,head_dim] each

        R = min(self.num_relations, adj_rel.size(0))
        outputs = []
        weights = []
        for r in range(R):
            adj = adj_rel[r]  # [N, N]
            if float(adj.sum().item()) == 0.0:
                continue
            # scores: [B, N, N, heads]
            scores = torch.einsum('bihd,bjhd->bijh', Q, K) / math.sqrt(self.head_dim)
            scores = scores + self.rel_bias[r]  # add small learned bias
            mask = adj.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, self.num_heads).bool()
            # large negative
            if scores.dtype == torch.float16:
                neg_inf_val = -1e4
            else:
                neg_inf_val = -1e9
            scores = scores.masked_fill(~mask, neg_inf_val)
            scores = scores + torch.log1p(adj).unsqueeze(0).unsqueeze(-1)
            attn = F.softmax(scores, dim=2)
            attn = self.dropout(attn)
            out = torch.einsum('bijh,bjhd->bihd', attn, V)  # [B,N,heads,head_dim]
            out = out.reshape(B, N, self.out_dim)
            outputs.append(out)
            weights.append(self.rel_importance[r])

        if outputs:
            weights_tensor = torch.stack([w if isinstance(w, torch.Tensor) else torch.tensor(w, device=x.device) for w in weights])
            weights_soft = F.softmax(weights_tensor / max(1e-6, self.temperature), dim=0)
            combined = sum(w * o for w, o in zip(weights_soft, outputs))
        else:
            combined = torch.zeros_like(x)

        residual = self.W_res(x)
        out = self.layer_norm(combined + residual)
        out = self.activation(out)
        if single_batch:
            out = out.squeeze(0)
        return out

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

# ==================== SetTransformer components (ISAB, PMA) - used for symptom set aggregation =========
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads=4, ln=True, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_V = dim_V

        # linear projections
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)

        # output projection
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.dropout = nn.Dropout(dropout)

        # LayerNorms (optional)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        else:
            self.ln0 = nn.Identity()
            self.ln1 = nn.Identity()

        # if dim_Q != dim_V, project Q for residual connection
        if dim_Q != dim_V:
            self.res_q = nn.Linear(dim_Q, dim_V)
        else:
            self.res_q = nn.Identity()

    def forward(self, Q, K, mask=None):
        """
        Q: [B, nq, dim_Q]
        K: [B, nk, dim_K]
        mask: [B, L] where L==nk (preferred) or L==nq
        returns: [B, nq, dim_V]
        """
        B, nq, _ = Q.size()
        _, nk, _ = K.size()

        # project to multihead space
        Q_proj = self.fc_q(Q).view(B, nq, self.num_heads, -1).transpose(1, 2)  # [B, H, nq, d]
        K_proj = self.fc_k(K).view(B, nk, self.num_heads, -1).transpose(1, 2)  # [B, H, nk, d]
        V_proj = self.fc_v(K).view(B, nk, self.num_heads, -1).transpose(1, 2)  # [B, H, nk, d]

        # attention scores
        scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / math.sqrt(Q_proj.size(-1))  # [B, H, nq, nk]

        # ---- robust mask handling ----
        if mask is not None:
            # ensure mask shape [B, L]
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            mask_bool = (mask > 0)
            mask_len = mask_bool.size(1)

            # Prefer mask that corresponds to keys (nk)
            if mask_len == nk:
                mK = mask_bool.unsqueeze(1).unsqueeze(1)  # [B,1,1,nk]
            elif mask_len == nq:
                # mask corresponds to queries: do NOT mask keys (safe fallback)
                mK = torch.ones((B, 1, 1, nk), device=mask_bool.device, dtype=torch.bool)
            else:
                # try to broadcast, otherwise fallback to no mask
                try:
                    mK = mask_bool.unsqueeze(1).unsqueeze(1).expand(-1, 1, 1, nk)
                except Exception:
                    mK = torch.ones((B, 1, 1, nk), device=mask_bool.device, dtype=torch.bool)

            # dtype-safe neg inf
            neg_inf_val = -1e4 if scores.dtype == torch.float16 else -1e9
            neg_inf = torch.tensor(neg_inf_val, device=scores.device, dtype=scores.dtype)
            scores = scores.masked_fill(~mK, neg_inf)

        # attention weights
        A = torch.softmax(scores, dim=-1)
        A = self.dropout(A)

        # weighted sum
        O = torch.matmul(A, V_proj)  # [B, H, nq, d]
        O = O.transpose(1, 2).contiguous().view(B, nq, -1)  # [B, nq, dim_V]

        # residual: project Q to dim_V if needed
        Q_res = self.res_q(Q)  # [B, nq, dim_V]

        # layernorm + feedforward residual
        H = self.ln0(O + Q_res)
        H = self.ln1(H + F.relu(self.fc_o(H)))
        return H

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, dropout=0.1):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln, dropout)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln, dropout)
    def forward(self, X, mask=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask)
        return self.mab1(X, H, mask)

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
        # we can stack another ISAB if desired; baseline often uses two
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
        self.out_linear = nn.Linear(dim_hidden, dim_hidden)  # project back to embedding dim if needed
    def forward(self, H, mask=None):
        out = self.pma(H, mask)  # [batch, num_outputs, dim_hidden]
        out = self.sab(out, out, mask=None)  # refine
        out = self.out_linear(out)  # [batch, num_outputs, dim_hidden]
        return out  # caller will squeeze num_outputs dim if needed

class RGAT_to_SetTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        D = cfg.gat_out_dim
        # input projections
        self.sym_in_proj = nn.Linear(cfg.symptom_feat_dim, D)
        self.herb_in_proj = nn.Linear(cfg.herb_feat_dim, D)
        self.sym_embedding = nn.Embedding(cfg.num_symptoms, D)
        nn.init.xavier_uniform_(self.sym_embedding.weight)
        self.herb_embedding = nn.Embedding(cfg.num_herbs, D)
        nn.init.xavier_uniform_(self.herb_embedding.weight)
        # S-SMRGAT and H-SMRGAT (using same GAT formula as SetTransformer-GAT)
        self.S_module = ResidualMultiGAT(D, D, num_layers=2, num_heads=cfg.gat_num_heads, dropout=cfg.gat_dropout)
        self.H_module = ResidualMultiGAT(D, D, num_layers=2, num_heads=cfg.gat_num_heads, dropout=cfg.gat_dropout)
        # Relation-aware GAT for SH graph (R-GAT)
        self.SH_module = SH_RGAT(D, D, cfg.gat_num_heads, cfg.default_num_relations, num_layers=1, dropout=cfg.gat_dropout)
        # gate fusion MLPs (small)
        self.gate_s = nn.Sequential(nn.Linear(D*2, D//2), nn.ReLU(), nn.Linear(D//2, D),)
        self.gate_h = nn.Sequential(nn.Linear(D*2, D//2), nn.ReLU(), nn.Linear(D//2, D),)
        self.layernorm_s = nn.LayerNorm(D)
        self.layernorm_h = nn.LayerNorm(D)
        self.layernorm_sh_s = nn.LayerNorm(D)
        self.layernorm_sh_h = nn.LayerNorm(D)
        # set transformer for symptom set -> we will produce a pooled vector per sample
        self.encoder = SetTransformerEncoder(dim_input=D, num_inds=cfg.st_num_inds, dim_hidden=cfg.st_dim_hidden, num_heads=cfg.st_num_heads, ln=cfg.st_ln, dropout=cfg.st_dropout)
        self.decoder = SetTransformerDecoder(dim_hidden=cfg.st_dim_hidden, num_heads=cfg.st_num_heads, num_outputs=1, ln=cfg.st_ln, dropout=cfg.st_dropout)
        # final linear to map decoder output dim to embedding dim (if decoder out != D)
        self.dec_to_D = nn.Linear(cfg.st_dim_hidden, D)
        self.sym_recon = nn.Linear(D, cfg.symptom_feat_dim)
        self.herb_recon = nn.Linear(D, cfg.herb_feat_dim)
        # herb projection (project herb embeddings to same D space; here herb embeddings already D)
        self.herb_proj = nn.Identity()
        # dropout
        self.dropout = nn.Dropout(cfg.dropout_rate)
        self.unk_sym = nn.Parameter(torch.zeros(D))
        self.unk_herb = nn.Parameter(torch.zeros(D))
        self.register_buffer("iso_sym_mask", torch.zeros(cfg.num_symptoms))
        self.register_buffer("iso_herb_mask", torch.zeros(cfg.num_herbs))

    def _build_relations(self, adj_ss, adj_hh, adj_sh):
        # Build relations tensor as in prior code:
        # adj_sh may be either full N x N matrix or R x N1 x N2; we will construct R_total x N x N
        Ns = self.cfg.num_symptoms
        Nh = self.cfg.num_herbs
        device = adj_ss.device
        sh_mats = []
        # assume adj_sh is full matrix of size (Ns+Nh, Ns+Nh) or shape (R, Ns, Nh) or (Ns, Nh)
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
                    # adj_sh is Ns x Nh
                    sh_mats.append(scale_adj(adj_sh))
        # compose relation list: first ss, then hh, then sh relations (as bipartite)
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

    def forward(self, sym_feat, herb_feat, adj_ss, adj_hh, adj_sh, batch_sym_indices:list, mask=None):
        """
        sym_feat: [Ns, symptom_feat_dim]
        herb_feat: [Nh, herb_feat_dim]
        adj_ss: [Ns, Ns], adj_hh: [Nh, Nh], adj_sh: either (Ns,Nh) or (N,N) or (R,N,N)
        batch_sym_indices: list of lists (each sample: indices of symptoms for that sample)
        mask: per sample symptom mask (padded) - optional
        returns: logits [batch_size, Nh]
        """
        device = sym_feat.device
        Ns = self.cfg.num_symptoms
        Nh = self.cfg.num_herbs
        D = self.cfg.gat_out_dim

        s_in = (self.sym_in_proj(sym_feat) + self.sym_embedding.weight
                + self.iso_sym_mask.unsqueeze(1) * self.unk_sym)    # [Ns, D]
        h_in = (self.herb_in_proj(herb_feat) + self.herb_embedding.weight
                + self.iso_herb_mask.unsqueeze(1) * self.unk_herb)  # [Nh, D]

        ss_norm = scale_adj(adj_ss.to(device))
        s1 = self.S_module(s_in, ss_norm)   # [Ns, D]

        # 3) GAT on HH -> h1
        hh_norm = scale_adj(adj_hh.to(device))
        h1 = self.H_module(h_in, hh_norm)   # [Nh, D]

        # 4) R-GAT on SH graph -> s2, h2
        # build relations tensor
        if isinstance(adj_sh, np.ndarray):
            adj_sh_t = torch.from_numpy(adj_sh).float().to(device)
        else:
            adj_sh_t = adj_sh.to(device)
        adj_rel = self._build_relations(adj_ss.to(device), adj_hh.to(device), adj_sh_t)
        # prepare full node feature (Ns+Nh)
        x_full = torch.cat([s_in, h_in], dim=0)  # [N, D]
        e_full = self.SH_module(x_full, adj_rel)  # returns [N, D]
        s2 = e_full[:Ns, :]
        h2 = e_full[Ns:, :]
        s2 = self.layernorm_sh_s(s2)
        h2 = self.layernorm_sh_h(h2)

        # 5) fuse s1+s2, h1+h2 with small gate (sigmoid of MLP)
        s_cat = torch.cat([s1, s2], dim=-1)  # [Ns, 2D]
        h_cat = torch.cat([h1, h2], dim=-1)  # [Nh, 2D]
        alpha_s = torch.sigmoid(self.gate_s(s_cat) / 2.0)  # scale down to avoid saturation
        alpha_h = torch.sigmoid(self.gate_h(h_cat) / 2.0)
        s_fused = self.layernorm_s(s1 + alpha_s * (s2 - s1))
        h_fused = self.layernorm_h(h1 + alpha_h * (h2 - h1))
        s_fused = self.dropout(s_fused)  # [Ns, D]
        h_fused = self.dropout(h_fused)  # [Nh, D]

        # 6) For each sample in the batch, we extract symptom embeddings by index -> form padded batch for SetTransformer
        # batch_sym_indices: list of lists of symptom indices
        # construct padded tensor [batch, L_max, D]
        batch_size = len(batch_sym_indices)
        lists = []
        lengths = []
        for idxs in batch_sym_indices:
            lengths.append(len(idxs))
            if len(idxs) == 0:
                # rare but guard
                lists.append(torch.zeros((1, D), device=device))
            else:
                lists.append(s_fused[idxs])  # [len_i, D]
        # pad
        padded = torch.nn.utils.rnn.pad_sequence(lists, batch_first=True)  # [batch, Lmax, D]
        # produce mask if not provided
        if mask is None:
            mask = torch.zeros(padded.shape[:2], dtype=torch.bool, device=device)
            for i, l in enumerate(lengths):
                mask[i, :l] = 1

        # 7) SetTransformer encoder/decoder
        H = self.encoder(padded, mask)  # [batch, Lmax, hidden]
        out = self.decoder(H, mask)     # [batch, 1, hidden]
        out = out.squeeze(1)            # [batch, hidden]
        out = self.dec_to_D(out)        # map to D (embedding dim) [batch, D]

        # 8) Now compute logits by dot with herb embeddings
        # herb embeddings are h_fused [Nh, D]; we compute out @ h_fused.T -> [batch, Nh]
        logits = torch.matmul(out, h_fused.t())  # [batch, Nh]
        sym_rec = self.sym_recon(s_fused)
        herb_rec = self.herb_recon(h_fused)
        return logits, s_fused, h_fused, sym_rec, herb_rec

# ==================== Dataset/Loader ====================
class PrescriptionDataset(Dataset):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # load index files: each line is a list of ints (symptom indices for sample; herbs for target)
        self.sym_idx = self._load_idx(cfg.pre_symptoms_file)
        self.herb_idx = self._load_idx(cfg.pre_herbs_file)
        # features
        self.sym_feat = torch.from_numpy(np.load(cfg.symptom_feature_file)).float()
        self.herb_feat = torch.from_numpy(np.load(cfg.herb_feature_file)).float()
        # graphs (numpy)
        self.adj_ss = self._load_graph(cfg.ss_graph_file, cfg.num_symptoms)
        self.adj_hh = self._load_graph(cfg.hh_graph_file, cfg.num_herbs)
        self.adj_sh = self._load_graph_full(cfg.sh_graph_file, cfg.num_symptoms + cfg.num_herbs)
        # targets: multi-hot vectors
        self.targets = [self._build_target(h, cfg.num_herbs) for h in self.herb_idx]

    def _load_idx(self, fpath):
        arr = []
        with open(fpath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 0:
                    arr.append([])
                else:
                    arr.append([int(x) for x in parts])
        return arr

    def _load_graph(self, fpath, n_nodes):
        # assume saved as numpy edges array: [2, num_edges] or else adjacency matrix; adapt accordingly
        data = np.load(fpath, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] == 2:
            adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
            if data.size > 0:
                rows, cols = data[0], data[1]
                adj[rows, cols] = 1
                adj = torch.max(adj, adj.t())
            return adj
        elif isinstance(data, np.ndarray) and data.shape == (n_nodes, n_nodes):
            return (torch.from_numpy(data).float() > 0).float()
        else:
            # fallback: empty
            return torch.zeros((n_nodes, n_nodes), dtype=torch.float32)

    def _load_graph_full(self, fpath, n_total):
        # adj_sh stored maybe as full N x N with Ns+Nh size
        data = np.load(fpath, allow_pickle=True)
        if isinstance(data, np.ndarray):
            if data.ndim == 2 and data.shape[0] == n_total and data.shape[1] == n_total:
                return (torch.from_numpy(data).float() > 0).float()
            else:
                # try load edges
                if data.ndim == 2 and data.shape[0] == 2:
                    adj = torch.zeros((n_total, n_total), dtype=torch.float32)
                    rows, cols = data[0], data[1]
                    adj[rows, cols] = 1
                    adj = torch.max(adj, adj.t())
                    return adj
        return torch.zeros((n_total, n_total), dtype=torch.float32)

    def _build_target(self, herbs: List[int], n):
        t = torch.zeros(n, dtype=torch.float32)
        for h in herbs:
            if h >= 0 and h < n:
                t[h] = 1.0
        return t

    def __len__(self):
        return len(self.sym_idx)

    def __getitem__(self, idx):
        sym_list = self.sym_idx[idx]
        # fetch features by indices
        sym_feat = self.sym_feat  # global features passed to model; per-sample we only return indices
        herb_feat = self.herb_feat
        target = self.targets[idx]
        return sym_list, target

def collate_fn(batch):
    # batch: list of (sym_list, target)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sym_lists, targets = zip(*batch)
    targets = torch.stack(targets, dim=0)
    # pass sym_lists as list-of-lists to model
    return list(sym_lists), targets

# ==================== Metrics & training/eval loops ====================
def precision_recall_f1_at_k(probs: torch.Tensor, targets: torch.Tensor, K_list=[5,10,20]):
    # probs: [B, C], targets: [B, C] binary (0/1)
    B, C = probs.shape
    res = {}
    for K in K_list:
        topk = torch.topk(probs, K, dim=1).indices  # [B, K]
        preds = torch.zeros_like(targets, dtype=torch.uint8)
        preds.scatter_(1, topk, 1)
        trues = (targets > 0.5).to(torch.uint8)
        inter = (preds & trues).sum(dim=1).to(torch.float32)
        prec_per_sample = inter / float(K)
        true_counts = trues.sum(dim=1).to(torch.float32)
        rec_per_sample = torch.where(true_counts > 0, inter / true_counts, torch.zeros_like(inter))
        f1_per_sample = torch.where((prec_per_sample + rec_per_sample) > 0,
                                    2 * prec_per_sample * rec_per_sample / (prec_per_sample + rec_per_sample),
                                    torch.zeros_like(prec_per_sample))
        res[f"P@{K}"] = float(prec_per_sample.mean().item())
        res[f"R@{K}"] = float(rec_per_sample.mean().item())
        res[f"F1@{K}"] = float(f1_per_sample.mean().item())
    return res

SSL_WEIGHT = float(os.environ.get('SSL_WEIGHT', '0.01'))
RECON_SCALE = float(os.environ.get('RECON_SCALE', '1.0'))

def train_one_epoch(model, loader, opt, scaler, device, cfg, sym_feat, herb_feat, adj_ss, adj_hh, adj_sh, criterion):
    model.train()
    total_loss = 0.0
    n = 0
    for sym_lists, targets in loader:
        targets = targets.to(device)
        if getattr(cfg, 'sym_drop', 0) > 0:
            sym_lists = [[s for s in lst if random.random() >= cfg.sym_drop] or lst for lst in sym_lists]
        opt.zero_grad()
        with autocast():
            logits, _, _, sym_rec, herb_rec = model(sym_feat, herb_feat, adj_ss, adj_hh, adj_sh, sym_lists)
            loss = criterion(logits, targets)
            recon_s = F.mse_loss(sym_rec, sym_feat.to(device))
            recon_h = F.mse_loss(herb_rec, herb_feat.to(device))
            recon_loss = 0.5 * (recon_s + recon_h)
            loss = loss + SSL_WEIGHT * RECON_SCALE * recon_loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        total_loss += float(loss.item())
        n += 1
    return total_loss / max(1, n)

@torch.no_grad()
def evaluate(model, loader, device, cfg, sym_feat, herb_feat, adj_ss, adj_hh, adj_sh, criterion, K_list=[5, 10, 20]):
    model.eval()
    total_loss = 0.0
    n = 0
    total_true = 0
    stats = {K: {"TP": 0.0, "total_pred": 0.0} for K in K_list}
    for sym_lists, targets in loader:
        targets = targets.to(device)
        logits, *_ = model(sym_feat, herb_feat, adj_ss, adj_hh, adj_sh, sym_lists)
        loss = criterion(logits, targets)
        total_loss += float(loss.item())
        n += 1
        probs = torch.sigmoid(logits)
        trues = (targets > 0.5).bool()
        total_true += trues.sum().item()
        for K in K_list:
            topk = torch.topk(probs, K, dim=1).indices
            pred_mask = torch.zeros_like(probs, dtype=torch.bool)
            pred_mask.scatter_(1, topk, True)
            TP = (pred_mask & trues).sum().item()
            stats[K]["TP"] += TP
            stats[K]["total_pred"] += probs.size(0) * K
    metrics = {}
    for K in K_list:
        P = stats[K]["TP"] / stats[K]["total_pred"] if stats[K]["total_pred"] > 0 else 0.0
        R = stats[K]["TP"] / total_true if total_true > 0 else 0.0
        F1 = 2 * P * R / (P + R + 1e-12) if (P + R) > 0 else 0.0
        metrics[f"P@{K}"] = float(P)
        metrics[f"R@{K}"] = float(R)
        metrics[f"F1@{K}"] = float(F1)
    return total_loss / max(1, n), metrics

# ==================== Main: initialize data, model, train loop ====================
def main(seed=None):
    if seed is None:
        seed = int(os.environ.get("SEED", 42))
        cfg.pos_weight = float(os.environ.get("POS_WEIGHT", cfg.pos_weight))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # check files
    files = [cfg.symptom_feature_file, cfg.herb_feature_file, cfg.ss_graph_file, cfg.hh_graph_file, cfg.sh_graph_file, cfg.pre_symptoms_file, cfg.pre_herbs_file]
    for f in files:
        if not os.path.exists(f):
            print(f"[WARNING] file not found: {f}")

    # dataset
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
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    # load global features and graphs to device
    sym_feat = ds.sym_feat.to(device)
    herb_feat = ds.herb_feat.to(device)
    adj_ss = ds.adj_ss.to(device)
    adj_hh = ds.adj_hh.to(device)
    adj_sh = ds.adj_sh.to(device)

    # model
    model = RGAT_to_SetTransformer(cfg).to(device)
    if os.path.exists(cfg.cold_nodes_file):
        cold = np.load(cfg.cold_nodes_file)
        model.iso_sym_mask.copy_(torch.from_numpy(cold['iso_sym'].astype(np.float32)))
        model.iso_herb_mask.copy_(torch.from_numpy(cold['iso_herb'].astype(np.float32)))
    else:
        pass
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # criterion, optimizer, scheduler
    pos_weight = torch.tensor(cfg.pos_weight, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.num_epochs)
    model.sym_embedding.weight.requires_grad_(False)
    model.herb_embedding.weight.requires_grad_(False)
    n_frozen = model.sym_embedding.weight.numel() + model.herb_embedding.weight.numel()
    n_total = sum(p.numel() for p in model.parameters())
    scaler = GradScaler()
    early = 0
    best_f1 = 0.0
    best_path = f"best_strgat_{os.environ.get('MODEL_NAME', 'ST-RGAT-bin-raw')}_s{seed}.pth"

    for epoch in range(cfg.num_epochs):
        tr_loss = train_one_epoch(model, tr_loader, opt, scaler, device, cfg, sym_feat, herb_feat, adj_ss, adj_hh, adj_sh, criterion)
        va_loss, metrics = evaluate(model, va_loader, device, cfg, sym_feat, herb_feat, adj_ss, adj_hh, adj_sh, criterion)
        scheduler.step()

        print(f"Epoch {epoch+1}/{cfg.num_epochs}:")
        for k, v in metrics.items():
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
    final_loss, final_metrics = evaluate(model, te_loader, device, cfg, sym_feat, herb_feat, adj_ss, adj_hh, adj_sh, criterion)
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    model_name = os.environ.get("MODEL_NAME", "ST-RGAT-bin-raw-frz")
    _script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    csv_path = os.path.join(_script_dir, "results_multiseed_strgat_bin.csv")
    metric_keys = sorted(final_metrics.keys())
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "seed"] + metric_keys)
        writer.writerow([model_name, seed] + [f"{final_metrics[k]:.6f}" for k in metric_keys])
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