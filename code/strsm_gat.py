
import os
import csv

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import random

class Config:
    num_symptoms = 390
    num_herbs = 811
    symptom_embed_dim = 100
    herb_embed_dim = 23

    gat_out_dim = 100
    gat_num_heads = 4
    gat_dropout = 0.1

    st_num_inds = 32
    st_dim_hidden = 128
    st_num_heads = 4
    st_ln = True

    batch_size = 32
    learning_rate = 0.001
    weight_decay = 1e-4
    num_epochs = 200
    train_ratio = 0.7
    dropout_rate = 0.3
    sym_drop = 0.15
    patience = 50

    symptom_feature_file = "../processing_data/symptom_features.npy"
    herb_feature_file = "../processing_data/herb_features.npy"
    ss_graph_file = "../processing_data/symptom_symptom_graph_train_622m.npy"
    hh_graph_file = "../processing_data/herb_herb_graph_train_622m.npy"
    sh_graph_file = "../processing_data/symptom_herb_graph_train_622m.npy"
    pre_symptoms_file = "../data/pre_symptoms.txt"
    pre_herbs_file = "../data/pre_herbs.txt"
    split_file = "../data/split_indices.npz"
    cold_nodes_file = "../processing_data/cold_nodes_622m.npz"

def scale_adj(adj: torch.Tensor):
    a = (adj.float() / adj.max().clamp(min=1.0)).clone()
    a.fill_diagonal_(1.0)
    return a

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, logits, targets):
        return self.bce(logits, targets)

# ==================== Focal Loss ====================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        p_t = torch.exp(-bce_loss)  # Probability of being correctly classified
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss

# ==================== EarlyStopping ====================
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score, model, path='best_model.pth'):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            torch.save(model.state_dict(), path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.1, activation=nn.ReLU(), use_layer_norm=True):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.Wq = nn.Linear(in_dim, out_dim, bias=False)
        self.Wk = nn.Linear(in_dim, out_dim, bias=False)
        self.Wv = nn.Linear(in_dim, out_dim, bias=False)
        self.W_res = nn.Linear(in_dim, out_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim) if use_layer_norm else None
        self.activation = activation

    def forward(self, x, adj):
        N = x.size(0)
        Q = self.Wq(x).view(N, self.num_heads, self.head_dim)
        K = self.Wk(x).view(N, self.num_heads, self.head_dim)
        V = self.Wv(x).view(N, self.num_heads, self.head_dim)

        scores = torch.einsum("nhd,mhd->hnm", Q, K) / math.sqrt(self.head_dim)
        mask = adj.unsqueeze(0).expand(self.num_heads, -1, -1)
        neg_inf = -1e4
        scores = scores.masked_fill(mask == 0, neg_inf)
        scores = scores + torch.log1p(adj).unsqueeze(0)

        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("hnm,mhd->nhd", attn, V)
        out = out.reshape(N, self.out_dim)

        out = out + self.W_res(x)
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        return self.activation(out)

class GATEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ss_encoder = MultiHeadAttentionLayer(config.symptom_embed_dim, config.gat_out_dim, config.gat_num_heads)
        self.hh_encoder = MultiHeadAttentionLayer(config.herb_embed_dim, config.gat_out_dim, config.gat_num_heads)
        self.symptom_proj = nn.Linear(config.symptom_embed_dim, config.gat_out_dim)
        self.herb_proj = nn.Linear(config.herb_embed_dim, config.gat_out_dim)
        self.sh_encoder = MultiHeadAttentionLayer(config.gat_out_dim, config.gat_out_dim, config.gat_num_heads)
        self.alpha = nn.Parameter(torch.tensor([0.2, 0.2]), requires_grad=True)

    def forward(self, symptom_features, herb_features, adj_ss, adj_hh, adj_sh):
        ss_emb = self.ss_encoder(symptom_features, scale_adj(adj_ss))
        hh_emb = self.hh_encoder(herb_features, scale_adj(adj_hh))
        sh_input = torch.cat([self.symptom_proj(symptom_features), self.herb_proj(herb_features)], dim=0)
        sh_emb = self.sh_encoder(sh_input, scale_adj(adj_sh))
        sh_sym, sh_herb = sh_emb[:symptom_features.size(0)], sh_emb[symptom_features.size(0):]
        alpha = torch.softmax(self.alpha, dim=0)
        return alpha[0]*ss_emb + alpha[1]*sh_sym, alpha[0]*hh_emb + alpha[1]*sh_herb

# ==================== Set Transformer ====================
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_V = dim_V
        self.fc_q, self.fc_k, self.fc_v = nn.Linear(dim_Q, dim_V), nn.Linear(dim_K, dim_V), nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.dropout = nn.Dropout(dropout)
        self.ln0 = nn.LayerNorm(dim_V) if ln else None
        self.ln1 = nn.LayerNorm(dim_V) if ln else None

    def forward(self, Q, K, mask=None):
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(K)
        b, nq, _ = Q.size()
        _, nk, _ = K.size()
        Q_ = Q.view(b, nq, self.num_heads, -1).transpose(1,2)
        K_ = K.view(b, nk, self.num_heads, -1).transpose(1,2)
        V_ = V.view(b, nk, self.num_heads, -1).transpose(1,2)
        scores = torch.matmul(Q_, K_.transpose(-2,-1)) / math.sqrt(Q_.size(-1))
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask==0, -1e9)
        A = torch.softmax(scores, dim=-1)
        O = torch.matmul(A, V_).transpose(1,2).reshape(b, nq, self.dim_V)
        O = O if self.ln0 is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        return O if self.ln1 is None else self.ln1(O)

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False, dropout=0.1):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln, dropout)
    def forward(self, X, mask=None): return self.mab(X, X, mask)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, dropout=0.1):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln, dropout)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln, dropout)
    def forward(self, X, mask=None):
        H = self.mab0(self.I.repeat(X.size(0),1,1), X, mask)
        return self.mab1(X,H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, dropout=0.1):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1,num_seeds,dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim,dim,dim,num_heads,ln,dropout)
    def forward(self,X,mask=None): return self.mab(self.S.repeat(X.size(0),1,1),X)

class SetTransformer(nn.Module):
    def __init__(self, dim_input,num_outputs,herb_proj,num_inds,dim_hidden,num_heads,ln=False,dropout=0.1):
        super().__init__()
        self.enc = nn.Sequential(ISAB(dim_input,dim_hidden,num_heads,num_inds,ln,dropout),
                                 ISAB(dim_hidden,dim_hidden,num_heads,num_inds,ln,dropout))
        self.dec = nn.Sequential(PMA(dim_hidden,num_heads,num_outputs,ln,dropout),
                                 SAB(dim_hidden,dim_hidden,num_heads,ln,dropout),
                                 nn.Linear(dim_hidden,100))
        self.herb_proj = herb_proj
        self.dropout = nn.Dropout(dropout)

    def forward(self,X,herb_emb,mask=None):
        out = self.dec(self.enc(X))
        out = self.dropout(out)
        herb_emb = self.herb_proj(herb_emb)
        return torch.matmul(out,herb_emb.t()).squeeze(1)

class GATSetTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gat = GATEncoder(config)
        self.herb_proj = nn.Linear(config.gat_out_dim, config.gat_out_dim)
        self.st = SetTransformer(config.gat_out_dim,1,self.herb_proj,config.st_num_inds,
                                 config.st_dim_hidden,config.st_num_heads,config.st_ln,config.dropout_rate)
        self.unk_sym = nn.Parameter(torch.zeros(config.symptom_embed_dim))
        self.unk_herb = nn.Parameter(torch.zeros(config.herb_embed_dim))
        self.register_buffer("iso_sym_mask", torch.zeros(config.num_symptoms))
        self.register_buffer("iso_herb_mask", torch.zeros(config.num_herbs))
    def forward(self,sym_feat,herb_feat,adj_ss,adj_hh,adj_sh,indices,mask):
        sym_feat = sym_feat + self.iso_sym_mask.unsqueeze(1) * self.unk_sym
        herb_feat = herb_feat + self.iso_herb_mask.unsqueeze(1) * self.unk_herb
        sym_emb, herb_emb = self.gat(sym_feat,herb_feat,adj_ss,adj_hh,adj_sh)
        batch = [sym_emb[idx] for idx in indices]
        batch_pad = torch.nn.utils.rnn.pad_sequence(batch,batch_first=True)
        return self.st(batch_pad,herb_emb,mask)

class PrescriptionDataset(Dataset):
    def __init__(self,config):
        self.sym_idx = self.load_idx(config.pre_symptoms_file)
        self.herb_idx = self.load_idx(config.pre_herbs_file)
        self.sym_feat = torch.from_numpy(np.load(config.symptom_feature_file)).float()
        self.herb_feat = torch.from_numpy(np.load(config.herb_feature_file)).float()
        self.adj_ss = (torch.from_numpy(np.load(config.ss_graph_file)).float() > 0).float()
        self.adj_hh = (torch.from_numpy(np.load(config.hh_graph_file)).float() > 0).float()
        self.adj_sh = (torch.from_numpy(np.load(config.sh_graph_file)).float() > 0).float()
        self.targets=[self.build_target(h,config.num_herbs) for h in self.herb_idx]

    def load_idx(self,f):
        return [list(map(int,line.strip().split())) for line in open(f)]
    def build_target(self,herbs,n): t=torch.zeros(n); [t.__setitem__(h,1) for h in herbs if h<n]; return t
    def __len__(self): return len(self.sym_idx)
    def __getitem__(self,idx):
        sym=self.sym_feat[self.sym_idx[idx]];mask=torch.ones(len(self.sym_idx[idx]))
        return sym,self.targets[idx],mask,self.sym_idx[idx]

def collate_fn(batch):
    batch.sort(key=lambda x:x[0].shape[0],reverse=True)
    syms,tars,masks,idxs=zip(*batch)
    return torch.nn.utils.rnn.pad_sequence(syms,batch_first=True),torch.stack(tars), \
           torch.nn.utils.rnn.pad_sequence(masks,batch_first=True),[s.shape[0] for s in syms],idxs

def train_model(model,loader,crit,opt,scaler,dev,sf,hf,ss,hh,sh,cfg):
    model.train();tot=0
    for sym,tar,mask,len_,idx in loader:
        sym,tar,mask=sym.to(dev),tar.to(dev),mask.to(dev)
        if getattr(cfg, 'sym_drop', 0) > 0:
            idx=[[s for s in lst if random.random()>=cfg.sym_drop] or lst for lst in idx]
        opt.zero_grad()
        with autocast(enabled=False):
            out=model(sf,hf,ss,hh,sh,idx,mask)
            loss=crit(out,tar)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(opt);scaler.update()
        tot+=loss.item()
    return tot/len(loader)

def evaluate(model,loader,crit,dev,sf,hf,ss,hh,sh,K_list=[5,10,20]):
    model.eval();tot=0
    total_true=0
    stats={K:{"TP":0.0,"total_pred":0.0} for K in K_list}
    with torch.no_grad():
        for sym,tar,mask,len_,idx in loader:
            sym,tar,mask=sym.to(dev),tar.to(dev),mask.to(dev)
            out=model(sf,hf,ss,hh,sh,idx,mask)
            loss=crit(out,tar);tot+=loss.item()
            probs=torch.sigmoid(out)
            trues=(tar>0.5).bool()
            total_true+=trues.sum().item()
            for K in K_list:
                topk=torch.topk(probs,K,dim=1).indices
                pred_mask=torch.zeros_like(probs,dtype=torch.bool)
                pred_mask.scatter_(1,topk,True)
                TP=(pred_mask&trues).sum().item()
                stats[K]["TP"]+=TP
                stats[K]["total_pred"]+=probs.size(0)*K
    metrics={}
    for K in K_list:
        P=stats[K]["TP"]/stats[K]["total_pred"] if stats[K]["total_pred"]>0 else 0.0
        R=stats[K]["TP"]/total_true if total_true>0 else 0.0
        F1=2*P*R/(P+R+1e-12) if (P+R)>0 else 0.0
        metrics[f"P@{K}"]=float(P);metrics[f"R@{K}"]=float(R);metrics[f"F1@{K}"]=float(F1)
    return tot/len(loader),metrics

def main(seed=None):
    if seed is None:
        seed = int(os.environ.get("SEED", 42))
    random.seed(int(os.environ.get("RANDOM_SEED", seed)))
    np.random.seed(seed)
    _init_seed = int(os.environ.get("INIT_SEED", seed))
    torch.manual_seed(_init_seed)
    torch.cuda.manual_seed_all(_init_seed)
    if os.environ.get("DETERMINISTIC", "0") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    config=Config();dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset=PrescriptionDataset(config)
    sp=np.load(config.split_file)
    merged_idx=np.concatenate([sp['train_idx'],sp['val_idx']])
    glabels=sp['group_labels'][merged_idx]
    unique_g=np.unique(glabels)
    rng=np.random.RandomState(seed)
    perm=rng.permutation(len(unique_g))
    n_val_g=int(len(unique_g)*0.1)
    val_groups=set(unique_g[perm[:n_val_g]].tolist())
    is_val=np.array([g in val_groups for g in glabels])
    tr_set=torch.utils.data.Subset(dataset,merged_idx[~is_val].tolist())
    va_set=torch.utils.data.Subset(dataset,merged_idx[is_val].tolist())
    te_set=torch.utils.data.Subset(dataset,sp['test_idx'].tolist())
    tr_loader=DataLoader(tr_set,batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn,
                         pin_memory=True,num_workers=4)
    va_loader=DataLoader(va_set,batch_size=config.batch_size,shuffle=False,collate_fn=collate_fn,
                         pin_memory=True,num_workers=4)
    te_loader=DataLoader(te_set,batch_size=config.batch_size,shuffle=False,collate_fn=collate_fn,
                         pin_memory=True,num_workers=4)
    model=GATSetTransformer(config).to(dev)
    if os.path.exists(config.cold_nodes_file):
        cold=np.load(config.cold_nodes_file)
        model.iso_sym_mask.copy_(torch.from_numpy(cold['iso_sym'].astype(np.float32)))
        model.iso_herb_mask.copy_(torch.from_numpy(cold['iso_herb'].astype(np.float32)))
    else:
        pass
    sf,hf,ss,hh,sh=dataset.sym_feat.to(dev),dataset.herb_feat.to(dev),dataset.adj_ss.to(dev),dataset.adj_hh.to(dev),dataset.adj_sh.to(dev)
    crit=WeightedBCEWithLogitsLoss(pos_weight=3.0).to(dev)
    opt=optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=config.num_epochs)
    patience=int(os.environ.get("PATIENCE",config.patience))
    warmup=int(os.environ.get("WARMUP", 0))
    scaler=GradScaler(enabled=False);es=EarlyStopping(patience=patience)
    best_path=f"best_stgat_{os.environ.get('MODEL_NAME', 'ST-GAT-bin-raw')}_s{seed}.pth"
    for ep in range(config.num_epochs):
        if warmup > 0 and ep < warmup:
            warm_lr=config.learning_rate*(ep+1)/warmup
            for pg in opt.param_groups: pg['lr']=warm_lr
        tr_loss=train_model(model,tr_loader,crit,opt,scaler,dev,sf,hf,ss,hh,sh,config)
        va_loss,metrics=evaluate(model,va_loader,crit,dev,sf,hf,ss,hh,sh)
        sch.step()
        print(f"Epoch {ep+1}: Train {tr_loss:.4f}, Val {va_loss:.4f}")
        for k,v in metrics.items(): print(f"  {k}: {v:.4f}",end=" ")
        print()
        es(metrics["F1@10"],model,best_path)
        if es.early_stop: print("Early stopping"); break

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path,map_location=dev))
    te_loss,final_metrics=evaluate(model,te_loader,crit,dev,sf,hf,ss,hh,sh)
    for k,v in final_metrics.items(): print(f"  {k}: {v:.4f}",end=" ")
    print()

    model_name=os.environ.get("MODEL_NAME","ST-GAT-bin-raw-fp32-frz")
    _script_dir=os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    csv_path=os.path.join(_script_dir,"results_multiseed_stgat_bin.csv")
    metric_keys=sorted(final_metrics.keys())
    write_header=not os.path.exists(csv_path)
    with open(csv_path,"a",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        if write_header:
            w.writerow(["model","seed"]+metric_keys)
        w.writerow([model_name,seed]+[f"{final_metrics[k]:.6f}" for k in metric_keys])
    return final_metrics

if __name__=="__main__":
    seeds=[int(s) for s in os.environ.get("SEEDS","42,43,44,45,46").split(",") if s.strip()]
    all_metrics=[]
    for s in seeds:
        print(f"\n========== SEED = {s} ==========")
        all_metrics.append((s,main(seed=s)))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if len(all_metrics)>1:
        for k in sorted(all_metrics[0][1].keys()):
            vals=np.array([m[k] for _,m in all_metrics],dtype=np.float64)
            print(f"  {k}: {vals.mean():.4f} ± {vals.std(ddof=1):.4f}")