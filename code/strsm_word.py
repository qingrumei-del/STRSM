
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import csv
import math
from gensim.models import Word2Vec
import random
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, dropout=0.2):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        if hasattr(self, 'ln0'):
            nn.init.ones_(self.ln0.weight)
            nn.init.zeros_(self.ln0.bias)
        if hasattr(self, 'ln1'):
            nn.init.ones_(self.ln1.weight)
            nn.init.zeros_(self.ln1.bias)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        scale = math.sqrt(dim_split)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / scale, 2)
        A = self.dropout(A)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        if hasattr(self, 'ln0'):
            O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = self.dropout(O)
        if hasattr(self, 'ln1'):
            O = self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False, dropout=0.2):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, dropout=dropout)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, dropout=0.2):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln, dropout=dropout)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln, dropout=dropout)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, dropout=0.2):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln, dropout=dropout)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

# ================== Set Transformer ==================
class SetTransformer(nn.Module):
    def __init__(self, dim_input, dim_output, num_heads, num_inds, num_seeds,
                 ln=False, vocab_size=None, embedding_matrix=None, dropout=0.2):
        super(SetTransformer, self).__init__()
        if embedding_matrix is not None:
            self.symptom_embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            if vocab_size is None:
                vocab_size = 390
            self.symptom_embedding = nn.Embedding(vocab_size, dim_input)

        self.enc = nn.Sequential(
            ISAB(dim_input, dim_input, num_heads, num_inds, ln=ln, dropout=dropout),
            ISAB(dim_input, dim_input, num_heads, num_inds, ln=ln, dropout=dropout)
        )
        self.dec = nn.Sequential(
            PMA(dim_input, num_heads, num_seeds, ln=ln, dropout=dropout),
            SAB(dim_input, dim_input, num_heads, ln=ln, dropout=dropout),
            nn.Linear(dim_input, dim_input),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_input, dim_output)
        )

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        for module in self.dec:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, X):
        device = self.symptom_embedding.weight.device
        embedded = [self.symptom_embedding(x.to(device)) for x in X]
        lengths = [len(x) for x in embedded]
        max_len = max(lengths)
        padded = torch.stack([
            torch.cat([x, torch.zeros(max_len - len(x), x.size(1), device=device)])
            for x in embedded
        ])
        Z = self.enc(padded)
        output = self.dec(Z)
        return output.squeeze(1)

# ================== Dataset ==================
class PrescriptionDataset(Dataset):
    def __init__(self, symptoms_list, herbs_list, symptom_vocab_size, herb_vocab_size):
        self.symptoms_list = symptoms_list
        self.herbs_list = herbs_list
        self.symptom_vocab_size = symptom_vocab_size
        self.herb_vocab_size = herb_vocab_size

    def __len__(self):
        return len(self.symptoms_list)

    def __getitem__(self, idx):
        symptoms = self.symptoms_list[idx]
        herbs = self.herbs_list[idx]
        symptoms_tensor = torch.tensor(symptoms, dtype=torch.long)
        herb_label = torch.zeros(self.herb_vocab_size)
        herb_label[herbs] = 1
        return symptoms_tensor, herb_label

def collate_fn(batch):
    symptoms, herbs = zip(*batch)
    herbs = torch.stack(herbs, dim=0)
    return symptoms, herbs

# ================== Evaluate ==================
def evaluate(model, dataloader, k_list, device):
    model.eval()
    precisions, recalls, f1_scores = {k: 0 for k in k_list}, {k: 0 for k in k_list}, {k: 0 for k in k_list}
    correct_total, pred_total, true_total = {k: 0 for k in k_list}, {k: 0 for k in k_list}, 0

    with torch.no_grad():
        for symptoms, herbs in dataloader:
            herbs = herbs.to(device)
            probs = torch.sigmoid(model(symptoms))
            true_total += herbs.sum().item()
            for k in k_list:
                topk = torch.topk(probs, k, dim=1).indices
                pred = torch.zeros_like(probs).scatter_(1, topk, 1)
                correct_total[k] += (pred * herbs).sum().item()
                pred_total[k] += k * herbs.size(0)
    for k in k_list:
        P = correct_total[k] / pred_total[k] if pred_total[k] else 0
        R = correct_total[k] / true_total if true_total else 0
        F1 = 2 * P * R / (P + R + 1e-12) if (P + R) else 0
        precisions[k], recalls[k], f1_scores[k] = P, R, F1
    return precisions, recalls, f1_scores

def load_group_split(split_file='../data/split_indices.npz', seed=42):
    sp = np.load(split_file)
    merged_idx = np.concatenate([sp['train_idx'], sp['val_idx']])
    glabels = sp['group_labels'][merged_idx]
    unique_g = np.unique(glabels)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique_g))
    n_val_g = int(len(unique_g) * 0.1)
    val_groups = set(unique_g[perm[:n_val_g]].tolist())
    is_val = np.array([g in val_groups for g in glabels])
    inner_train = merged_idx[~is_val].tolist()
    inner_val = merged_idx[is_val].tolist()
    test_idx = sp['test_idx'].tolist()
    return inner_train, inner_val, test_idx

# ================== Main ==================
def main(seed=None):
    if seed is None:
        seed = int(os.environ.get("SEED", 42))
    SEED = seed
    set_seed(SEED)

    with open('../data/symptom_contains.txt', 'r', encoding='utf-8') as f:
        symptoms_text = [line.strip() for line in f]
    vocab_size = len(symptoms_text)

    dim_input = 64
    sentences = [[text] for text in symptoms_text]

    word2vec_model = Word2Vec(
        sentences,
        vector_size=dim_input,
        window=5,
        min_count=1,
        workers=1,
        seed=42,
        epochs=10
    )

    embedding_matrix = torch.zeros(vocab_size, dim_input)
    for i, word in enumerate(symptoms_text):
        embedding_matrix[i] = torch.tensor(word2vec_model.wv[word])

    with open('../data/pre_symptoms.txt', 'r') as f:
        symptoms_data = [list(map(int, line.strip().split())) for line in f]
    with open('../data/pre_herbs.txt', 'r') as f:
        herbs_data = [list(map(int, line.strip().split())) for line in f]

    all_herbs = set()
    for h_list in herbs_data:
        all_herbs.update(h_list)
    herb_vocab_size = max(all_herbs) + 1

    inner_train, inner_val, test_idx = load_group_split('../data/split_indices.npz', SEED)

    train_dataset = PrescriptionDataset([symptoms_data[i] for i in inner_train],
                                        [herbs_data[i] for i in inner_train], vocab_size, herb_vocab_size)
    val_dataset = PrescriptionDataset([symptoms_data[i] for i in inner_val],
                                      [herbs_data[i] for i in inner_val], vocab_size, herb_vocab_size)
    test_dataset = PrescriptionDataset([symptoms_data[i] for i in test_idx],
                                       [herbs_data[i] for i in test_idx], vocab_size, herb_vocab_size)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        generator=generator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dim_output = herb_vocab_size
    num_heads = 4
    num_inds = 32
    num_seeds = 1
    ln = True
    dropout = 0.3
    k_list = [5, 10, 20]

    model = SetTransformer(
        dim_input=dim_input,
        dim_output=dim_output,
        num_heads=num_heads,
        num_inds=num_inds,
        num_seeds=num_seeds,
        ln=ln,
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix,
        dropout=dropout
    ).to(device)

    pos_weight = torch.tensor(3.0, device=device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=200,
        last_epoch=-1
    )
    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    num_epochs = 200
    patience = 40
    best_f1 = 0.0
    early = 0
    best_model_path = f"best_stword_aligned200_s{seed}.pth"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for symptoms, herbs in train_loader:
            herbs = herbs.to(device)
            symptoms = [x.to(device) for x in symptoms]
            optimizer.zero_grad()
            with autocast():
                logits = model(symptoms)
                loss = criterion(logits, herbs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()

        precisions, recalls, f1s = evaluate(model, val_loader, k_list, device)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}')
        for k in k_list:
            print(f'  K={k}: Precision: {precisions[k]:.4f}, Recall: {recalls[k]:.4f}, F1: {f1s[k]:.4f}')

        if f1s[10] > best_f1 + 1e-6:
            best_f1 = f1s[10]
            torch.save(model.state_dict(), best_model_path)
            early = 0
        else:
            early += 1
            if early >= patience:
                print("Early stopping triggered.")
                break

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    final_precisions, final_recalls, final_f1s = evaluate(model, test_loader, k_list, device)
    final_metrics = {}
    for k in k_list:
        print(
            f'  K={k}: Precision: {final_precisions[k]:.4f}, Recall: {final_recalls[k]:.4f}, F1: {final_f1s[k]:.4f}')
        final_metrics[f"P@{k}"] = float(final_precisions[k])
        final_metrics[f"R@{k}"] = float(final_recalls[k])
        final_metrics[f"F1@{k}"] = float(final_f1s[k])

    model_name = os.environ.get("MODEL_NAME", "ST-WORD-aligned-200ep")
    _script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    csv_path = os.path.join(_script_dir, "results_multiseed_stword_aligned_200ep.csv")
    metric_keys = sorted(final_metrics.keys())
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "seed"] + metric_keys)
        writer.writerow([model_name, seed] + [f"{final_metrics[k]:.6f}" for k in metric_keys])
    return final_metrics

if __name__ == '__main__':
    seeds = [int(s) for s in os.environ.get("SEEDS", "43,44,45,46").split(",") if s.strip()]
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