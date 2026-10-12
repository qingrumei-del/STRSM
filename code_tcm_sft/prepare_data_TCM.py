import os
import re
import json
import hashlib
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "SFT_structGeneral_310860.json"  # download from the public HuggingFace collection (see README)
OUT = os.path.join(HERE, '..', 'data_TCM')
N = 14567
SEED = 42
SYM_MIN_FREQ = 3

LABELS = ['基本信息', '主诉', '现病史', '体格检查', '既往史', '过敏史', '个人史',
          '家族史', '月经史', '舌象', '脉象', '证型', '治法', '方剂', '中药',
          '诊断', '治疗方案', '中医诊断', '西医诊断', '上次就诊内容']
META = re.compile(r'(年龄|性别|身高|体重|血压|体温|脉搏|呼吸|心率)[:：]\s*\S+')
HERB_RE = re.compile(r'([一-龥]+)(\d+\.?\d*)')

NOISE_TAIL = ('汤', '丸', '散', '丹', '饮', '煎', '膏', '方', '诊', '次',
              '胶囊', '颗粒', '片', '口服液', '冲剂', '糖浆',
              '证', '法')

def extract_symptoms(inp):
    inp = re.sub(r'[一-龥]+\d+\.?\d*', ' ', inp)
    inp = re.sub(r'\([^)]*\)', ' ', inp)
    inp = META.sub(' ', inp)
    for lab in LABELS:
        inp = inp.replace(lab, ' ')
    inp = inp.replace(':', ' ').replace('：', ' ')
    toks = []
    for t in inp.split():
        if not re.fullmatch(r'[一-龥]{2,8}', t):
            continue
        if any(t.endswith(s) for s in NOISE_TAIL):
            continue
        toks.append(t)
    return toks

HERB_SEG = re.compile(r'中药[:：]\s*(.*?)(?=方剂[:：]|证型[:：]|治法[:：]|诊断[:：]|治疗方案[:：]|中医诊断[:：]|西医诊断[:：]|中药[:：]|$)')

HERB_PREFIXES = [
    '黑豆制', '黑豆', '蜜炙', '醋炙', '盐炙', '酒炙', '麸炒', '土炒', '米炒', '清炒', '生晒',
    '盐', '姜', '醋', '蜜', '酒', '炒', '焦', '煅', '制', '炙', '炮', '炭', '土', '麸', '烫',
    '蒸', '煮', '煨', '烧', '焙', '淬', '净', '清', '法', '熟', '干', '鲜', '生',
]

HERB_ALIASES = {
    '萸肉': '山茱萸', '净萸肉': '山茱萸', '晒参': '人参',
    '血竭粉': '血竭', '三七粉': '三七', '琥珀粉': '琥珀', '滑石粉': '滑石',
    '珍珠粉': '珍珠', '青黛粉': '青黛', '朱砂粉': '朱砂',
    '蒲黄炭': '蒲黄', '地榆炭': '地榆', '栀子炭': '栀子', '荆芥炭': '荆芥',
    '杜仲炭': '杜仲', '棕榈炭': '棕榈', '侧柏炭': '侧柏叶', '姜炭': '干姜',
    '茅根炭': '白茅根', '贯众炭': '贯众', '大黄炭': '大黄', '黄芩炭': '黄芩',
    '黄柏炭': '黄柏', '牡丹皮炭': '牡丹皮', '藕节炭': '藕节', '炮姜': '干姜',
    '全瓜蒌': '瓜蒌',
}

def normalize_herb(h):
    if h in HERB_ALIASES:
        return HERB_ALIASES[h]
    for p in HERB_PREFIXES:
        if h.startswith(p) and len(h) - len(p) >= 2:
            base = h[len(p):]
            return HERB_ALIASES.get(base, base)
    return h

def extract_herbs(output):
    herbs = []
    for seg in HERB_SEG.findall(output):
        for m in HERB_RE.finditer(seg):
            herbs.append(normalize_herb(m.group(1)))
    return herbs

def dedup(xs):
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def main():
    os.makedirs(OUT, exist_ok=True)
    with open(SRC, encoding='utf-8') as f:
        data = json.load(f)[:N]

    all_sym, all_herb = [], []
    sym_freq, herb_freq = {}, {}
    for item in data:
        syms = extract_symptoms(item['input'])
        herbs = extract_herbs(item['output'])
        all_sym.append(syms)
        all_herb.append(herbs)
        for s in syms:
            sym_freq[s] = sym_freq.get(s, 0) + 1
        for h in herbs:
            herb_freq[h] = herb_freq.get(h, 0) + 1

    herb_vocab = sorted(herb_freq.keys(), key=lambda x: (-herb_freq[x], x))
    herb_set = set(herb_vocab)
    sym_vocab = sorted([s for s, c in sym_freq.items()
                        if c >= SYM_MIN_FREQ and s not in herb_set],
                       key=lambda x: (-sym_freq[x], x))
    sym2id = {s: i for i, s in enumerate(sym_vocab)}
    herb2id = {h: i for i, h in enumerate(herb_vocab)}

    with open(f'{OUT}/symptom_contains.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(sym_vocab) + '\n')
    with open(f'{OUT}/herbs_contains.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(herb_vocab) + '\n')

    sym_ids = [dedup([sym2id[s] for s in ss if s in sym2id]) for ss in all_sym]
    herb_ids = [dedup([herb2id[h] for h in hh]) for hh in all_herb]
    empty_sym = sum(1 for x in sym_ids if not x)
    empty_herb = sum(1 for x in herb_ids if not x)

    with open(f'{OUT}/pre_symptoms.txt', 'w', encoding='utf-8') as f:
        for ids in sym_ids:
            f.write(' '.join(map(str, ids)) + '\n')
    with open(f'{OUT}/pre_herbs.txt', 'w', encoding='utf-8') as f:
        for ids in herb_ids:
            f.write(' '.join(map(str, ids)) + '\n')

    rng = np.random.RandomState(SEED)
    n = len(data)
    perm = rng.permutation(n)
    n_train = int(n * 0.85)
    tr_idx = sorted(perm[:n_train].tolist())
    te_idx = sorted(perm[n_train:].tolist())

    def write_split(name, ids_list, idx_list):
        with open(f'{OUT}/{name}', 'w', encoding='utf-8') as f:
            for i in idx_list:
                f.write(' '.join(map(str, ids_list[i])) + '\n')

    write_split('pre_symptoms_train.txt', sym_ids, tr_idx)
    write_split('pre_symptoms_test.txt', sym_ids, te_idx)
    write_split('pre_herbs_train.txt', herb_ids, tr_idx)
    write_split('pre_herbs_test.txt', herb_ids, te_idx)

    with open(f'{OUT}/prescript.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['symptom', 'herb'])
        for s, h in zip(sym_ids, herb_ids):
            w.writerow([json.dumps(s, ensure_ascii=False), json.dumps(h, ensure_ascii=False)])

    def gkey(s, h):
        return hashlib.md5((repr(sorted(s)) + '|' + repr(sorted(h))).encode()).hexdigest()

    k2i = {}
    g = np.empty(n, dtype=np.int64)
    for i, (s, h) in enumerate(zip(sym_ids, herb_ids)):
        g[i] = k2i.setdefault(gkey(s, h), len(k2i))
    u, inv = np.unique(g, return_inverse=True)
    order = np.argsort(inv, kind='stable')
    inv_s = inv[order]
    st = np.searchsorted(inv_s, np.arange(len(u)), side='left')
    en = np.searchsorted(inv_s, np.arange(len(u)), side='right')
    perm2 = np.random.RandomState(SEED).permutation(len(u))
    tr, va, te = [], [], []
    c = 0
    for gi in perm2:
        m = order[st[gi]:en[gi]]
        (tr if c < n * 0.6 else va if c < n * 0.8 else te).append(m)
        c += len(m)
    tr = np.concatenate(tr)
    va = np.concatenate(va)
    te = np.concatenate(te)
    np.savez(f'{OUT}/split_indices.npz',
             train_idx=tr.astype(np.int64), val_idx=va.astype(np.int64),
             test_idx=te.astype(np.int64), group_labels=g)
    print(f"split_indices(6:2:2): train={len(tr)} val={len(va)} test={len(te)} groups={len(k2i)}")
    print('done ->', os.path.abspath(OUT))

if __name__ == '__main__':
    main()
