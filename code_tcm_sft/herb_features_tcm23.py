# -*- coding: utf-8 -*-
import json
import re
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = r'D:\project\PycharmProjects\STRSM_baseline'

PROPERTY_LABELS = ["寒", "热", "温", "凉", "平",
                   "酸", "苦", "甘", "辛", "咸",
                   "肺", "心包", "心", "大肠", "三焦", "小肠",
                   "胃", "胆", "膀胱", "脾", "肝", "肾", "毒"]
PROPERTY_MAPPING = {
    "寒": [("大寒", 2), ("微寒", 0.5), ("寒", 1)],
    "热": [("大热", 2), ("微热", 0.5), ("热", 1)],
    "温": [("微温", 0.5), ("温", 1)],
    "凉": [("微凉", 0.5), ("凉", 1)],
    "平": [("平", 1), ("微平", 0.5)],
    "酸": [("微酸", 0.5), ("酸", 1)],
    "苦": [("微苦", 0.5), ("苦", 1)],
    "甘": [("微甘", 0.5), ("甘", 1)],
    "辛": [("微辛", 0.5), ("辛", 1)],
    "咸": [("微咸", 0.5), ("咸", 1)],
    "肺": [], "心包": [], "心": [], "大肠": [], "三焦": [],
    "小肠": [], "胃": [], "胆": [], "膀胱": [], "脾": [],
    "肝": [], "肾": [],
    "毒": [("大毒", 2), ("有毒", 1), ("小毒", 0.5)],
}
ALIAS = {
    '附片': '附子', '淡附片': '附子', '天雄': '附子',
    '苏梗': '紫苏梗', '菖蒲': '石菖蒲', '粉防己': '防己', '木防己': '防己',
    '燀苦杏仁': '苦杏仁', '燀桃仁': '桃仁', '枯芩': '黄芩',
    '苁蓉': '肉苁蓉', '淮小麦': '小麦', '首乌': '何首乌',
    '粉萆薢': '萆薢', '绵萆薢': '萆薢', '龟板': '龟甲', '龟板胶': '龟甲胶',
    '佛片': '佛手', '银杏': '白果', '白果仁': '白果', '银杏果': '白果',
    '朱茯苓': '茯苓', '蜀椒': '花椒', '艾叶炭': '艾叶', '茜草炭': '茜草',
    '姜皮': '生姜', '当归尾': '当归', '当归头': '当归', '当归身': '当归',
    '老鹤草': '老鹳草', '灯芯草': '灯心草', '阿胶珠': '阿胶', '龙血竭': '血竭',
    '桂心': '肉桂', '紫葳': '凌霄花', '丹皮': '牡丹皮', '人参片': '人参',
    '全参': '人参', '米党参': '党参', '仙半夏': '半夏', '半夏曲': '半夏',
    '川贝母粉': '川贝母', '槐米': '槐花', '石硫磺': '硫磺', '粉葛': '葛根',
    '乌梅肉': '乌梅', '仙灵脾': '淫羊藿', '山楂炭': '山楂',
    '山甲': '穿山甲', '山甲粉': '穿山甲', '忍冬藤炭': '忍冬藤',
    '朱连翘': '连翘', '橘皮': '陈皮', '甘草稍': '甘草', '番泻叶粉': '番泻叶',
    '臭梧桐叶': '臭梧桐', '苇茎': '芦根', '诃子肉': '诃子',
    '豆豉': '淡豆豉', '鳖血柴胡': '柴胡', '黄芪粉': '黄芪',
    '建曲': '神曲', '六神曲': '神曲', '瓜蒌仁': '瓜蒌', '贝母': '川贝母',
    '鳖虫': '土鳖虫', '苡仁': '薏苡仁', '千斤拨': '千斤拔',
    '地黄': '生地黄', '山茱萸': '山茱萸肉', '生甘草': '甘草', '炙甘草': '甘草',
}
FORCE_MEAN = {'白花蛇', '藤梨根'}

MANUAL23 = {
    '萹蓄':   {'寒': 0.5, '苦': 1, '膀胱': 1},
    '甜叶菊': {'甘': 1, '凉': 1},
    '胡麻仁': {'甘': 1, '平': 1, '肝': 1, '肾': 1, '大肠': 1},
    '粳米':   {'甘': 1, '平': 1, '脾': 1, '胃': 1},
    '蜀漆':   {'苦': 1, '辛': 1, '温': 1, '肝': 1, '心包': 1, '毒': 1},
    '宽筋藤': {'苦': 0.5, '凉': 1, '肝': 1},
    '牛大力': {'甘': 1, '平': 1, '肺': 1, '肾': 1},
    '红曲':   {'甘': 1, '温': 1, '肝': 1, '脾': 1, '大肠': 1},
    '金箔':   {'辛': 1, '平': 1, '心': 1, '肝': 1},
    '梓白皮': {'苦': 1, '寒': 1, '胆': 1},
    '石硫磺': {'酸': 1, '温': 1, '肾': 1, '大肠': 1, '毒': 1},
    '石蜜':   {'甘': 1, '平': 1, '肺': 1, '脾': 1, '大肠': 1},
    '红芪':   {'甘': 1, '温': 0.5, '脾': 1, '肺': 1},
    '蛴螬':   {'咸': 1, '温': 0.5, '肝': 1, '毒': 1},
    '功劳叶': {'苦': 1, '凉': 1, '肝': 1, '肾': 1},
    '千斤拨': {'甘': 1, '平': 1, '肝': 1, '肾': 1, '脾': 1},
    '千斤拔': {'甘': 1, '平': 1, '肝': 1, '肾': 1, '脾': 1},
    '叶下珠': {'苦': 0.5, '凉': 1, '肝': 1},
    '喜树果': {'苦': 1, '寒': 1, '胃': 1, '毒': 1},
    '山姜':   {'辛': 1, '温': 1, '肺': 1, '胃': 1},
    '布渣叶': {'酸': 0.5, '凉': 1, '脾': 1, '胃': 1},
    '松叶':   {'苦': 1, '温': 1},
    '独一味': {'甘': 1, '苦': 1, '平': 1, '肝': 1},
    '田基黄': {'甘': 1, '苦': 0.5, '凉': 1, '肝': 1, '胆': 1},
    '百药煎': {'酸': 1, '甘': 1, '平': 1, '肺': 1, '胃': 1},
    '营实':   {'酸': 1, '凉': 1, '肝': 1},
    '葎草':   {'甘': 1, '苦': 1, '寒': 1, '肺': 1},
    '金雀根': {'甘': 1, '辛': 0.5, '平': 1, '脾': 1, '肺': 1},
    '黍米':   {'甘': 1, '平': 1, '脾': 1, '胃': 1},
    '白花蛇': {'甘': 1, '咸': 1, '温': 1, '肝': 1, '毒': 1},
    '藤梨根': {'酸': 1, '甘': 0.5, '凉': 1, '肝': 1, '胆': 1},
    '食盐':   {'咸': 1, '寒': 1, '胃': 1, '肾': 1},
    '黄酒':   {'辛': 1, '甘': 1, '温': 1, '肝': 1},
    '醋':     {'酸': 1, '苦': 1, '温': 1, '肝': 1, '胃': 1},
    '红糖':   {'甘': 1, '温': 1, '脾': 1, '胃': 1, '肝': 1},
    '猪蹄':   {'甘': 1, '咸': 1, '平': 1, '胃': 1},
}

def clean(s):
    return re.sub(r'[，。；、\s]', '', str(s))

def parse_property(values, prop_type):
    if prop_type == "毒":
        for level, score in PROPERTY_MAPPING["毒"]:
            if level in values:
                return score
        return 0
    if not PROPERTY_MAPPING[prop_type]:
        return 1 if prop_type in values else 0
    for level, score in PROPERTY_MAPPING[prop_type]:
        if level in values:
            return score
    return 1 if prop_type in values else 0

def herb_vec(herb):
    flavors = clean(herb.get("性味", ""))
    channels = clean(herb.get("归经", ""))
    toxicity = clean(herb.get("毒性", ""))
    v = [parse_property(flavors, p) for p in PROPERTY_LABELS[:10]]
    v += [parse_property(channels, p) for p in PROPERTY_LABELS[10:22]]
    v.append(parse_property(toxicity, "毒"))
    return v

info = json.load(open(f'{ROOT}/data/herbs_info.json', encoding='utf-8'))
name2herb = {clean(it["药名"]): it for it in info}
all_names = list(name2herb)
tcm_herbs = [l.strip() for l in open(f'{ROOT}/data_TCM/herbs_contains.txt', encoding='utf-8') if l.strip()]

mean_vec = None
vectors, exact, aliased, substr, failed = [], [], [], [], []
resolved = {}
for name in tcm_herbs:
    if name in name2herb:
        resolved[name] = name2herb[name]
        exact.append(name)
        continue
    base = ALIAS.get(name)
    if base and base in name2herb:
        resolved[name] = name2herb[base]
        aliased.append(f'{name}->{base}')
        continue
    cands = [n for n in all_names if n and (n in name or name in n)] if name not in FORCE_MEAN else []
    if len(cands) == 1:
        resolved[name] = name2herb[cands[0]]
        substr.append(f'{name}~{cands[0]}')
    elif len(cands) > 1:
        best = min(cands, key=len)
        resolved[name] = name2herb[best]
        substr.append(f'{name}~{best} (ambiguous, {len(cands)} candidates)')
    else:
        failed.append(name)

for name in tcm_herbs:
    vectors.append(herb_vec(resolved[name]) if name in resolved else [0.0] * 23)
vectors = np.array(vectors, dtype=np.float32)
nz_mean = vectors[vectors.sum(axis=1) > 0].mean(axis=0)
manual_used, mean_used = [], []
for i, name in enumerate(tcm_herbs):
    if name in MANUAL23:
        row = [0.0] * 23
        for k, val in MANUAL23[name].items():
            row[PROPERTY_LABELS.index(k)] = float(val)
        vectors[i] = row
        manual_used.append(name)
    elif name in failed:
        vectors[i] = nz_mean
        mean_used.append(name)
    elif vectors[i].sum() == 0:
        vectors[i] = nz_mean
        mean_used.append(name)

np.save(f'{ROOT}/data_TCM/herb_attr23.npy', vectors)
