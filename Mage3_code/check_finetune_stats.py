# 统计 finetune.json 的监督密度（非零自旋占比）与按元素的分布
import json
import numpy as np
from collections import defaultdict, Counter
from o3spin.elements import parse_element_symbol

FINETUNE_JSON_PATH = "/Users/hazecifer/Documents/Codeproject/Myprojects/25-9-9 MgGNN/Testment/train_toy.json"

with open(FINETUNE_JSON_PATH, "r") as f:
    data = json.load(f)

total_structs = len(data)
total_nodes = 0
valid_nodes = 0
sym_counter = Counter()
sym_valid = defaultdict(int)

for d in data:
    species = d["species"]
    spins = np.array(d.get("spins", [[0.0, 0.0, 0.0]] * len(species)), dtype=float)
    norms = np.linalg.norm(spins, axis=1)
    mask = norms > 1e-8

    total_nodes += len(species)
    valid_nodes += int(mask.sum())

    # 统计按元素的有效/总数（使用 notebook 同步的清洗策略）
    syms = [parse_element_symbol(s) or "UNK" for s in species]
    sym_counter.update(syms)
    for s, m in zip(syms, mask):
        if m:
            sym_valid[s] += 1

print(f"structures: {total_structs}")
print(f"nodes: total={total_nodes}, valid(spin!=0)={valid_nodes} ({valid_nodes/total_nodes*100:.2f}%)")

print("\nTop-20 元素分布（valid/total, 占比%）:")
for sym, tot in sym_counter.most_common(20):
    v = sym_valid.get(sym, 0)
    pct = v / tot * 100 if tot > 0 else 0.0
    print(f"  {sym:>3}: {v:5d}/{tot:5d} ({pct:5.1f}%)")
