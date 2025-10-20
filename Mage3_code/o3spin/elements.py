import re

# 元素→原子序数映射（1..118）
_PERIODIC = [
    "H","He","Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
    "Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn",
    "Nh","Fl","Mc","Lv","Ts","Og"
]
ELEMENT_Z = {s: i+1 for i, s in enumerate(_PERIODIC)}

UNK_INDEX = 0  # 保留 0 号给未知物种；正常元素索引仍用其原子序数(1..118)

ELEMENT_ALIASES = {
    "D": "H", "T": "H",          # 同位素->氢
    "1H": "H", "2H": "H", "3H": "H",
}

def parse_element_symbol(s: str) -> str | None:
    s = str(s).strip()
    if s in ELEMENT_Z:
        return s
    if s in ELEMENT_ALIASES:
        return ELEMENT_ALIASES[s]
    m = re.search(r'([A-Z][a-z]?)', s)  # 提取第一个元素片段
    if m:
        sym = m.group(1)
        if sym in ELEMENT_Z:
            return sym
    return None  # 返回 None 交由策略处理

def map_species_to_Z(species_list, unknown_policy="unk"):
    out = []
    for s in species_list:
        sym = parse_element_symbol(s)
        if sym is not None:
            out.append(ELEMENT_Z[sym])
        else:
            if unknown_policy == "unk":
                out.append(UNK_INDEX)  # 未知→0
            elif unknown_policy == "error":
                raise KeyError(f"Unrecognized species token '{s}'")
            else:
                out.append(UNK_INDEX)
    return out

# 确保 embedding 尺寸覆盖索引 0..118
NUM_ELEMENTS = max(ELEMENT_Z.values()) + 1  # 119
