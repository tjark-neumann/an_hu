import os, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

CSV_PATH = "./code_set_historical_unity.csv"
OUT_DIR  = "figures"
TAB_DIR  = "tables"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

def load_codeset(csv_path: str) -> pd.DataFrame:
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample)
        sep = dialect.delimiter
    except Exception:
        sep = ";"

    df = pd.read_csv(csv_path, sep=sep, engine="python")

    col_map = {c.lower(): c for c in df.columns}
    lower = {c: c.lower() for c in df.columns}
    df = df.rename(columns={lower.get(c, c): c for c in df.columns})
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ["document_name", "document", "doc", "doc_name"]:
            rename[c] = "document_name"
        elif lc in ["code", "codes"]:
            rename[c] = "code"
        elif lc in ["segment", "text", "excerpt"]:
            rename[c] = "segment"
    df = df.rename(columns=rename)
    
    for need in ["document_name", "code", "segment"]:
        if need not in df.columns:
            raise ValueError(f"Missing required column: {need}")

    df["document_name"] = df["document_name"].astype(str).str.strip().str.lower()
    df["code"] = df["code"].astype(str).str.strip().str.lower()
    df["segment"] = df["segment"].astype(str).str.strip()
    return df

df = load_codeset(CSV_PATH)

df.head()

total_by_code = df.groupby("code", dropna=False).size().sort_values(ascending=False).rename("count")
total_by_doc = df.groupby("document_name", dropna=False).size().rename("count")

code_doc = df.pivot_table(index="code", columns="document_name", aggfunc="size", fill_value=0)
percent_by_doc = code_doc.divide(code_doc.sum(axis=0), axis=1) * 100

total_by_code.to_csv(f"{TAB_DIR}/total_by_code.csv")
total_by_doc.to_csv(f"{TAB_DIR}/total_by_document.csv")
code_doc.to_csv(f"{TAB_DIR}/code_counts_by_document.csv")
percent_by_doc.to_csv(f"{TAB_DIR}/code_percent_by_document.csv")

total_by_code.head(), total_by_doc, code_doc.head(), percent_by_doc.head()

plt.figure(figsize=(9, 5.5))
ax = total_by_code.plot(kind="bar")
ax.set_title("Code Frequency Across Corpus")
ax.set_xlabel("Code")
ax.set_ylabel("Count of Coded Segments")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(f"{OUT_DIR}/fig1_code_frequency.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig1_code_frequency.svg", bbox_inches="tight")
plt.show()

preferred = [c for c in ["address", "historical_unity"] if c in code_doc.columns]
others = [c for c in code_doc.columns if c not in preferred]
code_doc_ord = code_doc[preferred + others] if preferred else code_doc

plt.figure(figsize=(10.5, 6))
x = np.arange(len(code_doc_ord.index))
width = 0.8 / max(1, len(code_doc_ord.columns))

for i, col in enumerate(code_doc_ord.columns):
    plt.bar(x + i*width, code_doc_ord[col].values, width=width, label=col)

plt.title("Code Frequency by Document")
plt.xlabel("Code")
plt.ylabel("Count of Coded Segments")
plt.xticks(x + (len(code_doc_ord.columns)-1)*width/2, code_doc_ord.index, rotation=0)
plt.legend(frameon=False, title="Document")
plt.tight_layout()

plt.savefig(f"{OUT_DIR}/fig2_code_by_document_grouped.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig2_code_by_document_grouped.svg", bbox_inches="tight")
plt.show()

for doc in code_doc_ord.columns:
    series = code_doc_ord[doc].copy()
    series = series[series > 0].sort_values(ascending=False)

    plt.figure(figsize=(7, 7))
    plt.pie(series.values, labels=series.index, autopct="%1.0f%%", startangle=90)
    plt.title(f"Composition of Codes — {doc}")
    plt.tight_layout()

    safe_doc = str(doc).replace(" ", "_")
    plt.savefig(f"{OUT_DIR}/fig_comp_{safe_doc}_pie.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{OUT_DIR}/fig_comp_{safe_doc}_pie.svg", bbox_inches="tight")
    plt.show()

codes = sorted(df["code"].dropna().unique().tolist())
idx = {c:i for i,c in enumerate(codes)}
co = np.zeros((len(codes), len(codes)), dtype=int)

for seg, g in df.groupby("segment"):
    codes_here = sorted(set(g["code"].dropna()))
    if len(codes_here) >= 2:
        for a, b in combinations(codes_here, 2):
            ia, ib = idx[a], idx[b]
            co[ia, ib] += 1
            co[ib, ia] += 1

plt.figure(figsize=(8.8, 7.2))
im = plt.imshow(co, aspect="auto")
plt.title("Code Co-occurrence Heatmap (Same Segment)")
plt.xlabel("Code")
plt.ylabel("Code")
plt.xticks(ticks=np.arange(len(codes)), labels=codes, rotation=90)
plt.yticks(ticks=np.arange(len(codes)), labels=codes)
cbar = plt.colorbar(im)
cbar.set_label("Co-occurrence Count")
plt.tight_layout()

plt.savefig(f"{OUT_DIR}/fig5_code_cooccurrence_heatmap.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig5_code_cooccurrence_heatmap.svg", bbox_inches="tight")
plt.show()

df["segment_id"] = df.groupby("document_name").cumcount() + 1
pivot = pd.crosstab(df["segment_id"], df["code"])
pivot.plot.area(figsize=(12,6))
plt.title("Distribution of Codes Across Text Progression")
plt.xlabel("Segment order")
plt.ylabel("Count")
plt.savefig(f"{OUT_DIR}/fig6_distribution_of_codes.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig6_distribution_of_codes.svg", bbox_inches="tight")
plt.show()

percent_by_doc.T.plot(kind="bar", stacked=True, figsize=(10,6))
plt.title("Relative Composition of Codes by Document")
plt.ylabel("% of coded segments")
plt.legend(bbox_to_anchor=(1,1))
plt.savefig(f"{OUT_DIR}/fig7_relative_composition_of_codes.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig7_relative_composition_of_codes.svg", bbox_inches="tight")
plt.show()

import networkx as nx

G = nx.Graph()
for i, a in enumerate(codes):
    for j, b in enumerate(codes):
        if i < j and co[i,j] > 0:
            G.add_edge(a, b, weight=co[i,j])

plt.figure(figsize=(8,8))
pos = nx.spring_layout(G, seed=42, k=0.5)
nx.draw_networkx_nodes(G, pos, node_size=800, node_color="lightblue")
nx.draw_networkx_edges(G, pos, width=[d['weight']*0.2 for _,_,d in G.edges(data=True)], alpha=0.6)
nx.draw_networkx_labels(G, pos, font_size=10)
plt.title("Code Co-occurrence Network")
plt.axis("off")
plt.savefig(f"{OUT_DIR}/fig8_co_occurrence.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig8_co_occurrence.svg", bbox_inches="tight")
plt.show()

from scipy.cluster.hierarchy import linkage, dendrogram

linkage_matrix = linkage(co, method="ward")
plt.figure(figsize=(10,5))
dendrogram(linkage_matrix, labels=codes, leaf_rotation=90)
plt.title("Hierarchical Clustering of Codes (by Co-occurrence)")
plt.savefig(f"{OUT_DIR}/fig9_clustering.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig9_clustering.svg", bbox_inches="tight")
plt.show()

!pip install wordcloud ##if ness

##redid follwing part for simplicity

import os, csv, re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

CSV_PATH = "code_set_historical_unity.csv"
OUT_BASE = "figures_useful_impressive"
WC_DIR   = os.path.join(OUT_BASE, "wordclouds")
TOP_DIR  = os.path.join(OUT_BASE, "top20_words")
os.makedirs(WC_DIR, exist_ok=True)
os.makedirs(TOP_DIR, exist_ok=True)

STOP = set([
    "the","and","for","that","with","this","from","are","was","were","have","has",
    "not","but","you","your","their","they","them","its","our","his","her","she","him",
    "into","over","under","than","then","also","about","there","here","been","being"
])

def load_codeset(csv_path: str) -> pd.DataFrame:
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample)
        sep = dialect.delimiter
    except Exception:
        sep = ";"
    df = pd.read_csv(csv_path, sep=sep, engine="python")
    
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ["document_name","document","doc","doc_name"]:
            rename[c] = "document_name"
        elif lc in ["code","codes"]:
            rename[c] = "code"
        elif lc in ["segment","text","excerpt"]:
            rename[c] = "segment"
    df = df.rename(columns=rename)

    for req in ["document_name","code","segment"]:
        if req not in df.columns:
            raise ValueError(f"Missing required column: {req}")

    df["document_name"] = df["document_name"].astype(str).str.strip().str.lower()
    df["code"] = df["code"].astype(str).str.strip().str.lower()
    df["segment"] = df["segment"].astype(str).str.strip()
    return df

df = load_codeset(CSV_PATH)

def tokenize(text: str):
    return re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

def safe(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name)).strip("_").lower()

for code in sorted(df["code"].unique()):
    text = " ".join(df.loc[df["code"] == code, "segment"]).strip()
    if not text:
        continue

    wc = WordCloud(width=1200, height=600, background_color="white").generate(text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud — {code}")

    base = safe(code)
    plt.savefig(os.path.join(WC_DIR, f"wordcloud_{base}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(WC_DIR, f"wordcloud_{base}.svg"), bbox_inches="tight")
    plt.close()

for code in sorted(df["code"].unique()):
    text = " ".join(df.loc[df["code"] == code, "segment"])
    tokens = [t for t in tokenize(text) if t not in STOP]
    counts = Counter(tokens).most_common(20)
    if not counts:
        continue

    words, freqs = zip(*counts)
    plt.figure(figsize=(12, 6))
    plt.bar(words, freqs)
    plt.title(f"Top 20 words — {code}")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Frequency")
    plt.tight_layout()

    base = safe(code)
    plt.savefig(os.path.join(TOP_DIR, f"top20_{base}.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(TOP_DIR, f"top20_{base}.svg"), bbox_inches="tight")
    plt.close()

print("wc_dir", WC_DIR)
print("top_dir", TOP_DIR)

