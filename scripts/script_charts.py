
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------
EXCEL_FILE = ".././data/Benchmark_QML_CICIDS2017.xlsx"  
OUTPUT_DIR = "chart_output"

# ---------------------------------------------------------------------------
# LETTURA DATI DAL FILE EXCEL
# ---------------------------------------------------------------------------
def carica_foglio(nome_foglio, colonne_numeriche):
    df = pd.read_excel(EXCEL_FILE, sheet_name=nome_foglio, header=2)
    df = df.dropna(subset=colonne_numeriche) 
    # converti a numerico (per sicurezza: qualche cella potrebbe essere stringa)
    for c in colonne_numeriche:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=colonne_numeriche)
    return df.reset_index(drop=True)


print(f"Lettura del file '{EXCEL_FILE}'...")

# Nei fogli VQC le colonne sono: Test, Seed, Sample, Qubit, Max Iter,
# FM Reps, Ansatz Reps, Precision, Recall, F1-Score, Accuracy, Time (s), Note
VQC_EFF  = carica_foglio("VQC efficient_su2",  ["Qubit", "Sample", "F1-Score"])
VQC_REAL = carica_foglio("VQC real_amplitudes",["Qubit", "Sample", "F1-Score"])
REUP     = carica_foglio("VQC Re-Upload",      ["Qubit", "Sample", "F1-Score"])

# QSVC: Test, Seed, Sample, N_train, N_test, Qubit, FM Reps, Precision, Recall, ...
QSVC     = carica_foglio("QSVC", ["Qubit", "Sample", "F1-Score", "N_test"])

# Random Forest: Test, Variante, Sample, PCA / Feat, Precision, Recall, F1-Score, ...
RF       = carica_foglio("Random Forest", ["PCA / Feat", "Sample", "F1-Score"])

print(f"  VQC efficient_su2:   {len(VQC_EFF)} test")
print(f"  VQC real_amplitudes: {len(VQC_REAL)} test")
print(f"  VQC Re-Upload:       {len(REUP)} test")
print(f"  QSVC:                {len(QSVC)} test")
print(f"  Random Forest:       {len(RF)} test")

# Rinomino per comodità le colonne più usate
for df in [VQC_EFF, VQC_REAL, REUP, QSVC]:
    df.rename(columns={"Qubit": "qubit", "Sample": "size",
                       "F1-Score": "F1", "Accuracy": "Acc",
                       "Time (s)": "time", "Test": "test"}, inplace=True)

QSVC.rename(columns={"N_train": "ntr", "N_test": "ntest", "FM Reps": "fm_reps"}, inplace=True)
RF.rename(columns={"Sample": "size", "PCA / Feat": "pca",
                   "F1-Score": "F1", "Test": "test", "Variante": "variante"}, inplace=True)

# ---------------------------------------------------------------------------
# STILE GRAFICI
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLOR = {
    "VQC efficient_su2":  "#5B9BD5",
    "VQC real_amplitudes":"#9DC3E6",
    "QSVC":               "#70AD47",
    "VQC Re-Upload":      "#7030A0",
    "Random Forest":      "#ED7D31",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig, name):
    fig.savefig(f"{OUTPUT_DIR}/{name}.png")
    fig.savefig(f"{OUTPUT_DIR}/{name}.pdf")
    print(f"  ✓ {OUTPUT_DIR}/{name}.png + .pdf")


print("\nGenerazione grafici...")

# ---------------------------------------------------------------------------
# GRAFICO 1 — F1 vs Qubit (miglior test per qubit, per tipologia)
# ---------------------------------------------------------------------------
def best_per_key(df, key_col, val_col="F1"):
    return df.loc[df.groupby(key_col)[val_col].idxmax()].sort_values(key_col)

vqc_best  = best_per_key(VQC_EFF,  "qubit")
real_best = best_per_key(VQC_REAL, "qubit")
qsvc_best = best_per_key(QSVC,     "qubit")
rf_best   = best_per_key(RF,       "pca")

fig, ax = plt.subplots(figsize=(11, 6))
qubits = sorted(set(vqc_best["qubit"]) | set(real_best["qubit"]) |
                set(qsvc_best["qubit"]) | set(rf_best["pca"]))
x = np.arange(len(qubits))
width = 0.2

def get_vals(df, qcol, val_col="F1"):
    return [df.loc[df[qcol] == q, val_col].values[0] if (df[qcol] == q).any() else np.nan for q in qubits]

ax.bar(x - 1.5*width, get_vals(vqc_best,  "qubit"), width,
       label="VQC efficient_su2",   color=COLOR["VQC efficient_su2"])
ax.bar(x - 0.5*width, get_vals(real_best, "qubit"), width,
       label="VQC real_amplitudes", color=COLOR["VQC real_amplitudes"])
ax.bar(x + 0.5*width, get_vals(qsvc_best, "qubit"), width,
       label="QSVC",                color=COLOR["QSVC"])
ax.bar(x + 1.5*width, get_vals(rf_best,   "pca"),   width,
       label="Random Forest (baseline)", color=COLOR["Random Forest"])

ax.set_xticks(x)
ax.set_xticklabels([f"{int(q)}" for q in qubits])
ax.set_xlabel("Numero di qubit / componenti PCA")
ax.set_ylabel("F1-Score (miglior test per qubit)")
ax.set_title("Miglior F1-Score per numero di qubit, confronto tra tipologie")
ax.set_ylim(0.60, 1.02)
ax.legend(loc="lower right", frameon=True)
rf_20 = RF[RF["pca"] == 20]["F1"].max() if (RF["pca"] == 20).any() else None
if rf_20:
    ax.axhline(rf_20, color="gray", linestyle=":", alpha=0.6)
    ax.text(len(qubits)-0.6, rf_20 + 0.005,
            f"RF 20 feat (max) ≈ {rf_20:.3f}",
            fontsize=9, color="gray", style="italic")
save(fig, "F1_per_qubit"); plt.close(fig)

# ---------------------------------------------------------------------------
# GRAFICO 2 — F1 vs Sample Size
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))

for df, label, marker in [(VQC_EFF,  "VQC efficient_su2",   "o"),
                          (VQC_REAL, "VQC real_amplitudes", "s"),
                          (QSVC,     "QSVC",                "^")]:
    gb = df.groupby("size")["F1"].max().reset_index()
    ax.plot(gb["size"], gb["F1"], marker=marker, linewidth=2, markersize=9,
            label=label, color=COLOR[label])

rf_by_size = RF.groupby("size")["F1"].max().reset_index()
ax.plot(rf_by_size["size"], rf_by_size["F1"], marker="D", linewidth=2, markersize=8,
        label="Random Forest (best)", color=COLOR["Random Forest"], linestyle="--")

ax.set_xlabel("Sample size")
ax.set_ylabel("F1-Score (miglior test per size)")
ax.set_title("Scaling: F1-Score in funzione della dimensione del flusso")
all_sizes = sorted(set(VQC_EFF["size"]) | set(VQC_REAL["size"]) |
                   set(QSVC["size"]) | set(RF["size"]))
ax.set_xticks(all_sizes)
ax.legend(loc="lower right", frameon=True)
ax.set_ylim(0.60, 1.02)
save(fig, "F1_vs_sample_size"); plt.close(fig)

# ---------------------------------------------------------------------------
# GRAFICO 3 — F1 vs Training Time (log x)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6.5))

for df, label, marker in [(VQC_EFF,  "VQC efficient_su2",   "o"),
                          (VQC_REAL, "VQC real_amplitudes", "s"),
                          (QSVC,     "QSVC",                "^"),
                          (REUP,     "VQC Re-Upload",       "P")]:
    df_plot = df.dropna(subset=["time"])
    ax.scatter(df_plot["time"], df_plot["F1"], s=90, alpha=0.75, label=label,
               color=COLOR[label], edgecolor="black", linewidth=0.4, marker=marker)

# Random Forest: tempi ~0.1-0.2 s (colonna "Time (s)")
rf_times = RF["Time (s)"] if "Time (s)" in RF.columns else pd.Series([0.15]*len(RF))
ax.scatter(rf_times, RF["F1"], s=70, alpha=0.6, label="Random Forest",
           color=COLOR["Random Forest"], edgecolor="black", linewidth=0.4, marker="D")

# Annotazioni sui best: prendi automaticamente il miglior F1 di ogni gruppo
best_vqc_row  = VQC_EFF.loc[VQC_EFF["F1"].idxmax()]
best_qsvc_row = QSVC.loc[QSVC["F1"].idxmax()]
for row, offset in [(best_qsvc_row, (10, 10)), (best_vqc_row, (10, -20))]:
    ax.annotate(f"★ {row['test']}", xy=(row["time"], row["F1"]),
                xytext=offset, textcoords="offset points",
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

ax.set_xscale("log")
ax.set_xlabel("Tempo di training (s, scala log)")
ax.set_ylabel("F1-Score")
ax.set_title("Trade-off: F1-Score vs tempo di training")
ax.legend(loc="lower right", frameon=True)
ax.set_ylim(0.60, 1.02)
save(fig, "F1_vs_tempo"); plt.close(fig)


# ---------------------------------------------------------------------------
# GRAFICO 5 — Sintesi: miglior modello per tipologia
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Prendo automaticamente il migliore di ciascuno
rf_best_row   = RF.loc[RF["F1"].idxmax()]
qsvc_best_row = QSVC.loc[QSVC["F1"].idxmax()]
vqc_best_row  = VQC_EFF.loc[VQC_EFF["F1"].idxmax()]
reup_best_row = REUP.loc[REUP["F1"].idxmax()]

models = [
    f"RF\n({rf_best_row['variante']})",
    f"QSVC\n({qsvc_best_row['test']})",
    f"VQC eff_su2\n({vqc_best_row['test']})",
    f"VQC Re-Up\n({reup_best_row['test']})",
]
f1_vals = [rf_best_row["F1"], qsvc_best_row["F1"],
           vqc_best_row["F1"], reup_best_row["F1"]]
time_vals = [rf_best_row.get("Time (s)", 0.2), qsvc_best_row["time"],
             vqc_best_row["time"], reup_best_row["time"]]
colors = [COLOR["Random Forest"], COLOR["QSVC"],
          COLOR["VQC efficient_su2"], COLOR["VQC Re-Upload"]]

b1 = ax1.bar(models, f1_vals, color=colors, edgecolor="black", linewidth=0.5)
ax1.set_ylabel("F1-Score")
ax1.set_title("Miglior F1-Score per tipologia")
ax1.set_ylim(min(f1_vals) - 0.03, 1.02)
for bar, val in zip(b1, f1_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.003,
             f"{val:.3f}", ha="center", fontweight="bold")

b2 = ax2.bar(models, time_vals, color=colors, edgecolor="black", linewidth=0.5)
ax2.set_ylabel("Tempo di training (s, scala log)")
ax2.set_yscale("log")
ax2.set_title("Tempo di training del miglior modello")
for bar, val in zip(b2, time_vals):
    label = f"{val:.1f}s" if val < 60 else f"{val/60:.1f}min"
    ax2.text(bar.get_x() + bar.get_width()/2, val * 1.4,
             label, ha="center", fontweight="bold")

fig.suptitle("Sintesi finale: qualità vs costo computazionale", fontsize=14, y=1.02)
save(fig, "sintesi_finale"); plt.close(fig)

# ---------------------------------------------------------------------------
# GRAFICO 6 — Errori totali QSVC (FP+FN)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.5))

qsvc_plot = QSVC.copy()
if "Errori Tot. (FP+FN)" in qsvc_plot.columns:
    qsvc_plot["errori"] = pd.to_numeric(qsvc_plot["Errori Tot. (FP+FN)"], errors="coerce").astype(int)
else:
    qsvc_plot["errori"] = ((1 - qsvc_plot["Acc"]) * qsvc_plot["ntest"]).round().astype(int)

qsvc_plot = qsvc_plot.sort_values(["qubit", "size", "fm_reps"]).reset_index(drop=True)
qsvc_plot["label"] = (qsvc_plot["test"].astype(str) + "\n" +
                     qsvc_plot["qubit"].astype(int).astype(str) + "q, " +
                     qsvc_plot["size"].astype(int).astype(str))

cmap = {6: "#B4DDB4", 8: "#70AD47", 10: "#4B8B2E", 12: "#2E5A1C"}
colors_q = [cmap.get(int(q), "#888888") for q in qsvc_plot["qubit"]]

bars = ax.bar(range(len(qsvc_plot)), qsvc_plot["errori"],
              color=colors_q, edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(qsvc_plot)))
ax.set_xticklabels(qsvc_plot["label"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Errori totali (FP + FN)")
ax.set_title("QSVC: errori di classificazione per test")

for bar, v in zip(bars, qsvc_plot["errori"]):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3,
            str(v), ha="center", fontweight="bold", fontsize=10)

qubits_present = set(int(q) for q in qsvc_plot["qubit"])
legend_el = [Patch(color=c, label=f"{q} qubit") for q, c in cmap.items() if q in qubits_present]
ax.legend(handles=legend_el, loc="upper left", frameon=True)
save(fig, "errori_QSVC"); plt.close(fig)

# ---------------------------------------------------------------------------
# GRAFICO 7 — A parita di qubit come varia con  le rep di ansatz
# ----------------------------------------  -----------------------------------
array = [("#0A3761","4 Qubit"), ("#CA3F7B","6 Qubit"), ("#70AD47","8 Qubit"), ("#7030A0","10 Qubit"), ("#ED7D31","12 Qubit")]

fig, ax = plt.subplots(figsize=(11, 6))

vqc_plot = VQC_EFF.copy()
vqc_plot = vqc_plot.sort_values(["qubit", "Ansatz Reps"]).reset_index(drop=True)

for i, qubit in enumerate(sorted(vqc_plot["qubit"].unique())):
    gb = vqc_plot[vqc_plot["qubit"] == qubit]
    gb_max_f1 = gb.loc[gb.groupby("Ansatz Reps")["F1"].idxmax()]  
    
    ax.plot(gb_max_f1["Ansatz Reps"], gb_max_f1["F1"], marker="o", linewidth=2, markersize=9, 
            label=array[i][1], color=array[i][0])

# Impostazione delle etichette
ax.set_xlabel("Numero di repliche di Ansatz (Ansatz Reps)")
ax.set_ylabel("F1-Score")
ax.set_ylim(0.7 , 1)  
ax.set_title("VQC efficient_su2: F1-Score massimo in funzione delle repliche di Ansatz per qubit") 

# Aggiungi la legenda
ax.legend(loc="lower right", frameon=True)

# Salvataggio del grafico
save(fig, "ansatz_reps_per_qubit_max_f1_per_approccio")
plt.close(fig)  


# ---------------------------------------------------------------------------
# GRAFICO 8 — F1-Score in funzione della dimensione del sample size
# ----------------------------------------  -----------------------------------

array = [
    ("#0A3761", "4 Qubit"),
    ("#CA3F7B", "6 Qubit"),
    ("#70AD47", "8 Qubit"),
]

fig, ax = plt.subplots(figsize=(11, 6))

vqc_plot = VQC_EFF.copy()
vqc_plot = vqc_plot.sort_values(["qubit", "size"]).reset_index(drop=True)

for i, qubit in enumerate(sorted(vqc_plot["qubit"].unique())):
    gb = vqc_plot[vqc_plot["qubit"] == qubit]

    # Per ogni size prendo il miglior F1 a parità di qubit
    gb_max_f1 = gb.loc[gb.groupby("size")["F1"].idxmax()]
    gb_max_f1 = gb_max_f1.sort_values("size")

    ax.plot(
        gb_max_f1["size"],
        gb_max_f1["F1"],
        marker="o",
        linewidth=2.5,
        markersize=6,
        label=array[i][1],
        color=array[i][0]
    )

ax.set_xlabel("Size", fontweight="bold")
ax.set_ylabel("F1", fontweight="bold")
ax.set_title("VQC efficient_su2: F1-Score in funzione della size a parità di qubit")

ax.set_ylim(0.70, 1.00)
ax.set_yticks(np.arange(0.70, 1.01, 0.05))

ax.grid(axis="y", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=5,
    frameon=False
)

plt.tight_layout()
save(fig, "F1_vs_size_per_qubit")
plt.close(fig)
# ---------------------------------------------------------------------------
# GRAFICO 9 — VQC efficient_su2: Precision e Recall a confronto
#              (mostra dove il modello sbaglia: FP vs FN)
# ---------------------------------------------------------------------------

# Assicuriamoci che le metriche siano numeriche
for c in ["Precision", "Recall"]:
    VQC_EFF[c] = pd.to_numeric(VQC_EFF[c], errors="coerce")

colori_qubit = {
    4:  "#0A3761",
    6:  "#CA3F7B",
    8:  "#70AD47",
    10: "#7030A0",
    12: "#ED7D31",
}

fig, (axP, axR) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

vqc_plot = VQC_EFF.copy().sort_values(["qubit", "size"])

for qubit in sorted(vqc_plot["qubit"].unique()):
    gb = vqc_plot[vqc_plot["qubit"] == qubit]

    # Per ogni size, prendo la configurazione con F1 massimo (stesso criterio dei grafici precedenti)
    idx = gb.groupby("size")["F1"].idxmax()
    best = gb.loc[idx].sort_values("size")

    color = colori_qubit.get(int(qubit), "#888888")
    label = f"{int(qubit)} qubit"

    # Evidenzio lo sweet spot 6q
    lw = 3.0 if qubit == 6 else 2.0
    ms = 10  if qubit == 6 else 7
    alpha = 1.0 if qubit == 6 else 0.75

    axP.plot(best["size"], best["Precision"], marker="o",
             linewidth=lw, markersize=ms, alpha=alpha, color=color, label=label)
    axR.plot(best["size"], best["Recall"], marker="s",
             linewidth=lw, markersize=ms, alpha=alpha, color=color, label=label)

# --- Pannello Precision (FP) ---
axP.set_title("Precision  →  pochi falsi positivi (FP)", fontweight="bold")
axP.set_xlabel("Sample size")
axP.set_ylabel("Valore della metrica")
axP.set_ylim(0.70, 1.02)
axP.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
axP.legend(loc="lower right", frameon=True, title="Configurazione")

# --- Pannello Recall (FN) ---
axR.set_title("Recall  →  pochi falsi negativi (FN)", fontweight="bold")
axR.set_xlabel("Sample size")
axR.set_ylim(0.70, 1.02)
axR.axhline(1.0, color="gray", linestyle=":", alpha=0.5)

fig.suptitle(
    "VQC efficient_su2: dove sbaglia il modello? Precision vs Recall al variare di qubit e size",
    fontsize=13, fontweight="bold", y=1.02
)

plt.tight_layout()
save(fig, "VQC_eff_precision_recall_per_qubit")
plt.close(fig)