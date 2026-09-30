"""
Reused final_submit/F3/code/F3X_0522/F3_combined_0522.py plotting layout.

All panel inputs are loaded from the checked current-manuscript tables under
assets/manuscript_figures/full/fig3_current_nm.

Combined Figure 3
Layout (3 outer rows):
  Row 0:  A (TE manifold)                |  B (3-model pred KDE, 3-stacked)
  Row 1:  C (UMAP 4-panel + cons strip)  |  D (Conceptual diagram)
  Row 2:  E (ablation UpSet)  |  F (MPRA bar)  |  G (benchmark bar)

Run from the repository root with the project environment, for example:
    python figures/fig3/reproduce_combined_checked.py
"""

import os, sys, re, pickle, glob, warnings, ast, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import seaborn as sns
from datetime import datetime
from scipy.stats import spearmanr
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# Style
# =============================================================================
def set_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.linewidth': 1.2,
        'font.size': 10,
        'axes.labelweight': 'bold',
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
    })

set_style()

def spine_off(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def panel_label(ax, letter, x=-0.13, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='top')

# =============================================================================
# Paths
# =============================================================================
ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(ROOT, 'assets/manuscript_figures/full/fig3_current_nm')
OUT_DIR  = os.path.join(ROOT, 'figures/figure_subplot_check/fig3')
F3A_DIR = F3B_DIR = F3C_DIR = F3E_DIR = F3F_DIR = UMAP_DIR = DATA_DIR
PANEL_C_REFERENCE_PNG = ''
os.makedirs(OUT_DIR, exist_ok=True)

CLADE_COLORS = {
    'Human':         '#b22222',
    'Mammals':       '#1f77b4',
    'Vertebrates':   '#17becf',
    'Invertebrates': '#2ca02c',
    'Fungi':         '#ff7f0e',
    'Unknown':       'gray',
}
CLADE_ORDER = ['Human', 'Mammals', 'Vertebrates', 'Invertebrates', 'Fungi']
INV_RED  = '#DC2626'
INV_GRAY = '#9CA3AF'

# =============================================================================
# === F3A: TE distribution data ===
# =============================================================================
print("=== F3A: Loading TE distribution data ===")

clade_map = {'Human': 'Human', 'Mammals': 'Mammals', 'Vertebrates': 'Vertebrates',
             'Invertebrates': 'Invertebrates', 'Fungi': 'Fungi'}
checked_a = pd.read_csv(os.path.join(DATA_DIR, 'F3A_TE_distribution_plot_values.csv.gz'))
plot_data_A = []
for (name, clade), group in checked_a.groupby(['species', 'clade'], sort=False):
    source = group.loc[group['plot_role'].eq('cell_level_kde_input')].copy()
    grand = group.loc[group['plot_role'].ne('cell_level_kde_input')].copy()
    if grand.empty:
        grand = group.copy()
    source = source.rename(columns={'log10_te': 'log_TE', 'te': 'TE_mean', 'cell_or_sample': 'cell'})
    grand = grand.rename(columns={'log10_te': 'log_TE', 'te': 'TE_mean'})
    plot_data_A.append({'name': name, 'clade': clade_map.get(clade, clade),
                        'source_df': source if not source.empty else None,
                        'grand_df': grand})
plot_data_A.sort(key=lambda x: (CLADE_ORDER.index(x['clade']) if x['clade'] in CLADE_ORDER else 99, x['name']))
print(f"  F3A: {len(plot_data_A)} species")

# =============================================================================
# === F3B: 3-model prediction distributions ===
# =============================================================================
print("=== F3B: Loading prediction distribution data ===")

F3B_MODELS = [
    {'label': 'InvaRNA',  'file': f'{DATA_DIR}/F3B_three_model_predictions_merged.csv', 'col': 'pred_invarna0412'},
    {'label': 'RiboNN',   'file': f'{DATA_DIR}/F3B_three_model_predictions_merged.csv', 'col': 'pred_ribonn'},
    {'label': 'UTR-LM',   'file': f'{DATA_DIR}/F3B_three_model_predictions_merged.csv', 'col': 'pred_utrlm'},
]

for m in F3B_MODELS:
    raw = pd.read_csv(m['file'])
    col = m['col']
    if 'true_te' in raw.columns:
        raw = raw[raw['true_te'] > 0].copy()
        raw['true_te_plot'] = np.log2(raw['true_te'])
    else:
        raw['true_te_plot'] = np.nan
    raw['pred_centered'] = raw.groupby('display')[col].transform(lambda x: x - x.mean())
    raw['true_centered'] = raw.groupby('display')['true_te_plot'].transform(
        lambda x: x - x.mean() if x.notna().any() else x)
    m['df'] = raw
    print(f"  {m['label']}: {len(raw)} rows")

ref_df_b = F3B_MODELS[0]['df']
checked_b = pd.read_csv(os.path.join(DATA_DIR, 'F3B_species_spearman_no_fly.csv'))
display_order_b = checked_b['display'].drop_duplicates().tolist()
for meta in F3B_MODELS:
    meta['df'] = meta['df'].loc[meta['df']['display'].isin(display_order_b)].copy()

# =============================================================================
# === F3C: UMAP embedding (use cache) ===
# =============================================================================
print("=== F3C: Loading UMAP ===")
checked_c = pd.read_csv(os.path.join(DATA_DIR, 'F3C_UMAP_plot_points.csv'))
df_sub_c = checked_c.copy()
X_umap_c = df_sub_c[['umap_1', 'umap_2']].to_numpy()
df_sub_c['mut_id'] = pd.to_numeric(df_sub_c['mut_id'], errors='coerce').fillna(-1).astype(int)
mask_wt = df_sub_c['mut_id'].eq(0)
mask_k80 = df_sub_c['mut_id'].between(1, 200)
mask_fm = df_sub_c['mut_id'].between(201, 215)
CMAP_K80 = plt.cm.viridis
NORM_K80 = mcolors.Normalize(vmin=0, vmax=200)
CMAP_FM = plt.cm.plasma
NORM_FM = mcolors.Normalize(vmin=201, vmax=215)
xlim_c = (X_umap_c[:, 0].min() - 0.5, X_umap_c[:, 0].max() + 0.5)
ylim_c = (X_umap_c[:, 1].min() - 0.5, X_umap_c[:, 1].max() + 0.5)
print(f"  Checked UMAP points: {len(df_sub_c)}")

# =============================================================================
# === F3C-2: Conservation track (2con, optional) ===
# =============================================================================
print("=== F3C-2: Conservation track ===")

checked_con = pd.read_csv(os.path.join(DATA_DIR, 'F3C_conservation_track_and_sampled_mutations.csv'))
con_data = {
    'seq': ''.join(checked_con['wt_base'].astype(str)),
    'mutated_seq': ''.join(checked_con['mutated_base'].astype(str)),
    'cons': checked_con['phastcons_probability'].to_numpy(),
    'symbol': str(checked_con['symbol'].iloc[0]),
}
print(f"  Checked conservation track: {len(checked_con)} bp")

# =============================================================================
# === F3F: MPRA 3k Spearman bar (InvaRNA0412 vs RiboNN vs UTRLM, 10 datasets) ===
# =============================================================================
print("=== F3F: Loading MPRA 3k data ===")

checked_g = pd.read_csv(os.path.join(DATA_DIR, 'F3G_external_MPRA_spearman_by_sample.csv'))
results_f = checked_g.rename(columns={
    'sample': 'Batch', 'model': 'Model', 'spearman_rho': 'Spearman_R'
})[['Batch', 'Model', 'Spearman_R']]
print(f"  Checked MPRA rows: {len(results_f)}")

# =============================================================================
# === F3G: Benchmark bar (hardcoded) ===
# =============================================================================
checked_f = pd.read_csv(os.path.join(DATA_DIR, 'F3F_heldout_human_TE_prediction_R2.csv'))
G_labels = checked_f['model'].tolist()
G_scores = checked_f['heldout_human_te_prediction_r2'].tolist()
G_colors = [INV_RED if name == 'InvaRNA' else ('#9CA3AF' if name == 'RiboNN' else '#DDDDDF')
            for name in G_labels]

# =============================================================================
# === FIGURE LAYOUT ===
# =============================================================================
print("=== Drawing combined figure ===")

fig = plt.figure(figsize=(30, 26), dpi=150)

outer_gs = gridspec.GridSpec(
    3, 1, figure=fig,
    height_ratios=[1.4, 1.5, 0.9],
    hspace=0.45)

# ─────────────────────────────────────────────────────────────────────────────
# Row 0: A (TE manifold) | B (3-model pred, 3-stacked)
# ─────────────────────────────────────────────────────────────────────────────
row0_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer_gs[0],
    width_ratios=[1.7, 1], wspace=0.42)

# --- Panel A ---
ax_a = fig.add_subplot(row0_gs[0])
for item in plot_data_A:
    color    = CLADE_COLORS.get(item['clade'], 'gray')
    is_human = (item['clade'] == 'Human')
    ls, lw, alpha = ('-', 5, 1.0) if is_human else ('--', 2.5, 0.7)
    label    = item['name'] if is_human else f"[{item['clade']}] {item['name']}"
    if item['source_df'] is not None and 'log_TE' in item['source_df'].columns:
        sns.kdeplot(data=item['source_df'], x='log_TE', hue='cell',
                    palette=[color] * item['source_df']['cell'].nunique(),
                    alpha=0.05, linewidth=0.4, common_norm=False, legend=False, ax=ax_a)
    sns.kdeplot(data=item['grand_df'], x='log_TE', color=color,
                linewidth=lw, linestyle=ls, alpha=alpha, label=label, ax=ax_a)

ax_a.set_xlabel('log\u2081\u2080(TE)', fontsize=11)
ax_a.set_ylabel('Density', fontsize=11)
ax_a.set_title('', pad=2)
ax_a.set_xlim(-2, 2)
ax_a.grid(True, linestyle=':', alpha=0.4)
ax_a.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7.5,
            title='Species', title_fontsize=8, frameon=False)
spine_off(ax_a)
panel_label(ax_a, 'A')

# --- Panel B (3 stacked) ---
b_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=row0_gs[1], hspace=0.50)
axes_b = []
for i, meta in enumerate(F3B_MODELS):
    ax_bi = fig.add_subplot(b_gs[i])
    df_bi, col = meta['df'], meta['col']
    human_sub = df_bi[(df_bi['clade'] == 'Human') & df_bi['true_te_plot'].notna()]
    if len(human_sub) >= 30:
        tv = human_sub['true_te_plot'].dropna()
        if tv.std() > 1e-4:
            sns.kdeplot(data=tv, ax=ax_bi, color='gray', fill=True,
                        alpha=0.15, linewidth=0, zorder=5)
            sns.kdeplot(data=tv, ax=ax_bi, color='gray', linewidth=1.8,
                        alpha=0.85, linestyle='-', zorder=6, label='H.sapiens True TE')
    for display in display_order_b:
        sub = df_bi[df_bi['display'] == display]
        if len(sub) < 30: continue
        clade_  = sub['clade'].iloc[0]
        color_  = CLADE_COLORS.get(clade_, 'gray')
        is_hum  = (clade_ == 'Human')
        pv = sub[col].dropna()
        if len(pv) < 30 or pv.std() < 1e-4: continue
        lw, alp = (2.5, 1.0) if is_hum else (1.5, 0.7)
        lbl = display if is_hum else f'[{clade_}] {display}'
        sns.kdeplot(data=pv, ax=ax_bi, color=color_, linewidth=lw,
                    linestyle='--', alpha=alp, label=lbl,
                    zorder=100 if is_hum else 50)
    _rho_sym = '\u03c1'
    lines = [f"{'Sp':<8s}  {_rho_sym:>4s}"]
    for d in display_order_b:
        sub = df_bi[df_bi['display'] == d].dropna(subset=[col, 'true_te_plot'])
        if len(sub) < 10: continue
        rho = spearmanr(sub['true_te_plot'].values, sub[col].values)[0]
        lines.append(f"{d:<8s}  {rho:>4.2f}")
    ax_bi.text(0.98, 0.97, '\n'.join(lines), transform=ax_bi.transAxes,
               ha='right', va='top', fontsize=6.5, family='monospace',
               bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7))
    ax_bi.set_title(meta['label'], fontsize=10, fontweight='bold')
    ax_bi.set_xlabel('log(TE)' if i == 2 else '', fontsize=9)
    ax_bi.set_ylabel('Density', fontsize=9)
    ax_bi.set_xlim(-3, 3)
    ax_bi.grid(True, linestyle=':', alpha=0.4)
    ax_bi.tick_params(labelsize=7)
    spine_off(ax_bi)
    axes_b.append(ax_bi)
    if i == 0:
        panel_label(ax_bi, 'B', x=-0.22)

ymax_b = max(ax.get_ylim()[1] for ax in axes_b)
for ax in axes_b:
    ax.set_ylim(0, ymax_b)

# ─────────────────────────────────────────────────────────────────────────────
# Row 1: C (UMAP 4-panel + conservation strip) | D (Conceptual diagram)
# ─────────────────────────────────────────────────────────────────────────────
row1_gs = gridspec.GridSpecFromSubplotSpec(
    1, 2, subplot_spec=outer_gs[1],
    width_ratios=[1, 1.2], wspace=0.32)

# --- Panel C: UMAP 2x2 + conservation strip below ---
c_outer = gridspec.GridSpecFromSubplotSpec(
    2, 1, subplot_spec=row1_gs[0],
    height_ratios=[3, 1.5], hspace=0.45)

# C top: UMAP 2x2
c_gs = gridspec.GridSpecFromSubplotSpec(
    2, 3, subplot_spec=c_outer[0], hspace=0.08,
    wspace=0.05, width_ratios=[1, 1, 0.06])

def _scat(ax, x, y, c, cmap, norm):
    return ax.scatter(x, y, c=c, cmap=cmap, norm=norm,
                      s=6, alpha=0.75, edgecolors='none', rasterized=True)

def _style_umap(ax, title):
    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    if X_umap_c is not None:
        ax.set_xlim(xlim_c); ax.set_ylim(ylim_c)
    ax.axis('off')

if X_umap_c is not None:
    ax_c1 = fig.add_subplot(c_gs[0, 0])
    ax_c1.scatter(X_umap_c[mask_wt, 0], X_umap_c[mask_wt, 1],
                  c='steelblue', s=6, alpha=0.75, edgecolors='none', rasterized=True)
    _style_umap(ax_c1, 'WT (mut_id = 0)')
    panel_label(ax_c1, 'C', x=-0.18, y=1.06)
    ax_c1.text(1.08, 1.25, '',
               transform=ax_c1.transAxes, ha='center', va='bottom',
               fontsize=11, fontweight='bold', color='#1a1a2e')

    ax_c2 = fig.add_subplot(c_gs[0, 1])
    _scat(ax_c2, X_umap_c[mask_k80, 0], X_umap_c[mask_k80, 1],
          df_sub_c.loc[mask_k80, 'mut_id'].values, CMAP_K80, NORM_K80)
    _style_umap(ax_c2, 'K80-mutant (1\u2013200)')
    cax_k80 = fig.add_subplot(c_gs[0, 2])
    cbar2 = plt.colorbar(plt.cm.ScalarMappable(norm=NORM_K80, cmap=CMAP_K80), cax=cax_k80)
    cbar2.set_label('Mutation Steps (K80)', fontsize=7)
    cbar2.outline.set_visible(False)

    ax_c3 = fig.add_subplot(c_gs[1, 0])
    _scat(ax_c3, X_umap_c[mask_fm, 0], X_umap_c[mask_fm, 1],
          df_sub_c.loc[mask_fm, 'mut_id'].values, CMAP_FM, NORM_FM)
    _style_umap(ax_c3, 'FM-mutant (201\u2013215)')

    ax_c4 = fig.add_subplot(c_gs[1, 1])
    ax_c4.scatter(X_umap_c[mask_wt, 0], X_umap_c[mask_wt, 1],
                  c='steelblue', s=6, alpha=0.75, edgecolors='none', zorder=1,
                  rasterized=True)
    _scat(ax_c4, X_umap_c[mask_k80, 0], X_umap_c[mask_k80, 1],
          df_sub_c.loc[mask_k80, 'mut_id'].values, CMAP_K80, NORM_K80)
    _scat(ax_c4, X_umap_c[mask_fm, 0], X_umap_c[mask_fm, 1],
          df_sub_c.loc[mask_fm, 'mut_id'].values, CMAP_FM, NORM_FM)
    _style_umap(ax_c4, 'All')
    cax_fm = fig.add_subplot(c_gs[1, 2])
    cbar3 = plt.colorbar(plt.cm.ScalarMappable(norm=NORM_FM, cmap=CMAP_FM), cax=cax_fm)
    cbar3.set_label('Mutation Steps (FM)', fontsize=7)
    cbar3.outline.set_visible(False)
else:
    ax_cdummy = fig.add_subplot(c_outer[0])
    ax_cdummy.text(0.5, 0.5, 'UMAP cache not found', ha='center', va='center',
                   fontsize=10, color='gray')
    ax_cdummy.axis('off')
    panel_label(ax_cdummy, 'C')

# C bottom: Conservation strip
ax_con = fig.add_subplot(c_outer[1])
if con_data is not None:
    seq  = con_data['seq']
    mseq = con_data['mutated_seq']
    cons = con_data['cons']
    pos  = np.arange(len(seq))
    muts = [i for i, (a, b) in enumerate(zip(seq, mseq)) if a != b]
    msc  = [cons[i] for i in muts]
    ax_con.fill_between(pos, cons, 0, color='#E5E7EB', alpha=0.8, edgecolor='none')
    ax_con.plot(pos, cons, color='#6B7280', linewidth=1.5, label='Conservation Score (PhastCons)')
    ax_con.scatter(muts, msc, color=INV_RED, edgecolor='white',
                   linewidth=0.8, s=50, zorder=5, label='Dynamically Sampled Mutation')
    ax_con.set_xlabel("Sequence Position in 5' UTR (bp)", fontsize=10, fontweight='bold')
    ax_con.set_ylabel('Conservation\nProbability', fontsize=10, fontweight='bold')
    ax_con.set_title('')
    ax_con.set_ylim(-0.02, 1.05)
    ax_con.set_xlim(0, len(seq))
    ax_con.legend(frameon=False, fontsize=8, loc='upper center',
                  bbox_to_anchor=(0.5, 1.25), ncol=2)
    spine_off(ax_con)
else:
    ax_con.text(0.5, 0.5, 'Conservation track unavailable',
                ha='center', va='center', fontsize=9, color='gray')
    ax_con.axis('off')

# --- Panel D: Conceptual diagram ---
ax_d = fig.add_subplot(row1_gs[1])
ax_d.set_facecolor('white')
ax_d.axis('off')
ax_d.set_xlim(-2.0, 6.5)
ax_d.set_ylim(0.5, 4.3)

def _tl(x): return 2.3 + 0.38*x - 0.065*x**2 - 0.001*x**3
def _sl(x): return 1.8 + 0.18*x - 0.025*x**2

x_wt = 0.5; y_wt = _tl(x_wt)
x_m  = np.array([-0.8, -0.1, 1.4, 2.2])
y_m  = _tl(x_m)
y_al = y_wt + (_sl(x_m) - _sl(x_wt))
kap  = 0.5
xc   = np.linspace(-1.5, 5.0, 500)
y_true_c = _tl(xc)
y_aln_c  = y_wt + (_sl(xc) - _sl(x_wt))
y_res_c  = kap * y_true_c + (1 - kap) * y_aln_c

ax_d.plot(xc, y_true_c, color='#e5e7eb', lw=6, alpha=0.7, zorder=1)
ax_d.text(0.5, 1.7, 'Latent TE landscape', ha='center', fontsize=8.5,
          color='#9ca3af', style='italic')
ax_d.plot(xc, y_aln_c, color='#fca5a5', lw=1.8, ls='--', alpha=0.8, zorder=2)
ax_d.plot(xc, y_res_c, color='#ef4444', lw=3.5, zorder=3)
ax_d.scatter(x_wt, y_wt, s=240, color='black', edgecolor='white', lw=2.2, zorder=15)
ax_d.scatter(x_m, y_m, s=100, color='#6b7280', zorder=5)
ax_d.scatter(x_m, y_al, s=110, marker='^', color='#2563eb', zorder=6)
for xi, yr, ya in zip(x_m, y_m, y_al):
    ax_d.plot([xi, xi], [yr, ya], color='#bfdbfe', lw=1.3, zorder=4)

ax_d.annotate('', xy=(5.6, 0.7),  xytext=(-1.5, 0.7),
              arrowprops=dict(arrowstyle='->', lw=1.6, color='black'))
ax_d.annotate('', xy=(-1.5, 3.95), xytext=(-1.5, 0.7),
              arrowprops=dict(arrowstyle='->', lw=1.6, color='black'))
ax_d.text(2.0, 0.42, 'Local Sequence Space ($x$)',
          ha='center', fontsize=13, fontweight='bold')
ax_d.text(-1.85, 2.25, 'Translation Efficiency (TE)',
          ha='center', va='center', rotation=90, fontsize=13, fontweight='bold')
ax_d.text(0.1, 4.15, 'Soft label: Taylor-Distance Correction (TDC)',
          color='#111827', fontweight='bold', fontsize=11.5)
ax_d.text(0.1, 3.08, 'Distance-aware evolutionary soft labeling',
          color='#1e40af', fontweight='bold', fontsize=11)
ax_d.text(0.1, 3.50,
          r'$y_{mut} = y_{wt}^{real} + [\mathcal{T}(x_m) - \mathcal{T}(x_{wt})] \cdot \left(\frac{n_{mut}}{n_{max}}\right)^{\!\alpha}$',
          fontsize=16, color='#1e40af')
ax_d.text(0.1, 3.22,
          r'$\alpha = 0.3$,  $n_{mut}$: Hamming distance,  $n_{max}$: max mutations per gene',
          fontsize=10, color='#1e40af')
ax_d.text(4.85, 3.42, 'Pure-sequence\nInvaRNA Student',
          ha='center', va='center', color='white', fontweight='bold', fontsize=10,
          bbox=dict(boxstyle='round,pad=0.45', facecolor='#ef4444', edgecolor='none'),
          zorder=10)
ax_d.text(5.3, 2.75, r'$\mathcal{T}$: teacher prediction',
          fontsize=10, color='#6b7280', ha='center', style='italic')
ax_d.text(5.3, 2.40, r'$\alpha$: distance decay exponent',
          fontsize=10, color='#6b7280', ha='center', style='italic')
legend_d = [
    mlines.Line2D([0],[0], marker='o', color='none', markerfacecolor='black',
                  markersize=9, label=r'Real WT anchor ($y_{wt}^{real}$)'),
    mlines.Line2D([0],[0], marker='o', color='none', markerfacecolor='#6b7280',
                  markersize=9, label=r'Local mutants ($x_m$)'),
    mlines.Line2D([0],[0], marker='^', color='none', markerfacecolor='#2563eb',
                  markersize=9, label=r'Distance-scaled soft labels ($y_{mut}$)'),
    mlines.Line2D([0],[0], color='#fca5a5', lw=1.8, ls='--',
                  label='Taylor correction only'),
    mlines.Line2D([0],[0], color='#ef4444', lw=3.5,
                  label='Final correction (InvaRNA)'),
]
ax_d.legend(handles=legend_d, loc='lower left', frameon=False,
            fontsize=9, bbox_to_anchor=(0.28, 0.01))
panel_label(ax_d, 'D', x=-0.05)

# ─────────────────────────────────────────────────────────────────────────────
# Row 2: E (ablation) | F (MPRA) | G (benchmark)
# ─────────────────────────────────────────────────────────────────────────────
row2_gs = gridspec.GridSpecFromSubplotSpec(
    1, 3, subplot_spec=outer_gs[2],
    width_ratios=[0.9, 1.2, 1.2], wspace=0.40)

# --- Panel E: Ablation UpSet ---
e_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=row2_gs[0],
                                         height_ratios=[3, 1.2], hspace=0.08)
ax_ebar = fig.add_subplot(e_gs[0])
ax_edot = fig.add_subplot(e_gs[1])

E_models = ['human', 'human_mouse', 'rand_mut', 'evol_mut', 'taylor', 'final']
E_labels_dot = ['Human data', 'Mouse data', 'Random aug.', 'Evolution-guided aug.', 'Taylor correction', 'TDC soft label']
checked_e = pd.read_csv(os.path.join(DATA_DIR, 'F3E_ablation_plot_data.csv')).set_index('configuration').loc[E_models]
E_val_r2 = checked_e['test_r2_plotted'].tolist()
E_c = INV_GRAY

x_e = np.arange(len(E_models))
ax_ebar.bar(x_e, E_val_r2, width=0.55, color=E_c, edgecolor='white', linewidth=0.5, zorder=3)
for i, (xi, v) in enumerate(zip(x_e, E_val_r2)):
    ax_ebar.text(xi, v + 0.002, f'{v:.4f}', ha='center', va='bottom',
                 fontsize=8, fontweight='bold', color='#1a1a1a')
for i in range(1, len(E_val_r2)):
    delta = E_val_r2[i] - E_val_r2[i-1]
    ax_ebar.text(x_e[i], E_val_r2[i] + 0.010, f'+{delta:.4f}',
                 ha='center', va='bottom', fontsize=6.5, color='#666666', fontstyle='italic')
ax_ebar.set_ylim(0.62, 0.78)
ax_ebar.set_ylabel('Test R\u00b2', fontsize=9, fontweight='bold')
ax_ebar.set_xticks([])
ax_ebar.spines[['top', 'right', 'bottom']].set_visible(False)
ax_ebar.tick_params(labelsize=8)
ax_ebar.grid(axis='y', alpha=0.15, zorder=0)
panel_label(ax_ebar, 'E', x=-0.18)

E_active = checked_e[[
    'human_data', 'mouse_data', 'random_aug', 'evolution-guided_aug',
    'taylor_correction', 'tdc_soft_label',
]].astype(int).values.tolist()
E_c_on = '#2F4F4F'; E_c_off = '#D0D0D0'
for col in range(len(E_models)):
    for row in range(len(E_labels_dot)):
        color = E_c_on if E_active[col][row] else E_c_off
        ax_edot.scatter(col, len(E_labels_dot)-1-row, s=80, color=color,
                        zorder=3, edgecolors='white', linewidths=0.5)
    active_rows = [len(E_labels_dot)-1-r for r in range(len(E_labels_dot)) if E_active[col][r]]
    if len(active_rows) > 1:
        ax_edot.plot([col, col], [min(active_rows), max(active_rows)],
                     color=E_c_on, lw=2.0, zorder=2)
ax_edot.set_yticks(range(len(E_labels_dot)))
ax_edot.set_yticklabels(E_labels_dot[::-1], fontsize=6.5, fontweight='normal')
ax_edot.set_xticks([])
ax_edot.set_xlim(-0.5, len(E_models)-0.5)
ax_edot.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
ax_edot.tick_params(left=False)

# --- Panel F: MPRA mean±SD summary bar ---
ax_f = fig.add_subplot(row2_gs[2])
model_order_f = ['UTRLM', 'RiboNN', 'InvaRNA']
palette_f = {'InvaRNA': INV_RED, 'RiboNN': INV_GRAY, 'UTRLM': '#D1D5DB'}
f_means, f_sds = [], []
for m in model_order_f:
    vals = results_f[results_f['Model'] == m]['Spearman_R'].dropna()
    f_means.append(vals.mean())
    f_sds.append(vals.std())
x_f = np.arange(len(model_order_f))
bars_f = ax_f.bar(x_f, f_means, yerr=f_sds, capsize=6, width=0.55,
                  color=[palette_f[m] for m in model_order_f],
                  edgecolor='none', error_kw=dict(lw=1.2), zorder=3)
for i, (m, v) in enumerate(zip(model_order_f, f_means)):
    ax_f.text(x_f[i], v + f_sds[i] + 0.015, f'{v:.3f}',
              ha='center', va='bottom', fontsize=10, fontweight='bold')
ax_f.set_xticks(x_f)
ax_f.set_xticklabels(model_order_f, fontsize=10, fontweight='bold')
ax_f.set_ylabel('Spearman \u03c1 on external MPRA\n(mean \u00b1 SD, n=10)', fontsize=10)
ax_f.set_ylim(0, 0.55)
ax_f.axhline(0, color='black', linewidth=1.0)
ax_f.grid(False)
spine_off(ax_f)
ax_f.spines['bottom'].set_visible(False)
ax_f.tick_params(axis='x', length=0)
panel_label(ax_f, 'G', x=-0.18)

# --- Panel G: Benchmark bar ---
ax_g = fig.add_subplot(row2_gs[1])
x_g = np.arange(len(G_labels))
width_g = 0.62
ax_g.bar(x_g, G_scores, width_g, color=G_colors, edgecolor='none', zorder=3)
for i in range(len(G_labels)):
    if i == len(G_labels) - 1:
        ax_g.text(x_g[i], G_scores[i] + 0.005, f'{G_scores[i]:.3f}',
                  ha='center', va='bottom', fontsize=12, fontweight='bold', color='#111111')
    elif i == 5:
        ax_g.text(x_g[i], G_scores[i] + 0.004, f'{G_scores[i]:.3f}',
                  ha='center', va='bottom', fontsize=9, fontweight='bold')
ax_g.set_title('', pad=2)
ax_g.set_ylim(0.36, 0.82)
ax_g.set_ylabel("Held-out Human Test Performance (R\u00b2)", fontsize=11)
ax_g.set_xticks(x_g)
tick_labels_g = ax_g.set_xticklabels(G_labels, rotation=28, ha='right')
for i, tick in enumerate(tick_labels_g):
    if i < 5:
        tick.set_color('#9CA3AF'); tick.set_fontsize(9); tick.set_fontweight('normal')
    elif i == 5:
        tick.set_color('#4B5563'); tick.set_fontsize(9.5); tick.set_fontweight('semibold')
    else:
        tick.set_color('black'); tick.set_fontsize(10); tick.set_fontweight('bold')
ax_g.spines['top'].set_visible(False)
ax_g.spines['right'].set_visible(False)
ax_g.spines['bottom'].set_visible(False)
ax_g.tick_params(axis='x', length=0)
panel_label(ax_g, 'F', x=-0.10)

# =============================================================================
# Save
# =============================================================================
date_str = 'checked'
out_pdf = f'{OUT_DIR}/InvaRNA_Fig3_Combined_{date_str}.pdf'
out_png = f'{OUT_DIR}/InvaRNA_Fig3_Combined_{date_str}.png'
plt.savefig(out_pdf, bbox_inches='tight')
print(f"Saved: {out_pdf}")
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"Saved: {out_png}")

def _save_standalone(fig_single, name, png_dpi=200):
    pdf_path = f'{OUT_DIR}/panel_{name}_{date_str}.pdf'
    png_path = f'{OUT_DIR}/panel_{name}_{date_str}.png'
    fig_single.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved: {pdf_path}")
    fig_single.savefig(png_path, dpi=png_dpi, bbox_inches='tight')
    print(f"Saved: {png_path}")
    plt.close(fig_single)


def _save_reference_panel_c():
    png_path = f'{OUT_DIR}/panel_C_{date_str}.png'
    pdf_path = f'{OUT_DIR}/panel_C_{date_str}.pdf'
    shutil.copyfile(PANEL_C_REFERENCE_PNG, png_path)
    print(f"Saved: {png_path}")

    ref_img = plt.imread(PANEL_C_REFERENCE_PNG)
    height, width = ref_img.shape[:2]
    fig_ref = plt.figure(figsize=(width / 200.0, height / 200.0), dpi=200)
    ax_ref = fig_ref.add_axes([0, 0, 1, 1])
    ax_ref.imshow(ref_img)
    ax_ref.axis('off')
    fig_ref.savefig(pdf_path, dpi=200)
    print(f"Saved: {pdf_path}")
    plt.close(fig_ref)


def _make_panel_a():
    fig_single, ax = plt.subplots(figsize=(12.5, 5.0), dpi=150)
    for item in plot_data_A:
        color = CLADE_COLORS.get(item['clade'], 'gray')
        is_human = (item['clade'] == 'Human')
        ls, lw, alpha = ('-', 5, 1.0) if is_human else ('--', 2.5, 0.7)
        label = item['name'] if is_human else f"[{item['clade']}] {item['name']}"
        if item['source_df'] is not None and 'log_TE' in item['source_df'].columns:
            sns.kdeplot(data=item['source_df'], x='log_TE', hue='cell',
                        palette=[color] * item['source_df']['cell'].nunique(),
                        alpha=0.05, linewidth=0.4, common_norm=False, legend=False, ax=ax)
        sns.kdeplot(data=item['grand_df'], x='log_TE', color=color,
                    linewidth=lw, linestyle=ls, alpha=alpha, label=label, ax=ax)
    ax.set_xlabel('log\u2081\u2080(TE)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_xlim(-2, 2)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7.5,
              title='Species', title_fontsize=8, frameon=False)
    spine_off(ax)
    panel_label(ax, 'A')
    fig_single.tight_layout()
    return fig_single


def _make_panel_b():
    fig_single = plt.figure(figsize=(7.2, 8.6), dpi=150)
    b_single = gridspec.GridSpec(3, 1, figure=fig_single, hspace=0.50)
    local_axes = []
    for i, meta in enumerate(F3B_MODELS):
        ax_bi = fig_single.add_subplot(b_single[i])
        df_bi, col = meta['df'], meta['col']
        human_sub = df_bi[(df_bi['clade'] == 'Human') & df_bi['true_te_plot'].notna()]
        if len(human_sub) >= 30:
            tv = human_sub['true_te_plot'].dropna()
            if tv.std() > 1e-4:
                sns.kdeplot(data=tv, ax=ax_bi, color='gray', fill=True,
                            alpha=0.15, linewidth=0, zorder=5)
                sns.kdeplot(data=tv, ax=ax_bi, color='gray', linewidth=1.8,
                            alpha=0.85, linestyle='-', zorder=6, label='H.sapiens True TE')
        for display in display_order_b:
            sub = df_bi[df_bi['display'] == display]
            if len(sub) < 30:
                continue
            clade_ = sub['clade'].iloc[0]
            color_ = CLADE_COLORS.get(clade_, 'gray')
            is_hum = (clade_ == 'Human')
            pv = sub[col].dropna()
            if len(pv) < 30 or pv.std() < 1e-4:
                continue
            lw, alp = (2.5, 1.0) if is_hum else (1.5, 0.7)
            lbl = display if is_hum else f'[{clade_}] {display}'
            sns.kdeplot(data=pv, ax=ax_bi, color=color_, linewidth=lw,
                        linestyle='--', alpha=alp, label=lbl,
                        zorder=100 if is_hum else 50)
        _rho_sym = '\u03c1'
        lines = [f"{'Sp':<8s}  {_rho_sym:>4s}"]
        for d in display_order_b:
            sub = df_bi[df_bi['display'] == d].dropna(subset=[col, 'true_te_plot'])
            if len(sub) < 10:
                continue
            rho = spearmanr(sub['true_te_plot'].values, sub[col].values)[0]
            lines.append(f'{d:<8s}  {rho:>4.2f}')
        ax_bi.text(0.98, 0.97, '\n'.join(lines), transform=ax_bi.transAxes,
                   ha='right', va='top', fontsize=6.5, family='monospace',
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7))
        ax_bi.set_title(meta['label'], fontsize=10, fontweight='bold')
        ax_bi.set_xlabel('log(TE)' if i == 2 else '', fontsize=9)
        ax_bi.set_ylabel('Density', fontsize=9)
        ax_bi.set_xlim(-3, 3)
        ax_bi.grid(True, linestyle=':', alpha=0.4)
        ax_bi.tick_params(labelsize=7)
        spine_off(ax_bi)
        local_axes.append(ax_bi)
        if i == 0:
            panel_label(ax_bi, 'B', x=-0.22)
    ymax_b_local = max(ax.get_ylim()[1] for ax in local_axes)
    for ax in local_axes:
        ax.set_ylim(0, ymax_b_local)
    fig_single.tight_layout()
    return fig_single


def _make_panel_c():
    fig_single = plt.figure(figsize=(12.19, 15.485), dpi=150)
    fig_single.suptitle('Multi-scale manifold consistency', fontsize=11.5,
                        fontweight='bold', y=0.972)
    c_outer_single = gridspec.GridSpec(
        2, 1, figure=fig_single, height_ratios=[3.0, 1.45], hspace=0.28)

    if X_umap_c is not None:
        c_grid_single = gridspec.GridSpecFromSubplotSpec(
            2, 3, subplot_spec=c_outer_single[0], hspace=0.10,
            wspace=0.06, width_ratios=[1, 1, 0.055])

        ax_c1_s = fig_single.add_subplot(c_grid_single[0, 0])
        ax_c1_s.scatter(
            X_umap_c[mask_wt, 0], X_umap_c[mask_wt, 1],
            c='steelblue', s=6, alpha=0.75, edgecolors='none', rasterized=True)
        _style_umap(ax_c1_s, 'WT (mut_id = 0)')
        ax_c1_s.set_aspect('equal', adjustable='box')
        panel_label(ax_c1_s, 'C', x=-0.22, y=1.08)

        ax_c2_s = fig_single.add_subplot(c_grid_single[0, 1])
        _scat(
            ax_c2_s, X_umap_c[mask_k80, 0], X_umap_c[mask_k80, 1],
            df_sub_c.loc[mask_k80, 'mut_id'].values, CMAP_K80, NORM_K80)
        _style_umap(ax_c2_s, 'K80-mutant (1\u2013200)')
        ax_c2_s.set_aspect('equal', adjustable='box')
        cax_k80_s = fig_single.add_subplot(c_grid_single[0, 2])
        cbar2_s = plt.colorbar(
            plt.cm.ScalarMappable(norm=NORM_K80, cmap=CMAP_K80), cax=cax_k80_s)
        cbar2_s.set_label('Mutation Steps (K80)', fontsize=8)
        cbar2_s.outline.set_visible(False)

        ax_c3_s = fig_single.add_subplot(c_grid_single[1, 0])
        _scat(
            ax_c3_s, X_umap_c[mask_fm, 0], X_umap_c[mask_fm, 1],
            df_sub_c.loc[mask_fm, 'mut_id'].values, CMAP_FM, NORM_FM)
        _style_umap(ax_c3_s, 'FM-mutant (201\u2013215)')
        ax_c3_s.set_aspect('equal', adjustable='box')

        ax_c4_s = fig_single.add_subplot(c_grid_single[1, 1])
        ax_c4_s.scatter(
            X_umap_c[mask_wt, 0], X_umap_c[mask_wt, 1],
            c='steelblue', s=6, alpha=0.75, edgecolors='none', zorder=1,
            rasterized=True)
        _scat(
            ax_c4_s, X_umap_c[mask_k80, 0], X_umap_c[mask_k80, 1],
            df_sub_c.loc[mask_k80, 'mut_id'].values, CMAP_K80, NORM_K80)
        _scat(
            ax_c4_s, X_umap_c[mask_fm, 0], X_umap_c[mask_fm, 1],
            df_sub_c.loc[mask_fm, 'mut_id'].values, CMAP_FM, NORM_FM)
        _style_umap(ax_c4_s, 'All')
        ax_c4_s.set_aspect('equal', adjustable='box')
        cax_fm_s = fig_single.add_subplot(c_grid_single[1, 2])
        cbar3_s = plt.colorbar(
            plt.cm.ScalarMappable(norm=NORM_FM, cmap=CMAP_FM), cax=cax_fm_s)
        cbar3_s.set_label('Mutation Steps (FM)', fontsize=8)
        cbar3_s.outline.set_visible(False)
    else:
        ax_msg = fig_single.add_subplot(c_outer_single[0])
        ax_msg.text(0.5, 0.5, 'UMAP cache not found', ha='center', va='center',
                    fontsize=10, color='gray')
        ax_msg.axis('off')
        panel_label(ax_msg, 'C')

    ax_con_s = fig_single.add_subplot(c_outer_single[1])
    if con_data is not None:
        seq = con_data['seq']
        mseq = con_data['mutated_seq']
        cons = con_data['cons']
        pos = np.arange(len(seq))
        muts = [i for i, (a, b) in enumerate(zip(seq, mseq)) if a != b]
        msc = [cons[i] for i in muts]
        ax_con_s.fill_between(pos, cons, 0, color='#E5E7EB', alpha=0.8, edgecolor='none')
        ax_con_s.plot(pos, cons, color='#6B7280', linewidth=1.5,
                      label='Conservation Score (PhastCons)')
        ax_con_s.scatter(muts, msc, color=INV_RED, edgecolor='white',
                         linewidth=0.8, s=50, zorder=5,
                         label='Dynamically Sampled Mutation')
        ax_con_s.set_xlabel("Sequence Position in 5' UTR (bp)", fontsize=10, fontweight='bold')
        ax_con_s.set_ylabel('Conservation\nProbability', fontsize=10, fontweight='bold')
        ax_con_s.set_ylim(-0.02, 1.05)
        ax_con_s.set_xlim(0, len(seq))
        ax_con_s.legend(frameon=False, fontsize=8, loc='upper center',
                        bbox_to_anchor=(0.5, 1.25), ncol=2)
        spine_off(ax_con_s)
    else:
        ax_con_s.text(0.5, 0.5, 'Conservation track unavailable',
                      ha='center', va='center', fontsize=9, color='gray')
        ax_con_s.axis('off')

    fig_single.tight_layout(rect=[0.02, 0.02, 0.985, 0.96])
    return fig_single


def _make_panel_d():
    fig_single, ax = plt.subplots(figsize=(10.2, 5.6), dpi=150)
    ax.set_facecolor('white')
    ax.axis('off')
    ax.set_xlim(-2.0, 6.5)
    ax.set_ylim(0.5, 4.3)
    ax.plot(xc, y_true_c, color='#e5e7eb', lw=6, alpha=0.7, zorder=1)
    ax.text(0.5, 1.7, 'Latent TE landscape', ha='center', fontsize=8.5,
            color='#9ca3af', style='italic')
    ax.plot(xc, y_aln_c, color='#fca5a5', lw=1.8, ls='--', alpha=0.8, zorder=2)
    ax.plot(xc, y_res_c, color='#ef4444', lw=3.5, zorder=3)
    ax.scatter(x_wt, y_wt, s=240, color='black', edgecolor='white', lw=2.2, zorder=15)
    ax.scatter(x_m, y_m, s=100, color='#6b7280', zorder=5)
    ax.scatter(x_m, y_al, s=110, marker='^', color='#2563eb', zorder=6)
    for xi, yr, ya in zip(x_m, y_m, y_al):
        ax.plot([xi, xi], [yr, ya], color='#bfdbfe', lw=1.3, zorder=4)
    ax.annotate('', xy=(5.6, 0.7), xytext=(-1.5, 0.7),
                arrowprops=dict(arrowstyle='->', lw=1.6, color='black'))
    ax.annotate('', xy=(-1.5, 3.95), xytext=(-1.5, 0.7),
                arrowprops=dict(arrowstyle='->', lw=1.6, color='black'))
    ax.text(2.0, 0.42, 'Local Sequence Space ($x$)',
            ha='center', fontsize=13, fontweight='bold')
    ax.text(-1.85, 2.25, 'Translation Efficiency (TE)',
            ha='center', va='center', rotation=90, fontsize=13, fontweight='bold')
    ax.text(0.1, 4.15, 'Soft label: Taylor-Distance Correction (TDC)',
            color='#111827', fontweight='bold', fontsize=11.5)
    ax.text(0.1, 3.08, 'Distance-aware evolutionary soft labeling',
            color='#1e40af', fontweight='bold', fontsize=11)
    ax.text(0.1, 3.50,
            r'$y_{mut} = y_{wt}^{real} + [\mathcal{T}(x_m) - \mathcal{T}(x_{wt})] \cdot \left(\frac{n_{mut}}{n_{max}}\right)^{\!\alpha}$',
            fontsize=16, color='#1e40af')
    ax.text(0.1, 3.22,
            r'$\alpha = 0.3$,  $n_{mut}$: Hamming distance,  $n_{max}$: max mutations per gene',
            fontsize=10, color='#1e40af')
    ax.text(4.85, 3.42, 'Pure-sequence\nInvaRNA Student',
            ha='center', va='center', color='white', fontweight='bold', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.45', facecolor='#ef4444', edgecolor='none'),
            zorder=10)
    ax.text(5.3, 2.75, r'$\mathcal{T}$: teacher prediction',
            fontsize=10, color='#6b7280', ha='center', style='italic')
    ax.text(5.3, 2.40, r'$\alpha$: distance decay exponent',
            fontsize=10, color='#6b7280', ha='center', style='italic')
    ax.legend(handles=legend_d, loc='lower left', frameon=False,
              fontsize=9, bbox_to_anchor=(0.28, 0.01))
    panel_label(ax, 'D', x=-0.05)
    fig_single.tight_layout()
    return fig_single


def _make_panel_e():
    fig_single = plt.figure(figsize=(5.8, 5.8), dpi=150)
    e_single = gridspec.GridSpec(2, 1, figure=fig_single,
                                 height_ratios=[3, 1.2], hspace=0.08)
    ax_bar = fig_single.add_subplot(e_single[0])
    ax_dot = fig_single.add_subplot(e_single[1])
    x_e_local = np.arange(len(E_models))
    ax_bar.bar(x_e_local, E_val_r2, width=0.55, color=E_c, edgecolor='white', linewidth=0.5, zorder=3)
    for xi, v in zip(x_e_local, E_val_r2):
        ax_bar.text(xi, v + 0.002, f'{v:.4f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='#1a1a1a')
    for i in range(1, len(E_val_r2)):
        delta = E_val_r2[i] - E_val_r2[i - 1]
        ax_bar.text(x_e_local[i], E_val_r2[i] + 0.010, f'+{delta:.4f}',
                    ha='center', va='bottom', fontsize=6.5, color='#666666', fontstyle='italic')
    ax_bar.set_ylim(0.62, 0.78)
    ax_bar.set_ylabel('Test R\u00b2', fontsize=9, fontweight='bold')
    ax_bar.set_xticks([])
    ax_bar.spines[['top', 'right', 'bottom']].set_visible(False)
    ax_bar.tick_params(labelsize=8)
    ax_bar.grid(axis='y', alpha=0.15, zorder=0)
    panel_label(ax_bar, 'E', x=-0.18)

    for col in range(len(E_models)):
        for row in range(len(E_labels_dot)):
            color = E_c_on if E_active[col][row] else E_c_off
            ax_dot.scatter(col, len(E_labels_dot) - 1 - row, s=80, color=color,
                           zorder=3, edgecolors='white', linewidths=0.5)
        active_rows = [len(E_labels_dot) - 1 - r for r in range(len(E_labels_dot)) if E_active[col][r]]
        if len(active_rows) > 1:
            ax_dot.plot([col, col], [min(active_rows), max(active_rows)],
                        color=E_c_on, lw=2.0, zorder=2)
    ax_dot.set_yticks(range(len(E_labels_dot)))
    ax_dot.set_yticklabels(E_labels_dot[::-1], fontsize=6.5, fontweight='normal')
    ax_dot.set_xticks([])
    ax_dot.set_xlim(-0.5, len(E_models) - 0.5)
    ax_dot.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax_dot.tick_params(left=False)
    fig_single.tight_layout()
    return fig_single


def _make_panel_f():
    fig_single, ax = plt.subplots(figsize=(7.4, 4.8), dpi=150)
    x_g_local = np.arange(len(G_labels))
    ax.bar(x_g_local, G_scores, width_g, color=G_colors, edgecolor='none', zorder=3)
    for i in range(len(G_labels)):
        if i == len(G_labels) - 1:
            ax.text(x_g_local[i], G_scores[i] + 0.005, f'{G_scores[i]:.3f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='#111111')
        elif i == 5:
            ax.text(x_g_local[i], G_scores[i] + 0.004, f'{G_scores[i]:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_ylim(0.36, 0.82)
    ax.set_ylabel("Held-out Human Test Performance (R\u00b2)", fontsize=11)
    ax.set_xticks(x_g_local)
    tick_labels_local = ax.set_xticklabels(G_labels, rotation=28, ha='right')
    for i, tick in enumerate(tick_labels_local):
        if i < 5:
            tick.set_color('#9CA3AF'); tick.set_fontsize(9); tick.set_fontweight('normal')
        elif i == 5:
            tick.set_color('#4B5563'); tick.set_fontsize(9.5); tick.set_fontweight('semibold')
        else:
            tick.set_color('black'); tick.set_fontsize(10); tick.set_fontweight('bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', length=0)
    panel_label(ax, 'F', x=-0.10)
    fig_single.tight_layout()
    return fig_single


def _make_panel_g():
    fig_single, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    x_f_local = np.arange(len(model_order_f))
    f_means_local, f_sds_local = [], []
    for model_name in model_order_f:
        vals = results_f[results_f['Model'] == model_name]['Spearman_R'].dropna()
        f_means_local.append(vals.mean())
        f_sds_local.append(vals.std())
    ax.bar(x_f_local, f_means_local, yerr=f_sds_local, capsize=6, width=0.55,
           color=[palette_f[m] for m in model_order_f], edgecolor='none',
           error_kw=dict(lw=1.2), zorder=3)
    for i, value in enumerate(f_means_local):
        ax.text(x_f_local[i], value + f_sds_local[i] + 0.015, f'{value:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x_f_local)
    ax.set_xticklabels(model_order_f, fontsize=10, fontweight='bold')
    ax.set_ylabel('Spearman \u03c1 on external MPRA\n(mean \u00b1 SD, n=10)', fontsize=10)
    ax.set_ylim(0, 0.55)
    ax.axhline(0, color='black', linewidth=1.0)
    ax.grid(False)
    spine_off(ax)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', length=0)
    panel_label(ax, 'G', x=-0.18)
    fig_single.tight_layout()
    return fig_single


plt.close(fig)
print("Done: combined figure rendered from checked Fig. 3 tables.")
