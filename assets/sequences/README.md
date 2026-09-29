# Fig. 5 fixed sequence components

This directory contains only the WT/reference components needed to define the
three RL starting contexts. No optimized candidates, screening constructs, or
historical optimization traces are distributed.

| File | Role | Length |
|---|---|---:|
| `blnk_wt.fa` | BLNK WT transcript context (5′UTR + CDS + 3′UTR) | 4344 nt |
| `blnk_wt_utr5.fa` | BLNK WT 5′UTR (optimized region starting point) | 171 nt |
| `hba_utr3.fa` | Fixed HBA/α-globin 3′UTR for BLNK 5′UTR optimization | 111 nt |
| `hbb_wt.fa` | HBB WT transcript context (5′UTR + CDS + 3′UTR) | 628 nt |
| `hbb_wt_utr5.fa` | Fixed HBB native 5′UTR for HBB 3′UTR optimization | 50 nt |
| `hbb_wt_utr3.fa` | HBB WT 3′UTR (optimization starting point) | 134 nt |
| `ngf_wt.fa` | NGF WT transcript context (5′UTR + CDS + 3′UTR; smoke selects the `_cds` record) | 1061 nt |
| `gluc_cds.fa` | Fixed Gaussia luciferase reporter CDS | 558 nt |

The BLNK context is `BLNK WT 5′UTR + GLuc CDS + HBA 3′UTR`. The HBB context is
`HBB native 5′UTR + GLuc CDS + HBB WT 3′UTR`.
