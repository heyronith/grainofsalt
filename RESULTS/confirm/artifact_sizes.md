# Confirm Volume artifacts (pre–remaining extracts)
| arm | ckpt ep0 | ckpt ep8 | act ep0 | act ep8 | endorsement | extract_complete |
|---|---:|---:|---:|---:|---:|:---:|
| CORE_K1 | 2.77 GiB | 2.77 GiB | 132.82 MiB | 132.82 MiB | 84.73 KiB | True |
| MIS_K1 | 2.77 GiB | 2.77 GiB | 132.82 MiB | 132.82 MiB | 91.32 KiB | True |
| NH_K1 | 2.77 GiB | 2.77 GiB | 132.82 MiB | 132.82 MiB | 90.20 KiB | True |
| CORE_K2 | 2.77 GiB | 2.77 GiB | 132.82 MiB | 132.82 MiB | 92.38 KiB | True |
| MIS_K2 | 2.77 GiB | 2.77 GiB | 132.82 MiB | 132.82 MiB | 0 B | False |
| NH_K2 | 2.77 GiB | 2.77 GiB | 0 B | 0 B | 0 B | False |
| CORE_K3 | 2.77 GiB | 2.77 GiB | 0 B | 0 B | 0 B | False |
| MIS_K3 | 2.77 GiB | 2.77 GiB | 0 B | 0 B | 0 B | False |
| NH_K3 | 2.77 GiB | 2.77 GiB | 0 B | 0 B | 0 B | False |

n_ckpt_ok=9 n_extract_complete=4
All 9 arms have epoch-0/2/4/8 checkpoints (~2.77 GiB each).
