# 2026-11-05/06: why long context helps Timestral and not our models

Owner's questions after the 5-minute long-context report: why does longer context not help, why
does Timestral work and our models do not, how do TCN / PatchTST etc. behave as context grows,
what does fine-tuning Timestral do, is it the amount of data (download 2-5 years of I2), and
how does our preprocessing compare with the ladder's (log1p + RevIN + Huber) and Timestral's.
Times CET; nothing committed.

Report: `docs/reports/why-long-context.html` (published as a claude.ai artifact, link in AGENTS.md).
Code (`recipes/ladder_diagnosis/`): `i2_history.py` (I2 hourly 2021 -> 2024 download and link /
door loads), `hourly_train.py` (`--hist ext21`, `--train-months`, `--extra i2doors`,
`--eval-bs`, models `dlinear` / `gru` / `tcn`), `long5m.py` (`--norm lastmean|log_revin|ts`,
`--loss l1|huber`, `--save`, context ablation), `uses_context.py`, `decompose.py`,
`ft_timestral.py`. Grids: `l5e_*` (preprocessing), `l5f` (saved models), `h_arch*`
(architectures), `h_data` (data size), `h_doors` (more series). Output under
`artifacts/ladder_diagnosis/` (`i2/data_size_runs.csv`, `*/uses_context_*.csv`, runs).

## Answer

Long history is useful for one thing: a better estimate of each link's level (daily volume) and
daily shape. Timestral gets it (blanking its weeks 5-24 costs 0.75 / 1.44 points on the I2 / Switch
test blocks; from 4 to 24 weeks its volume error falls 12.09 -> 11.47 and shape 13.53 -> 12.91 on I2,
volume 18.94 -> 17.46 on Switch). Our trained models read the old weeks too (forecasts move 3-8 %
when they are blanked, Timestral 4-7 %) and fit the training period better with them (I2: 24 w
17.80 vs 4 w 18.44 on the last 13 training weeks), but that does not carry over (validation 23.30
vs 22.86, test 18.30 vs 18.00). More data removes this damage (PatchTST 24 w minus 4 w: +2.94 with 6
months of I2, +1.26 with 1 year, +0.45 with 2, +0.37 with 5, +0.19 with 5 years + 567 door series) but
no gain appears, and the 4-week models do not improve with data (18.3 at every size); the best
trained model (profile MLP 24 w, 5 years + doors) is 18.00 vs Timestral 16.22 (+1.78 [+0.88,
+2.31]). Preprocessing is not the gap (our scheme beats the ladder's and Timestral's inside the same
transformer), and fine-tuning Timestral on 5 years of I2 or on Switch changes nothing (+-0.15, n.s.;
training loss flat at ~0.10). The missing piece is a prior for reading long, noisy history, which
one network's data cannot teach.

## Results

Data (task 1): Internet2's current core network reports hourly from 2021-05 (probe of
core1.chic: no backbone ports before 2021-06, all 16 from then). `i2_history.py fetch`: 73 routers x
8 half-years, 584 queries, 557 MB, ~70 min at 1 s pause. Link loads equal the recipe's file over
2024-11 (median relative difference 0, 222 common links). For the 113 modelled links: training hours
14,160 (ext) -> 44,912 (ext21), 96 links reporting from 2022. Doors: 567 door directions with >= 20 %
of the training hours and >= 1 Mbit/s (414 at >= 50 %). Switch LAN's 5-minute export starts 2023-12-29
(350 of 534 links), so no further Switch history exists.

Context use (`uses_context.py`, day-ahead median link hourly WAPE; train = last 13 training weeks):

| | I2 train | val | test | Switch train | val | test |
|---|---|---|---|---|---|---|
| trained 4 w (multires s3) | 18.44 | 22.86 | 18.00 | 31.23 | 29.27 | 31.46 |
| trained 24 w (multires s3) | 17.80 | 23.30 | 18.30 | 30.89 | 28.25 | 31.25 |
| same, weeks 5-24 blanked | 19.41 | 24.16 | 20.29 | 31.85 | 29.12 | 31.35 |
| Timestral 24 w | 16.60 | 20.29 | 16.22 | 27.99 | 24.46 | 28.05 |
| Timestral, weeks 5-24 blanked | 17.55 | 22.26 | 16.97 | 29.07 | 25.59 | 29.49 |

Coarse 24 w and profile MLP 24 w (s3): I2 test 18.69 / 18.97, blanked 21.17 / 19.52. "Tile" (weeks
5-24 = copies of the last 4) is unusable for Timestral: it repeats the exact 4-week pattern (37-55 %).

Data size (I2 hourly, ext21, 2 seeds, day; `h_data`, `h_doors`): PatchTST 4 w 18.73 / 18.45 / 18.27 /
18.43 / 18.36 and 24 w 22.00 / 20.10 / 19.04 / 19.02 / 18.66 for 6 / 12 / 24 / 61 months / 61 + doors;
profile MLP 4 w 18.55 / 18.29 / 18.38 / 18.42 / 18.25, 24 w 19.08 / 18.35 / 18.25 / 18.45 / 18.04.
Paired 24 w minus 4 w (PatchTST, 2-seed ensembles): +2.94 [1.85, 3.23], +1.26 [0.78, 2.08], +0.45
[0.21, 1.23], +0.37 [-0.05, 0.97], +0.19 [-0.21, 0.87]. Seed spread <= 0.27.

Volume / shape (`decompose.py`, day, median link): I2 Timestral 1/4/24 w volume 14.29 / 12.09 / 11.47,
shape 16.26 / 13.53 / 12.91; PatchTST (5 years) 12.78 / 12.57 / 12.47 and 15.33 / 14.10 / 14.75. Switch
Timestral 26.16 / 18.94 / 17.46 and 27.28 / 24.43 / 23.97; 5-minute transformer (3 seeds) 22.23 / 20.82 /
20.44 and 27.37 / 25.81 / 25.82.

Fine-tuning (`ft_timestral.py`, 24 w hourly, bs 16, 300 steps/epoch): I2 ext21 LR 1e-5 16.13 / 20.55
(-0.10 [-0.25, +0.12]), LR 3e-5 16.07 / 20.55 (-0.15 [-0.30, +0.21]); Switch LR 1e-5 28.08 / 32.40
(+0.03), LR 3e-5 = zero-shot (best checkpoint epoch 0). Validation 20.29 -> 20.19-20.28 (I2), 24.46 ->
24.36 (Switch).

Preprocessing inside the same 5-minute transformer (`l5e`, day): ladder (log1p + RevIN + Huber) I2
22.96 / 18.44 / 19.76 at 1 / 4 / 24 w, Switch 38.25 / 32.90 / 33.19; Timestral's scaler (its
`causal_patch_loc_scale` on 1-hour patches + asinh, target in the last patch's frame, L1) I2 20.72 /
19.48 / 22.48, Switch 33.08 / 31.73 / 32.27; ours 19.16 / 17.99 / 18.66 and 32.88 / 31.50 / 31.57.

Architectures (hourly, ext history, one budget; day, I2 / Switch): DLinear 21.95 / 20.68 / 20.45 and
37.40 / 35.93 / 36.11; PatchTST 18.85 / 18.35 / 19.51 and 32.73 / 32.49 / 33.25; GRU 19.49 / 19.23 / – and
36.21 / 35.79 / 35.47; TCN 19.99 / 19.48 / 18.84 and 35.40 / 36.28 / 36.49. Paired 24 vs 4 w: I2 GRU -0.43 [-1.06,
-0.03], TCN -0.64 [-0.71, +0.13], DLinear -0.23, PatchTST +1.16 [+0.84, +2.04]; Switch GRU -0.33, TCN +0.20, DLinear +0.18
(n.s.), PatchTST +0.76 [+0.35, +2.42]. The small models (0.2 M parameters) are not hurt; PatchTST's flatten head
over-fits. Best 24 w GRU / TCN vs PatchTST 4 w on I2: +0.45 / +0.49 (n.s.).

## Log

- 11-05 ~10:00 Preprocessing grid (`l5e`), I2 probe (core router names need the FQDN), download
  started; architecture grid `h_arch` (DLinear/GRU/TCN added to the harness; TCN without weight
  decay on weight-norm directions, pitfall 1).
- ~11:15 Download done; link build; `h_data` (36 runs). Saved-model runs `l5f`; `uses_context.py` for
  Timestral (both networks) and the saved models; fine-tunes queued.
- ~12:30 Door series added (`--extra i2doors`), `h_doors`.
- Six GRU / TCN runs (4 and 24 w) ran out of GPU memory next to six other jobs (validation batch
  4,096); rerun 11-06 with `--eval-bs 256` (`grids/h_arch_{i2,sw}.txt`).
- 11-06 13:50 `decompose.py`; report drafted and published. GRU / TCN reruns done 14:55; the Switch 24 w
  GRU ran out of memory a second time next to the I2 one and was rerun alone (`grids/h_arch_sw2.txt`).

## Not done / caveats

- Seeds: 2 for the data-size and door runs, 1 for most others.
- The old I2 network (2016-2022, different routers) was not used.
- Timestral's pretraining data are not described in the files we have.
- Doors are training-only series; they are customer ports, not backbone links.
