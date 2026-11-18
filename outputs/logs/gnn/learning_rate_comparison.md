# Comparison of GraphSAGE learning-rate trials

Inspected trial001 (learning rate 0.001) and trial002 (0.0005). Both use hidden size 128, fanout [10, 5], batch 1024, dropout 0.2, weight decay 0.0001, gradient clipping 1.0, maximum 80 epochs and patience 10. Both use initialization seeds 0–4 and wallet split seed 0. Other resolved model settings and package versions match.

## Completion and failures

Each trial completed 22 of 24 model × k × calendar-period jobs: all 12 GraphSAGE jobs and 10 of 12 heterogeneous GraphSAGE jobs. Each completed job has five fitted seeds, giving 110 successful fits per trial (220 total). Heterogeneous GraphSAGE failed for the later period (`test`) at k=20 and k=50 in both trials, before the first model seed began training.

The errors are `query.rev_held.edge_attr: nonfinite input`, first reported at (803753, 3) for k=20 and (1387827, 3) for k=50. Attribute column 3 (zero based) is `log_h_to_close`, the log time-to-close feature on a market-to-query-wallet edge. These are CSR edge-array positions, not wallet ids. The logs do not distinguish NaN from infinity or identify the original raw row. Missing/invalid closing-time data are a candidate source; the original prepared data are required to establish the exact cause. The errors are input validation failures, not CUDA out-of-memory errors or loss divergence. Identical failures under both learning rates do not support reducing the learning rate as a remedy.

The launcher returns before validation/test reporting when any job fails. Neither trial folder contains val_report.log or test_report.log. Successful jobs saved scores according to their logs, but the prediction files and prepared labels are absent from this local data/processed folder. Final metrics cannot be reconstructed from these text logs alone.

## What the scores below measure

These numbers are the mean and sample standard deviation, over five initialization seeds, of each fitted model’s best early-stopping validation Pearson correlation. They use the half-A normalized continuous edge target, combining new and existing validation wallets that have usable targets. They are not classification accuracy, a confidence interval, independent test scores, or the final half-B weighted edge correlation. The same validation set selects the checkpoint, so best scores are selection-biased.

`train` and `test` in filenames refer to the earlier and later calendar datasets. Each dataset separately has 60% fitting / 20% validation / 20% test wallets. A file starting with `test_` is not by itself a final held-out test evaluation, and these fits do not measure earlier-to-later model transfer.

## All configurations

Each cell is mean ± sample SD across five seeds. Delta is lower learning rate minus higher learning rate.

| Calendar period | k | Model | LR 0.001 | LR 0.0005 | Delta |
|---|---:|---|---:|---:|---:|
| train | 1 | GraphSAGE | 0.32992 ± 0.00266 | 0.32956 ± 0.00150 | -0.00036 |
| train | 1 | Heterogeneous GraphSAGE | 0.34149 ± 0.00193 | 0.34166 ± 0.00402 | +0.00017 |
| train | 3 | GraphSAGE | 0.35407 ± 0.00415 | 0.35424 ± 0.00301 | +0.00018 |
| train | 3 | Heterogeneous GraphSAGE | 0.36483 ± 0.00262 | 0.36787 ± 0.00191 | +0.00304 |
| train | 5 | GraphSAGE | 0.37550 ± 0.00205 | 0.37296 ± 0.00330 | -0.00253 |
| train | 5 | Heterogeneous GraphSAGE | 0.38345 ± 0.00194 | 0.38525 ± 0.00247 | +0.00180 |
| train | 10 | GraphSAGE | 0.36473 ± 0.00568 | 0.36295 ± 0.00498 | -0.00178 |
| train | 10 | Heterogeneous GraphSAGE | 0.36632 ± 0.00332 | 0.36751 ± 0.00171 | +0.00119 |
| train | 20 | GraphSAGE | 0.41763 ± 0.00390 | 0.41369 ± 0.00447 | -0.00395 |
| train | 20 | Heterogeneous GraphSAGE | 0.42095 ± 0.00397 | 0.42099 ± 0.00608 | +0.00003 |
| train | 50 | GraphSAGE | 0.40193 ± 0.00608 | 0.40374 ± 0.00396 | +0.00182 |
| train | 50 | Heterogeneous GraphSAGE | 0.40058 ± 0.00704 | 0.40287 ± 0.00684 | +0.00229 |
| test | 1 | GraphSAGE | 0.35062 ± 0.00077 | 0.35020 ± 0.00102 | -0.00042 |
| test | 1 | Heterogeneous GraphSAGE | 0.36129 ± 0.00180 | 0.36114 ± 0.00136 | -0.00016 |
| test | 3 | GraphSAGE | 0.36699 ± 0.00199 | 0.36483 ± 0.00142 | -0.00216 |
| test | 3 | Heterogeneous GraphSAGE | 0.37971 ± 0.00107 | 0.38037 ± 0.00099 | +0.00066 |
| test | 5 | GraphSAGE | 0.37359 ± 0.00194 | 0.37386 ± 0.00147 | +0.00026 |
| test | 5 | Heterogeneous GraphSAGE | 0.38659 ± 0.00156 | 0.38653 ± 0.00055 | -0.00005 |
| test | 10 | GraphSAGE | 0.41125 ± 0.00147 | 0.41003 ± 0.00223 | -0.00122 |
| test | 10 | Heterogeneous GraphSAGE | 0.41623 ± 0.00110 | 0.41494 ± 0.00052 | -0.00130 |
| test | 20 | GraphSAGE | 0.44478 ± 0.00299 | 0.44297 ± 0.00361 | -0.00181 |
| test | 20 | Heterogeneous GraphSAGE | Failed before fitting | Failed before fitting | — |
| test | 50 | GraphSAGE | 0.48146 ± 0.00289 | 0.48216 ± 0.00429 | +0.00069 |
| test | 50 | Heterogeneous GraphSAGE | Failed before fitting | Failed before fitting | — |

GraphSAGE slightly favors 0.001 in 8 of 12 configurations; heterogeneous GraphSAGE slightly favors 0.0005 in 7 of 10 completed configurations. Every learning-rate change is less than 0.004 correlation units in magnitude. These mixed, small differences do not establish a reliable overall winner. Initialization SD does not measure uncertainty over the wallet population. Paired half-B validation evaluation is needed for tuning decisions.

Heterogeneous GraphSAGE has higher mean early-stopping validation correlation than GraphSAGE in 9 of 10 matched completed settings at each learning rate. Its improvement is generally around 0.005–0.016 on the later-period k=1–10 settings. This is encouraging evidence for the additional relations under this early-stopping objective, but it does not demonstrate a final new-wallet or held-out-test advantage. Earlier-period k=50 slightly favors ordinary GraphSAGE.

## Learning curves and resources

All 220 successful fits early-stopped before the 80-epoch cap. They ran 19–51 epochs; their best checkpoints occurred at epochs 9–41. Every run ended exactly 10 epochs after its best checkpoint, matching patience 10. Increasing the maximum epoch cap alone would not extend these runs. Learning curves typically show decreasing training loss with validation scores plateauing or fluctuating: early stopping is working.

| Trial | Model | Fitted seeds | Parameters per model | Sum of fit/inference seconds | Maximum CUDA allocated GiB | Maximum CUDA reserved GiB | Maximum process peak RAM GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| trial001 | sage | 60 | 70,273 | 988.2 | 0.164 | 0.250 | 5.759 |
| trial001 | hgnn | 50 | 339,073 | 3119.9 | 1.221 | 2.342 | 4.807 |
| trial002 | sage | 60 | 70,273 | 1148.5 | 0.164 | 0.250 | 5.421 |
| trial002 | hgnn | 50 | 339,073 | 3318.0 | 1.221 | 2.359 | 4.821 |

All successful fits report CUDA on NVIDIA GeForce RTX 4090. Memory values are PyTorch allocator statistics, not whole-card usage including driver/context/other processes. CPU RAM is a per-process high-water mark. Summed elapsed fit times are not wall-clock duration because jobs run in parallel, and omit graph loading/preprocessing before each fit. They include fitting and prediction, not measured GPU-active compute time. The eight GPUs run independent jobs; their memory is not pooled.

## Recommended interpretation and next step

The successful runs show repeatable learning; neither learning rate has a clear decisive advantage. Fix and diagnose the invalid time-to-close edge input before spending on wider tuning. Preserve existing successful checkpoints/predictions. Because trial tags fingerprint source/data/settings, a corrected code/data run needs a new tag or an explicit, validated migration of old artifacts; simply reusing an old tag after a source change is rejected.

After a complete corrected run, compare configurations using half-B validation metrics (especially wcorr_edge for new wallets), then evaluate the selected configuration on held-out test wallets. Do not select the learning rate by repeatedly inspecting final test scores. Existing historical exp4 baseline metrics used a different evaluation protocol and cannot be used as a direct tuning-improvement benchmark here.
