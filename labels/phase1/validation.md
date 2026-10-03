Excluded 1 episode(s) with empty clips: ['ep168']

## Detector vs hand label (labeled sample)

| detector        | label                 |   n |   label_pos |   flag_pos |   precision |   recall |    f1 |   kappa |   pop_precision |   pop_recall |
|:----------------|:----------------------|----:|------------:|-----------:|------------:|---------:|------:|--------:|----------------:|-------------:|
| flag_truncation | truncated             | 199 |           1 |         59 |       0.017 |    1     | 0.033 |   0.024 |           0.003 |        1     |
| flag_idle_start | idle_start            | 199 |           0 |          8 |       0     |    0     | 0     | nan     |           0     |        0     |
| flag_idle_end   | idle_end              | 199 |           0 |         10 |       0     |    0     | 0     | nan     |           0     |        0     |
| flag_flailing   | flailing              | 199 |           0 |         29 |       0     |    0     | 0     | nan     |           0     |        0     |
| flag_spike      | glitch                | 199 |           0 |         18 |       0     |    0     | 0     | nan     |           0     |        0     |
| flag_any        | any problem or failed | 199 |          38 |         95 |       0.232 |    0.579 | 0.331 |   0.08  |           0.298 |        0.429 |

## Do detector flags predict hand-labeled failure?

Failure = task completed 'no' (episodes labeled 'unclear' excluded). Fisher exact test.

| detector        |   n_flagged |   fail_rate_flagged |   fail_rate_unflagged |   odds_ratio |   p_value |
|:----------------|------------:|--------------------:|----------------------:|-------------:|----------:|
| flag_truncation |          57 |               0.175 |                 0.208 |        0.812 |     0.693 |
| flag_idle_start |           8 |               0.625 |                 0.179 |        7.656 |     0.008 |
| flag_idle_end   |           9 |               0.444 |                 0.185 |        3.515 |     0.078 |
| flag_flailing   |          27 |               0.148 |                 0.206 |        0.669 |     0.607 |
| flag_spike      |          17 |               0.294 |                 0.188 |        1.797 |     0.337 |
| flag_any        |          91 |               0.231 |                 0.167 |        1.5   |     0.359 |

## Within length tertiles (F1)

| detector        |   short |   mid |   long |
|:----------------|--------:|------:|-------:|
| flag_truncation |   0.083 |     0 |      0 |
| flag_idle_start |   0     |     0 |      0 |
| flag_idle_end   |   0     |     0 |      0 |
| flag_flailing   |   0     |     0 |      0 |
| flag_spike      |   0     |     0 |      0 |

## VLM judge

AUROC vs hand-labeled completion: 0.693 (n=186)
- flag if p_yes < 0.3: precision 0.24, recall 0.89, kappa 0.09
- flag if p_yes < 0.5: precision 0.23, recall 0.95, kappa 0.08
- flag if p_yes < 0.7: precision 0.23, recall 1.00, kappa 0.08

## Hand-label base rates (sample)

|                |     0 |
|:---------------|------:|
| lab_glitch     | 0     |
| lab_wrong_task | 0.025 |
| lab_idle_end   | 0     |
| lab_flailing   | 0     |
| lab_truncated  | 0.005 |
| lab_idle_start | 0     |
| lab_failed     | 0.186 |
| lab_bad        | 0.191 |
