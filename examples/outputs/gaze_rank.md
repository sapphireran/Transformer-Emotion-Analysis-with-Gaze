# Gaze sidecar rank

The training models always feed **five** gaze scalars into `nn.Linear(5, 16)`.
On the committed full-SST train table those five columns are almost a single
direction: nFix, TRT, FFD, and GPT correlate at r > 0.98. GD is the only
column with a distinct residual (PC2).

A 16-d linear sidecar is therefore **overcomplete**. Unless the projection
learns to isolate the GD residual, most of those 16 hidden units are
rotated copies of one reading-time axis.

## Full SST train (z-scored sentence gaze)

n = 9482, features = ['nFix', 'GD', 'TRT', 'FFD', 'GPT']

| component | explained | cumulative | singular |
| --- | --- | --- | --- |
| PC1 | 0.9168 | 0.9168 | 208.4829 |
| PC2 | 0.0823 | 0.9991 | 62.4695 |
| PC3 | 0.0006 | 0.9997 | 5.4683 |
| PC4 | 0.0002 | 0.9999 | 3.1096 |
| PC5 | 0.0001 | 1.0000 | 1.6946 |

Loadings (rows are principal axes, columns are original features):

| component | nFix | GD | TRT | FFD | GPT |
| --- | --- | --- | --- | --- | --- |
| PC1 | -0.4661 | -0.3788 | -0.4572 | -0.4636 | -0.4640 |
| PC2 | -0.0783 | 0.9119 | -0.3144 | -0.1813 | -0.1748 |
| PC3 | 0.6869 | -0.1242 | -0.4291 | -0.4798 | 0.3137 |
| PC4 | 0.1544 | -0.0971 | -0.6482 | 0.7221 | -0.1587 |
| PC5 | -0.5301 | 0.0091 | -0.2963 | 0.0230 | 0.7941 |

Correlation matrix:

| feature | nFix | GD | TRT | FFD | GPT |
| --- | --- | --- | --- | --- | --- |
| nFix | 1.0000 | 0.7796 | 0.9860 | 0.9954 | 0.9975 |
| GD | 0.7796 | 1.0000 | 0.6761 | 0.7370 | 0.7400 |
| TRT | 0.9860 | 0.6761 | 1.0000 | 0.9953 | 0.9947 |
| FFD | 0.9954 | 0.7370 | 0.9953 | 1.0000 | 0.9985 |
| GPT | 0.9975 | 0.7400 | 0.9947 | 0.9985 | 1.0000 |

## ZuCo-SST combined, the five columns the model actually consumes

n = 400, features = ['nFixations', 'FFD', 'GPT', 'TRT', 'GD']

| component | explained | cumulative | singular |
| --- | --- | --- | --- |
| PC1 | 0.7864 | 0.7864 | 39.6579 |
| PC2 | 0.1412 | 0.9275 | 16.8025 |
| PC3 | 0.0560 | 0.9835 | 10.5792 |
| PC4 | 0.0139 | 0.9974 | 5.2762 |
| PC5 | 0.0026 | 1.0000 | 2.2732 |

Loadings (rows are principal axes, columns are original features):

| component | nFixations | FFD | GPT | TRT | GD |
| --- | --- | --- | --- | --- | --- |
| PC1 | -0.4634 | -0.3594 | -0.4705 | -0.4953 | -0.4351 |
| PC2 | -0.4309 | 0.7835 | -0.2783 | -0.1702 | 0.3065 |
| PC3 | -0.0247 | 0.4485 | 0.3587 | 0.0729 | -0.8150 |
| PC4 | -0.5407 | -0.1780 | 0.7448 | -0.2678 | 0.2224 |
| PC5 | -0.5536 | -0.1552 | -0.1330 | 0.8054 | -0.0552 |

Correlation matrix:

| feature | nFixations | FFD | GPT | TRT | GD |
| --- | --- | --- | --- | --- | --- |
| nFixations | 1.0000 | 0.4213 | 0.9125 | 0.9581 | 0.6973 |
| FFD | 0.4213 | 1.0000 | 0.5470 | 0.6167 | 0.6794 |
| GPT | 0.9125 | 0.5470 | 1.0000 | 0.9419 | 0.6746 |
| TRT | 0.9581 | 0.6167 | 0.9419 | 1.0000 | 0.7893 |
| GD | 0.6973 | 0.6794 | 0.6746 | 0.7893 | 1.0000 |

ZuCo still has TRT ≈ nFixations ≈ GPT, but FFD vs SFD is only weakly
related (SFD is *not* in the five-column model input). Mean pupil size
and omission rate — also unused by `EyeTrackingModel` — are the more
independent physiological channels sitting in the CSV.
