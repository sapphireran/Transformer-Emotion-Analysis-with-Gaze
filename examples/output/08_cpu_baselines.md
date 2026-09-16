# CPU baselines

Ridge one-vs-rest, 5 stratified folds, seed 42. Not RoBERTa.

## ZuCo
| model | mean acc | acc sd | mean weighted F1 |
| --- | --- | --- | --- |
| majority | 0.3500 | 0.0039 | 0.1815 |
| length only | 0.3351 | 0.0282 | 0.2383 |
| gaze only | 0.4104 | 0.0495 | 0.3895 |
| gaze ⟂ length | 0.4104 | 0.0483 | 0.4053 |
| hashed text | 0.3575 | 0.0229 | 0.3541 |
| text + gaze | 0.3624 | 0.0246 | 0.3624 |
| text + shuffled gaze | 0.3801 | 0.0333 | 0.3786 |

## Full SST
| model | mean acc | acc sd | mean weighted F1 |
| --- | --- | --- | --- |
| majority | 0.4187 | 0.0001 | 0.2472 |
| length only | 0.4197 | 0.0013 | 0.2576 |
| gaze only | 0.4231 | 0.0094 | 0.3623 |
| gaze ⟂ length | 0.4231 | 0.0097 | 0.3658 |
| hashed text | 0.5037 | 0.0013 | 0.4519 |
| text + gaze | 0.5043 | 0.0036 | 0.4525 |
| text + shuffled gaze | 0.5038 | 0.0030 | 0.4520 |
