# Gaze feature analysis

Computed from committed CSVs. Model concat uses ZuCo `['nFixations', 'FFD', 'GPT', 'TRT', 'GD']` and SST `['nFix', 'FFD', 'GPT', 'TRT', 'GD']`.

## ZuCo correlations

|  | `omissionRate` | `nFixations` | `meanPupilSize` | `GD` | `TRT` | `FFD` | `SFD` | `GPT` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `omissionRate` | 1.000 | -0.257 | 0.171 | -0.026 | -0.201 | -0.095 | 0.438 | -0.229 |
| `nFixations` | -0.257 | 1.000 | 0.055 | 0.697 | 0.958 | 0.421 | -0.585 | 0.912 |
| `meanPupilSize` | 0.171 | 0.055 | 1.000 | 0.012 | 0.047 | -0.057 | 0.017 | 0.041 |
| `GD` | -0.026 | 0.697 | 0.012 | 1.000 | 0.789 | 0.679 | -0.146 | 0.675 |
| `TRT` | -0.201 | 0.958 | 0.047 | 0.789 | 1.000 | 0.617 | -0.420 | 0.942 |
| `FFD` | -0.095 | 0.421 | -0.057 | 0.679 | 0.617 | 1.000 | 0.172 | 0.547 |
| `SFD` | 0.438 | -0.585 | 0.017 | -0.146 | -0.420 | 0.172 | 1.000 | -0.444 |
| `GPT` | -0.229 | 0.912 | 0.041 | 0.675 | 0.942 | 0.547 | -0.444 | 1.000 |

### ZuCo r(feature, sentiment_label)

| Feature | r vs label |
| --- | ---: |
| `FFD` | 0.0715 |
| `TRT` | 0.0509 |
| `SFD` | 0.0442 |
| `meanPupilSize` | -0.0412 |
| `GD` | 0.0356 |
| `nFixations` | 0.0326 |
| `GPT` | 0.0304 |
| `omissionRate` | 0.0280 |

Pairs with |r| ≥ 0.90: [('nFixations', 'TRT', 0.958147130101129), ('TRT', 'GPT', 0.9418588018829125), ('nFixations', 'GPT', 0.9124615276188492)]

## Full SST predicted correlations

|  | `nFix` | `FFD` | `GPT` | `TRT` | `GD` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nFix` | 1.000 | 0.995 | 0.998 | 0.986 | 0.783 |
| `FFD` | 0.995 | 1.000 | 0.999 | 0.995 | 0.741 |
| `GPT` | 0.998 | 0.999 | 1.000 | 0.995 | 0.744 |
| `TRT` | 0.986 | 0.995 | 0.995 | 1.000 | 0.680 |
| `GD` | 0.783 | 0.741 | 0.744 | 0.680 | 1.000 |

### SST r(feature, sentiment_label)

| Feature | r vs label |
| --- | ---: |
| `GD` | -0.0600 |
| `nFix` | -0.0552 |
| `FFD` | -0.0549 |
| `GPT` | -0.0539 |
| `TRT` | -0.0518 |

Pairs with |r| ≥ 0.95: [('FFD', 'GPT', 0.9985143869006835), ('nFix', 'GPT', 0.9975614671497182), ('nFix', 'FFD', 0.9954810572006118), ('FFD', 'TRT', 0.9952543095372755), ('GPT', 'TRT', 0.9946008846037939), ('nFix', 'TRT', 0.9860190968687664)]

## Scaling checks

- ZuCo standard columns look z-scored: {'omissionRate': True, 'nFixations': True, 'meanPupilSize': True, 'GD': True, 'TRT': True, 'FFD': True, 'SFD': True, 'GPT': True}
- SST predicted columns look z-scored: {'nFix': True, 'FFD': True, 'GPT': True, 'TRT': True, 'GD': True}
- ZuCo min-max columns in [0, 1]: {'omissionRate': True, 'nFixations': True, 'meanPupilSize': True, 'GD': True, 'TRT': True, 'FFD': True, 'SFD': True, 'GPT': True}
