# Full-SST test-loop overwrite

In ``model_full_SST.py`` the validation loop extends prediction lists. The test loop assigns ``all_preds = preds.cpu().numpy()``, so only the last batch survives. Default ``batch_size=256`` and 1186 test rows → last batch is 162 examples.

## Simulated numbers

| quantity | value |
| --- | --- |
| items | 1186 |
| batch size | 256 |
| batches | 5 |
| overwrite coverage | 162 |
| full-set accuracy (extend) | 0.8642 |
| printed-style accuracy (overwrite) | 0.0062 |

overwrite_accuracy is computed on the last batch only, which is what the current test loop prints. extend_accuracy is the full-set number.

The dummy model is correct everywhere except the last batch, so the two aggregation rules disagree on purpose. The real network will not look like this; the point is coverage, not the dummy accuracy values.

## Other known issues (not patched here)

| id | file | severity | summary |
| --- | --- | --- | --- |
| full-sst-test-overwrite | model_full_SST.py | high | Test loop assigns all_preds = batch instead of extend. |
| checkpoint-comment-f1 | model_full_SST.py | low | Comment and save log say F1; the predicate uses validation accuracy. |
| hardcoded-three-classes | model_ZuCo_SST.py / model_full_SST.py | low | Fusion loss uses logits.view(-1, 3) even though num_labels exists. |
| subject3-reindex | utils_ZuCo.py + get_average_sentence_level.py | high | Subject 3 drops sentences 150–249 and 399, then reindexes 0..298. |
| path-leftovers | read_ZuCo_mat.py, convert_full_SST.py, utils_ZuCo.py | medium | Producers point at folders that are not in this clone. |
| spilt-typo | ZuCo_SST_data/spilt.py, SST_data/spilt.py | low | Filename is spilt, not split. |
