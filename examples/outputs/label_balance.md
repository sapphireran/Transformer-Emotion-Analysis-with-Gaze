# Label balance

| table         | n     | neg  | neu  | pos  | majority     | majority acc |
|---------------|-------|------|------|------|--------------|--------------|
| zuco_text     | 400   | 123  | 137  | 140  | positive (2) | 0.3500       |
| zuco_standard | 400   | 123  | 137  | 140  | positive (2) | 0.3500       |
| zuco_minmax   | 400   | 123  | 137  | 140  | positive (2) | 0.3500       |
| zuco_train    | 320   | 103  | 107  | 110  | positive (2) | 0.3438       |
| zuco_valid    | 40    | 7    | 14   | 19   | positive (2) | 0.4750       |
| zuco_test     | 40    | 13   | 16   | 11   | neutral (1)  | 0.4000       |
| sst_combined  | 11853 | 4649 | 2241 | 4963 | positive (2) | 0.4187       |
| sst_train     | 9482  | 3710 | 1833 | 3939 | positive (2) | 0.4154       |
| sst_valid     | 1185  | 476  | 209  | 500  | positive (2) | 0.4219       |
| sst_test      | 1186  | 463  | 199  | 524  | positive (2) | 0.4418       |

- Majority accuracy is the number a classifier must beat.
- ZuCo combined is nearly balanced; the 40-row valid/test slices are not.
- Full SST is short on neutrals (~19%), so prefer macro F1 over weighted F1.
