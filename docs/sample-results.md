# Sample results

Filled by the first green `python3 examples/run_all.py` on this branch.
Until that run lands, treat this file as a stub and read
`examples/output/` after you execute the suite.

Locked targets the suite already asserts (independent of the ridge
numbers):

- ZuCo labels 123 / 137 / 140; full SST 4649 / 2241 / 4963
- Reader 3: 299 sentence rows, 5,293 word rows
- nFixations contamination after remap: 246 sentences; worst id 239
- SentLen contamination: 147 sentences; id 150 published length 9.5
- Word streams diverge at row 2,594; 2,674 word mismatches in the overlap
- Full-SST last test batch: 162 / 1,186
- ZuCo stored valid mix: 7 / 14 / 19
- Standard scaler fingerprint: population std (`ddof=0`)
