# Sidecar geometry

Both trainers implement the same late-fusion idea: keep a pretrained
transformer frozen-or-finetuned as a text encoder, and bolt five reading-time
scalars onto the pooled representation with a tiny MLP that is not even an
MLP (there is no nonlinearity on the gaze branch).

## Forward pass as committed

```python
base_output = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
pooled_output = base_output.pooler_output          # (B, 768)
eye_tracking_output = self.eye_tracking_layer(eye_tracking_features)  # (B, 16)
combined_output = torch.cat((pooled_output, eye_tracking_output), dim=1)
combined_output = self.dropout(combined_output)   # p = 0.1
return self.classifier(combined_output)            # (B, 3)
```

`eye_tracking_layer` is `nn.Linear(5, 16)`. No ReLU, GELU, LayerNorm, or
gating. The sidecar is a linear basis expansion of five scalars, then 10%
dropout on the 784-d concat, then a 3-way classifier.

Gaze is **2.0%** of the concatenated vector (`16 / 784`). A randomly
initialized 784→3 head can ignore those 16 coordinates unless the training
signal is strong. The CSV-level correlations are not strong
([findings.md](findings.md)).

## Pooler, not token-wise gaze

The architecture never aligns word-level gaze with subword tokens. Even
though `ZuCo_et_csv_data/word/` exists, the trainers pass **sentence-level**
aggregates. There is no cross-attention between `input_ids` and a gaze
sequence. Claims of “gaze-informed tokens” do not match this code.

RoBERTa’s `pooler_output` is the `<s>` hidden state through a tanh dense
layer. BERT’s is the `[CLS]` pooler. Both trainers switch tokenizer and
`from_pretrained` name together via `model_type.startswith('bert')`.

## Why 16 dimensions is a lot

`examples/03_sidecar_rank.py` standardizes the five full-SST train columns
and takes an SVD:

| component | explained variance (full SST train) |
| --- | --- |
| PC1 | ≈ 0.917 |
| PC2 | ≈ 0.082 (loadings dominated by GD) |
| PC3–PC5 | < 0.001 combined |

nFix, TRT, FFD, and GPT correlate at r > 0.98. GD is the only column with a
distinct residual. A 16-d linear map of that table is a wide rotation of one
or two axes. If you ablate, shuffle, or PCA-reduce the sidecar to 2-d, you
are not throwing away a secret 5-d cognitive state — you are throwing away
near-duplicates.

ZuCo’s five *model* columns (`nFixations, FFD, GPT, TRT, GD`) are less
collapsed than the predicted full-SST ones, but TRT/nFixations/GPT still
sit at r ≈ 0.94–0.96. The unused columns (`meanPupilSize`, `omissionRate`,
`SFD`) are the more independent physiological channels sitting in the CSV
unused.

## NumPy stand-in

`examples/sidecar/fusion.py` rebuilds the concat with NumPy so the atlas can
unit-test shapes without `torch` or the Hub:

```
gaze  (B, 5)  --Linear 5→16-->  eye (B, 16)
pooler(B, 768) --------------\
                              concat (B, 784) → Dropout → Linear → logits (B, 3)
```

`examples/08_fusion_forward.py` runs that on five real train gaze rows plus
fake N(0, 0.02) poolers. The predictions are meaningless; the shapes are
the point.

## Design choices the 2024 scripts did not take

These are notes, not a rewrite:

- **No activation on gaze.** A ReLU after `Linear(5, 16)` would at least make
  the sidecar a one-layer MLP.
- **No feature selection.** Feeding PC1+PC2 (or GD + nFix) would match the
  empirical rank.
- **No token alignment.** Word-level TRT/FFD could be pooled with attention
  weights instead of a sentence mean.
- **No gated fusion.** Concat + linear lets the 768-d side drown the 16-d
  side. A FiLM/gate on the pooler using the gaze vector would force an
  interaction.
- **Dropout on the concat** also regularizes the 768-d linguistic features,
  not just the sidecar.
