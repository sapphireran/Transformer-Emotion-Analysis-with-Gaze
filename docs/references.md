# References and third-party data

Personal reading list for this repo. Not a formal bibliography. Links are to the public project pages; check current licenses before any redistribution beyond this private experiment.

## Corpora

### ZuCo — Zurich Cognitive Language Processing Corpus

Hollenstein, N., Rotsztejn, J., Troendle, M., Pedroni, A., Zhang, C., & Langer, N.  
*ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading.*  
Scientific Data, 2018.

- Task 1 (used here): sentiment / movie-review sentences, natural reading, 12 subjects.
- Also contains EEG, which this repo never loads.
- Official description: https://osf.io/q3zws/ (OSF; confirm before download)

MATLAB field names this code expects (`nFixations`, `FFD`, `GD`, `GPT`, `TRT`, `SFD`, `meanPupilSize`, `omissionRate`, `word.content`) come from that release’s `sentenceData` structs.

### Stanford Sentiment Treebank (SST)

Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., & Potts, C.  
*Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank.*  
EMNLP 2013.

The files in `SST_data/` are a **three-class** collapse (negative / neutral / positive), not the original 5-way fine-grained SST-5 leaf labels. Neutral is the smallest class in the combined table.

https://nlp.stanford.edu/sentiment/

### PROVO

Luke, S. G., & Christianson, K.  
*The Provo Corpus: A large eye-tracking corpus with predictability norms.*  
Behavior Research Methods, 2018.

`gaze_prediction/data/provo.csv` is a reduced column view (`nFix, FFD, GPT, TRT, fixProp`) for the gaze-prediction side experiment.

https://osf.io/sjefs/

## Encoders

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL 2019. Checkpoint: `bert-base-uncased`.
- Liu, Y. et al. *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* 2019. Checkpoint: `roberta-base`.

Both are loaded through Hugging Face `transformers`. Fine-tuning uses Adam at 5e-5, which is the usual BERT-base starting point, not a tuned recommendation for 400-row ZuCo.

## Eye-tracking measures

Standard first-pass / regression-path definitions (FFD, GD, GPT, TRT, SFD) are summarized in:

- Rayner, K. *Eye movements in reading and information processing: 20 years of research.* Psychological Bulletin, 1998.
- The ZuCo data descriptor (above), which documents how those fields were extracted from the SR Research recordings.

## Related personal context

This repository is a **personal** attempt to attach a 5-d gaze vector to a pooled transformer state for 3-way sentiment. It is not affiliated with the ZuCo authors, Stanford NLP, or any employer. Comments in the original Python files are a mix of English and Chinese because that is how the notes were written.

## Software

| Library | Used for |
| --- | --- |
| PyTorch | `EyeTrackingModel`, training loops |
| Hugging Face `transformers` / `datasets` | tokenizers, BERT/RoBERTa, `Dataset.map` |
| scikit-learn | `StratifiedKFold`, `train_test_split`, scalers, metrics |
| pandas / numpy | tables |
| scipy | `loadmat` in `utils_ZuCo.py` |
| nltk | SST word tokenization in `convert_sst_to_et.py` |
| tqdm | progress bars |

Example scripts intentionally avoid this stack so they run on a bare Python 3.

## Citation (if you mention this snapshot)

There is no paper attached to this git history (single “first commit” plus these notes). If you cite the *idea*, cite ZuCo + SST + the encoder papers, and describe this repo as unpublished personal code.
