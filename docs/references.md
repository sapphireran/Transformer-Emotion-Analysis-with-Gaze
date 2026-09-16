# References

Cite the **corpora** if you publish with these CSVs. This repository is personal research code, not an official ZuCo or SST release.

## Corpora

**ZuCo (Zurich Cognitive Language Processing Corpus)**  
Hollenstein, N., Rotsztejn, J., Troendle, M., Pedroni, A., Zhang, C., & Langer, N. (2018). *ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading.* Scientific Data, 5, 180291.  
<https://doi.org/10.1038/sdata.2018.291>  
Data: <https://osf.io/q3zws/>

Task 1 in ZuCo is normal reading of 400 Stanford Sentiment Treebank movie-review sentences — the measured-gaze track in this repo. ZuCo 2.0 exists; these CSVs follow the original 12-subject Task 1 dump.

**Stanford Sentiment Treebank (SST)**  
Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., & Potts, C. (2013). *Recursive deep models for semantic compositionality over a sentiment treebank.* EMNLP.  
<https://nlp.stanford.edu/sentiment/>

This project uses **sentence-level 3-class** labels (negative / neutral / positive), not the 5-class fine-grained SST-5 set and not phrase-level trees.

**Provo Corpus**  
Luke, S. G., & Christianson, K. (2018). *The Provo Corpus: A large eye-tracking corpus with predictability norms.* Behavior Research Methods, 50, 582–597.  
<https://doi.org/10.3758/s13428-017-0908-4>

`gaze_prediction/data/provo.csv` is a word-level extract (`nFix, FFD, GPT, TRT, fixProp`) used as a distributional reference for predicted gaze, not as training text for sentiment.

## Models

**BERT**  
Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of deep bidirectional transformers for language understanding.* NAACL.  
Checkpoint used: `bert-base-uncased` (Hugging Face).

**RoBERTa**  
Liu, Y., et al. (2019). *RoBERTa: A robustly optimized BERT pretraining approach.* arXiv:1907.11692.  
Checkpoint used: `roberta-base`.

## Eye-tracking measures

Rayner, K. (1998). *Eye movements in reading and information processing: 20 years of research.* Psychological Bulletin, 124(3), 372–422.

Standard names used in the CSVs (FFD, GD, SFD, GPT / regression-path, TRT) are defined for this repo in [gaze-features.md](gaze-features.md).

## Related lines of work (context, not dependencies)

Using cognitive signals with NLP classifiers is an active area (EEG, eye-tracking, fMRI). This codebase is a **concat fusion** baseline: `Linear(5, 16)` on sentence gaze + transformer `pooler_output`. It does not implement token-level gaze injection, MAG, or adapter methods from later papers.

## Software

Training: PyTorch, Hugging Face `transformers` + `datasets`, scikit-learn, pandas, tqdm.  
MATLAB I/O: SciPy `loadmat`.  
SST word placeholders: NLTK `word_tokenize`.  
Docs examples: pandas, NumPy, matplotlib, scikit-learn, pytest (see `requirements-examples.txt`).
