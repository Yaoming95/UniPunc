# UniPunc


## Code Release
We release UniPunc model code under `fairseq_code` folder.

The code is implemented based on [fairseq](https://github.com/facebookresearch/fairseq). 



## Case Study and Multilingual Performance

The case study instance and multilingual performance of our ICASSP submission. We provide those results for reviewers' convenience.

This repo does not contain an analysis, which is provided in section 5 of the paper.

We will release the code upon paper acceptance.

### Case Study Instances
Please refer to case.tsv

### Multilingual Performance 

We also compare UniPunc and other baseline on multilingual sentences from mTEDx, where we select 6 languages, namely English, German, French, Spanish, Portuguese, Italian.

|             | Overall | Comma | Full Stop | Question Mark |
|-------------|---------|-------|-----------|---------------|
| Att-GRU     | 54.8    | 48.2  | 65.6      | 32.1          |
| BiLSTM      | 53.6    | 46    | 65.2      | 30.1          |
| BERT        | 74.7    | 71.5  | 80.1      | 61.1          |
| UniPunc-Mix | 75.4    | 72.1  | 80.8      | 71.3          |


