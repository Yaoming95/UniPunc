# UniPunc


## Code Release
We release [UniPunc](https://ieeexplore.ieee.org/document/9747131) model code under `fairseq_code` folder.

The code is implemented based on [fairseq](https://github.com/facebookresearch/fairseq). 


## Data

You can download MUST-C data [here](https://ict.fbk.eu/must-c/), where we use release v1.0.  
We also use [mTEDx](https://www.openslr.org/100/) for construct English-mixed subset.

The data split is in `data/` folder. 

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


## Citation
https://ieeexplore.ieee.org/document/9747131
```LaTeX
@INPROCEEDINGS{9747131,
  author={Zhu, Yaoming and Wu, Liwei and Cheng, Shanbo and Wang, Mingxuan},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Unified Multimodal Punctuation Restoration Framework for Mixed-Modality Corpus}, 
  year={2022},
  volume={},
  number={},
  pages={7272-7276},
  doi={10.1109/ICASSP43922.2022.9747131}}
```