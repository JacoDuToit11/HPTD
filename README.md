# Source code for paper: [Prompt Tuning Discriminative Language Models for Hierarchical Text Classification](https://doi.org/10.1017/nlp.2024.51)

## Requirements
    $pip install -r requirements.txt

## Datasets
### Web Of Science
    Acquire dataset: https://github.com/kk7nc/HDLTex
    Save Data.xlsx in 'WOS/Meta-data' to 'Data.txt'
    $cd Data/WOS
    $python preprocess_wos.py
    $python data_wos.py

### RCV1-V2
    Acquire dataset: https://github.com/ductri/reuters_loader
    $cd Data/RCV1
    $python preprocess_rcv1.py
    $python data_rcv1.py

### NYT
    Acquire dataset: https://catalog.ldc.upenn.edu/LDC2008T19
    Unzip 'nyt_corpus_LDC2008T19.tgz' into Data/NYT
    $cd Data/NYT
    $python data_nyt.py

## Train and Test
    $python TrainHDPT.py
