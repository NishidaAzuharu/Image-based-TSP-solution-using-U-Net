# Image-based-TSP-solution-using-U-Net
This repository contains code for the paper "A solution of Traveling Salesman Problem Using Deep Learning" by Azuharu Fayyaz Nishida  
Accepted to The 2024 International Conference on Computational Science and Computational Intelligence (CSCI 2024), CSCI3034, (2024-12)  
Download datasets from [this link](https://www.kaggle.com/datasets/azuharunishida/huge-optimal-tour-tsp-datasets)

## run preprocessing
```bash
python preprocess.py --input_file {YOUR INPUT FILE PATH}
```

other options
|option|description|
|----|----|
|--num_data|Number of data to be used. Default is 87000|
|--test_split_ratio|Split ratio for test data. Default is 0.2|

## run train
```bash
python train.py
```