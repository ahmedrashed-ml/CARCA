 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carca-context-and-attribute-aware-next-item/sequential-recommendation-on-amazon-men)](https://paperswithcode.com/sota/sequential-recommendation-on-amazon-men?p=carca-context-and-attribute-aware-next-item)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carca-context-and-attribute-aware-next-item/recommendation-systems-on-amazon-games)](https://paperswithcode.com/sota/recommendation-systems-on-amazon-games?p=carca-context-and-attribute-aware-next-item)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carca-context-and-attribute-aware-next-item/recommendation-systems-on-amazon-fashion)](https://paperswithcode.com/sota/recommendation-systems-on-amazon-fashion?p=carca-context-and-attribute-aware-next-item)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/carca-context-and-attribute-aware-next-item/recommendation-systems-on-amazon-beauty)](https://paperswithcode.com/sota/recommendation-systems-on-amazon-beauty?p=carca-context-and-attribute-aware-next-item)

# CARCA

This is our implementation for the CARCA paper accepted at RecSys 2022 
https://dl.acm.org/doi/10.1145/3523227.3546777:

Rashed, Ahmed, et al. "Context and Attribute-Aware Sequential Recommendation via Cross-Attention"

Please cite our paper if you use the code or datasets.

## Enviroment 
	* pandas==1.0.3
	* tensorflow==1.14.0
	* matplotlib==3.1.3
	* numpy==1.18.1
	* six==1.14.0
	* scikit_learn==0.23.1
	
## Steps
1) Download preprocessed data from here "https://drive.google.com/drive/folders/1a_u52mIEUA-1WrwsNZZa-aoGJcMmVugs?usp=sharing" or the raw data from "https://jmcauley.ucsd.edu/data/amazon/"

2) Add the data files inside the "Data/" folder

3) To run the respective dataset, please use the below commands
- python CARCA.py 'Video_Games'
- python CARCA.py 'Men'
- python CARCA.py 'Beauty'
- python CARCA.py 'Fashion'

4) To preprocess raw Amazon reviews data, please use the DataProcessing.py and put the reviews and metadata in the RawData folder. Also, generate the context dictionaries using the commented section in the CARCA.py

5) To preprocess the Men and Fashion image features from scratch you will need to download all products images and pass them through a pre-trained resnet 50 model. Then match them using their ASIN code with the reviews data.



## Important Note
If you are planning to apply CARCA on datasets without attributes or context, it is advisable to use rolling window protocol for training the model as the current training protocol (right shifted input) might not be stable in those scenarios.
