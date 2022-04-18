# CARCA

This is our implementation for the 2022 CARCA paper https://arxiv.org/abs/2204.06519:

Rashed, Ahmed, et al. "CARCA: Context and Attribute-Aware Next-Item Recommendation via Cross-Attention"


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
