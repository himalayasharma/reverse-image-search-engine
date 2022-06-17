Small Image Search Engine
==============================
![GitHub Repo stars](https://img.shields.io/github/stars/himalayasharma/small-image-search-engine?style=social) ![GitHub forks](https://img.shields.io/github/forks/himalayasharma/small-image-search-engine?style=social) ![GitHub pull requests](https://img.shields.io/github/issues-pr/himalayasharma/small-image-search-engine)   ![GitHub  issues](https://img.shields.io/github/issues-raw/himalayasharma/small-image-search-engine)  ![GitHub  all  releases](https://img.shields.io/github/downloads/himalayasharma/small-image-search-engine/total)

Gives top 5 matches (from the CIFAR-10 dataset) for given query image. 

A VGG-16 model is used to generate encodings of images in the CIFAR-10 dataset. Top 5 images with highest similarity to the query image are returned back.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── train_model.py
        │   ├── generate_encodings.py
        │   └── predict_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py
    
--------
Prerequisites
------------
Before you begin, ensure you have met the following requirements:
* You have a `Linux/Mac/Windows` machine.
* You have installed a `python` distribution. For NVIDIA GPU support, `conda` is preferred as `cudatoolkit` and `cudnn` packages hosted on `conda-forge` channel can be installed easily.
* You have installed `pip`.
* You have installed `make`.

Setup
------------
1. Clone the repo
	```
	git clone https://github.com/himalayasharma/small-image-search-engine.git
	```
2. Create virtual environment.
	```make
	make create_environment
	```
3. Activate virtual environment.
4. Download and install all required packages.
	```make
	make requirements
	```
5. Download and process CIFAR-10 dataset.
	```make
	make data
	```
6. Train model on CIFAR-10 dataset.
	```make
	make train
	```
7. Generate encodings for all images in the dataset.
	```make
	make encodings
	```
    
Usage
------------
After the model is trained it can be used for inference. 

1. Put a query image in `query-images` directory. 
2. Run the following command:
```make
make predict
```
3. Enter name of the image (with extension).
![alt text](https://github.com/himalayasharma/small-image-search-engine/blob/master/readme-assets/enter-query-image.png)
![alt text](https://github.com/himalayasharma/small-image-search-engine/blob/master/readme-assets/output-dog.png)



