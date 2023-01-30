<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a>
    <img src="readme-assets/DALL·E-logo.png" alt="Logo" width="224" height="224">
  </a>

  <h1 align="center"><img src="readme-assets/magnifier-zoom-search-lineal.gif" width="35px"> Reverse Image Search Engine</h1>
</div>

<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/himalayasharma/small-image-search-engine?style=social"> <img alt="GitHub forks" src="https://img.shields.io/github/forks/himalayasharma/small-image-search-engine?style=social"> <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/himalayasharma/small-image-search-engine"> <img alt="GitHub issues" src="https://img.shields.io/github/issues-raw/himalayasharma/small-image-search-engine">

I've tried to create a clone of [Google's Reverse Image Search Engine](https://www.google.com/imghp?hl=en).

The project aimed to build a content-based image retrieval system using a [VGG-16](https://keras.io/api/applications/vgg/) deep learning model and the [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset. The model was initialized with ImageNet weights and trained for multi-class classification. The performance of the model was evaluated and an accuracy of 89% was achieved on the validation set and 90% on the test set. The network front-end was then utilized for feature extraction and generated 60k image encodings. These encodings were used to compute similarity scores against the query image, resulting in the top 5 matches being retrieved. The project aimed to demonstrate the effectiveness of using deep learning models in content-based image retrieval.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project
    ├── query-images       <- Contains query images against which to find matches
    ├── readme-assets      <- Contains images to be used in README.md
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling
    │   └── raw            <- The original, immutable data dump
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    └── src                <- Source code for use in this project
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
3. Enter name of the image (with extension) and press enter.

![alt text](https://github.com/himalayasharma/small-image-search-engine/blob/master/readme-assets/enter-query-image.png)

Outputs
------------
Query images used here are not from the CIFAR-10 dataset.

Example 1:

![alt text](https://github.com/himalayasharma/small-image-search-engine/blob/master/readme-assets/output-dog.png)

Example 2:

![alt text](https://github.com/himalayasharma/small-image-search-engine/blob/master/readme-assets/output-frog.png)

Example 3:

![alt text](https://github.com/himalayasharma/small-image-search-engine/blob/master/readme-assets/output-horse.png)

Contributing
------------
Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated. If you have a suggestion that would make this better, please fork the repo and create a pull request. Don't forget to give the project a star! Thanks again!

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

License
------------
Distributed under the MIT License. See `LICENSE.txt` for more information.

Ackowledgements
------------
* [Aleksa Gordić](https://github.com/gordicaleksa)
* [drivendata](https://github.com/drivendata)
* [James.Scott](https://github.com/scottydocs)
* [Othneil Drew](https://github.com/othneildrew)
--------
