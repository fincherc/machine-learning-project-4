# Movie Recommendation using Maching Learning

## Overview 
This repository contains a project aimed at building a machine learning model that will allow us to provide a user with a set of recommended movies based on the user's inputs, such as gender, age, occupation, etc. The algorithm used is Random Forest.

## Table of Contents 

- [Overview](#overview)
- [Project Description](#project-description)
- [Key Aspects to Explore](#key-aspects-to-explore)
- [Databases to be Used](#databases-to-be-used)
- [Breakdown of Tasks](#breakdown-of-tasks)
- [Data Standardization and Analysis](#data-standardization-and-analysis)
- [Building the Model](#building-the-model)
- [Recommender](#recommender)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [User Interactivity](#user-interactivity)
- [License](#license)
- [References](#references)


## Project Description

This project focuses on building a personalized movie recommendation system using collaborative filtering techniques with the k-Nearest Neighbors (k-NN) algorithm. The system leverages user rating data to identify patterns and similarities between users. By analyzing these interactions, the model can recommend movies that are likely to align with a user's preferences. The goal is to develop a scalable and efficient recommendation engine that provides relevant and meaningful suggestions. This project utilizes the MovieLens dataset, which offers extensive user-movie interaction data, allowing us to evaluate the performance of our model with real-world data.

## Key Aspects to Explore

1. **Collaborative Filtering Techniques:**

     -  Explore user-based and item-based collaborative filtering approaches to find the best fit for the project.
    
2. **k-NN Algorithm:**

     - Explore the performance of the k-NN algorithm in recommending movies based on historical user ratings.
     - Experiment with different values of k (number of neighbors) to find the optimal number of similar movies for accurate recommendations.

3. **Evaluation Metrics:**
     - Evaluate the system using metrics such as precision, recall, mean absolute error (MAE), to measure the accuracy and relevance of the recommendations.


## Databases to be Used 
* [Movie Lens dataset:](https://grouplens.org/datasets/movielens/1m/) 
     - Contains 1,000,000 ratings from 6,000 users on 4,000 movies.
     - Includes user IDs, movie IDs, ratings (from 1 to 5), and optional movie metadata (e.g., genres, titles).


## Breakdown of Tasks

* **Steve Yuan:** Data Collection, preprocessing and encoding.
* **Santiago Cardenas:** Encoding, clean up, create a SQLite databse and README file.
* **Christian Fincher:** Data analysis, build the machine learning model.
* **Nicole Navarijo:** Build the model and improve it, create a PPT presentation.

## Data Standardization, Analysis and Model

* **Data Standardization:** The process of converting the data from a `.dat` file to a `.csv` format was performed by first reading the .dat file using Python's pandas library. The appropriate delimiter (such as :: or :) was specified to properly separate the columns. The data was then cleaned by dropping unnecessary columns or handling missing values, if necessary. After assigning meaningful column names, the data was saved as a .csv file using the `to_csv()` function, ensuring the file was stored in the desired location with proper formatting.
* **Encoding:** In order to have a meaninful dataset to train our model, we decide to encode some of the features.
- Gender is denoted by a "M" for male and "F" for female
      * 1: Female
      * 2: Male
- Age is chosen from the following ranges:

	*  1:  "Under 18"
	* 18:  "18-24"
	* 25:  "25-34"
	* 35:  "35-44"
	* 45:  "45-49"
	* 50:  "50-55"
	* 56:  "56+"

- Occupation is chosen from the following choices:

	*  0:  "other" or not specified
	*  1:  "academic/educator"
	*  2:  "artist"
	*  3:  "clerical/admin"
	*  4:  "college/grad student"
	*  5:  "customer service"
	*  6:  "doctor/health care"
	*  7:  "executive/managerial"
	*  8:  "farmer"
	*  9:  "homemaker"
	* 10:  "K-12 student"
	* 11:  "lawyer"
	* 12:  "programmer"
	* 13:  "retired"
	* 14:  "sales/marketing"
	* 15:  "scientist"
	* 16:  "self-employed"
	* 17:  "technician/engineer"
	* 18:  "tradesman/craftsman"
	* 19:  "unemployed"
	* 20:  "writer"

- Converted genres into numbers, created 18 new columns,  like 'genres_action',  0 means false, 1 means true.


* **Database:** The SQLite database was created by first loading the cleaned .csv files. After reading the data into DataFrames, a connection to an SQLite database was established using the sqlite3 library. The data from each DataFrame was then written to the database as tables. Finally, a SQL query was created to combine these tables into one.

## Building the Model

* **Feature and Target Selection:**
  - Features (X): Includes user demographics `(Age, Gender, Occupation)`, movie IDs, and ratings.
  - Target (y): The model is trained to predict the cluster of each interaction, likely grouping users or movies into clusters based on behavior or similarity.

* **Splitting the Data into Train and Test Sets:**
  - Train_test_split ensures that the model is trained on one part of the data and tested on unseen data to evaluate generalization.

* **Training the Random Forest Model:**
  - n_estimators=500: Specifies the number of decision trees in the forest.
  - The fit method trains the model using the training data (`X_train` and `y_train`).

* **Making Predictions:**
  - Predictions were made on both the test set and training set

* **Training Set Accuracy:**
  - Test Set Accuracy: The accuracy on the test set was 0.61(approximately).
  - Training Set Accuracy: The accuracy on the training set was 0.81(approximately). 

## Recommender

**How to Use the Model**

    1. Loading Data and Model:

      - The script loads movie data and user demographics from the dataset.
      - It also loads a pre-trained Random Forest model from the joblib file.

    2. Collecting User Input:

      - The user is prompted to enter demographic information such as:
        - Gender (1 for female, 2 for male)
        - Age group (e.g., under 18, 18-24, etc.)
        - Occupation (from a predefined set of occupation codes)

    3. Making Predictions:

      - The model takes the user’s demographic inputs and predicts the most relevant movie recommendations.
      - The predicted recommendations are tailored to match the user’s profile and past behaviors based on clustering.

    4. Displaying Results:

      - Once the predictions are made, the system displays recommended movies along with relevant information like movie titles.


## Repository Structure
```
machine-learning-project-4/
│
├── code/
│ ├── started_code.ipynb
|    ├── Resources
|        ├── movies.csv
|        ├── ratings.csv
│        └── users.csv
│ 
├── DataSets/
│ ├── Movies_data_V3.csv
│ ├── ratings_clean.csv
│ └── users_clean_V2.csv
|
├── Recommender/
│ └── recommender.py
|
├── Research/
│ └── random_forest.ipynb
│
├── Training/
│ ├── Resources
│     ├── combined_movies_dataset.sqlite
│     └── movie_traning.ipynb
|
├── df_ratings_clusters.csv/
|
├── Get_movie_missing_title.ipynb/
├── KNN_solution.ipynb/
|
├── output_no_userid_2_100.png/
├── output_no_userid_100_200.png/
|
└── README.md
```

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/fincherc/machine-learning-project-4.git
    ```
2. Install the required packages:
    ```bash
    pip install pandas
    sqlalchemy
    numpy
    matplotlib
    plotly
    hvplot.pandas
    ssklearn.cluster 
    sklearn.metrics 
    sklearn.preprocessing
    jupyter
    ```
3. Install Jupyter Notebook and Visual Studio Code:
    - Install [Visual Studio Code](https://code.visualstudio.com/).
    - Install the [Python extension for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=ms-python.python).
    - Install Jupyter Notebook:
      ```bash
      pip install notebook
      ```
4. Open the project in Visual Studio Code and start Jupyter Notebook:
    - Open Visual Studio Code.
    - Open the project folder.
    - Open a new terminal in Visual Studio Code and run:
      ```bash
      jupyter notebook
      ```
    - This will open Jupyter Notebook in your default web browser.



## License
This project is licensed under the MIT License.

## References

1. Transformed '.dat' format dataset to csv files. Stackoverflow.
      https://stackoverflow.com/questions/11483920/convert-dat-files-to-either-dta-shp-bdf-or-csv
2. Understanding of K Means Clustering on High Dimensional Data. Medium.com
      https://medium.com/swlh/k-means-clustering-on-high-dimensional-data-d2151e1a4240
3. How to deal with a dataset that contains multiples features. Analytics Vidhya.
      https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/
4. Visualize our features in a three dimentinal way. Project TensorFlow.
      https://projector.tensorflow.org/? 
      _gl=1*pybufr*_ga*OTQ0ODY2NzUxLjE3MjczODkwOTk.*_ga_W0YLR4190T*MTcyODY1OTA4Ny45LjAuMTcyODY1OTA4Ny4wLjAuMA
5. How to use calinski_harabasz_score. scikit-learn.org 
      https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html
6. Under "Data Standardization, Analysis and Model/Encoding" part of the README, "Gender, age, and recommendation" were taken from the Movie Lenses' README file.
       https://grouplens.org/datasets/movielens/1m/
7. Logistic Regression documentation used to build our classification report. Sckit learn.
       https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
8. Random oversample was used to improve our accuracy. Sckit learn.
       https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html#imblearn.over_sampling.RandomOverSampler
