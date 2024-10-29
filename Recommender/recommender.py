import pandas as pd
import joblib

# Load the movie dataset and trained random forest model
movie_df = pd.read_csv('../df_ratings_cluster.csv')
ratings_cluster_df = movie_df[["Age","Gender","Occupation","MovieID","UserID","Rating", "cluster", "Title"]]
random_forest_model = joblib.load('../Research/random_forest_model.joblib')

# Function to gather user information
def get_user_info():
    gender = input("Enter your gender (female = 1, male = 2): ")
    print(
    "1: Under 18\n"
    "18: 18-24\n"
    "25: 25-34\n"
    "35: 35-44\n"
    "45: 45-49\n"
    "50: 50-55\n"
    "56: 56+\n"
    )
    age = int(input("Enter your age: "))
    print("Here is a list of occupations, please select a number from the list: \n")
    print(
    "0: other or not specified\n"
    "1: academic/educator\n"
    "2: artist\n"
    "3: clerical/admin\n"
    "4: college/grad student\n"
    "5: customer service\n"
    "6: doctor/health care\n"
    "7: executive/managerial\n"
    "8: farmer\n"
    "9: homemaker\n"
    "10: K-12 student\n"
    "11: lawyer\n"
    "12: programmer\n"
    "13: retired\n"
    "14: sales/marketing\n"
    "15: scientist\n"
    "16: self-employed\n"
    "17: technician/engineer\n"
    "18: tradesman/craftsman\n"
    "19: unemployed\n"
    "20: writer"
    )
    occupation = input("Enter your occupation from the above list: ")
    min_rating = input("Enter minimum movie rating between 1 - 5 (default is 4): ")
    min_rating = float(min_rating) if min_rating else 4.0
    return gender, age, occupation, min_rating

# Preprocess user information for model input
def preprocess_user_info(gender, age, occupation):
    # Example encoding (adjust according to your model's expected input format)
    gender_map = {'male': 1, 'female': 2}
    occupation_map = {
    0: "other or not specified",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer"
    }

    # Add placeholder values for the other fields
    movie_id = 0         # Placeholder for MovieID
    user_id = 0          # Placeholder for UserID

    gender_encoded = gender_map.get(gender.lower(), 2)  # Defaults to 'other' if unknown
    occupation_encoded = occupation_map.get(occupation.lower(), 3)
    return [age, gender_encoded, occupation_encoded, movie_id, user_id, min_rating] 

# Gather user information
gender, age, occupation, min_rating = get_user_info()
user_info = preprocess_user_info(gender, age, occupation)

# Predict the user's cluster/category using the random forest model
predicted_cluster = random_forest_model.predict([user_info])[0]

# Filter movies by the predicted cluster/category and minimum rating
filtered_movies = ratings_cluster_df[(ratings_cluster_df['cluster'] == predicted_cluster) & (ratings_cluster_df['Rating'] >= min_rating)]

# Sample movies and return a list
recommended_movies = filtered_movies.sample(n=10, replace=True)
movie_list = list(zip(recommended_movies['Title'], recommended_movies['Rating']))

print("Recommended movies with ratings:")
for title, rating in movie_list:
    print(f"{title} - Rating: {rating}")