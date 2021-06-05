import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans

# Create a dictionary of ratings
ratings = {
    "item": ['book 1', 'book 2', 'book 1', 'book 2', 'book 1', 'book 2', 'book 1'],
    "user": ['Dries', 'Dries', 'Alistair', 'Alistair', 'Harrison', 'Harrison', 'You'],
    "rating": [2.5, 4, 3, 4, 2.5, 4, 2],
}

# Create a pandas dataframe
df = pd.DataFrame(ratings)
# Create our reader and set the scale
reader = Reader(rating_scale=(1, 5))

# Load the dataset from the dataframe
data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
# Build the set
trainingSet = data.build_full_trainset()

# Initiate our algorithm
algo = KNNWithMeans()
# Fit the algorithm to our set
algo.fit(trainingSet)

# Create a prediction!
prediction = algo.predict('You', 'book 2')