import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load ratings data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=column_names)

# Prepare data for Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Build SVD collaborative filtering model
model = SVD()
model.fit(trainset)

# Predict ratings for testset
predictions = model.test(testset)

# Evaluate the model using RMSE
rmse_value = accuracy.rmse(predictions)
print(f'Root Mean Squared Error: {rmse_value}')
