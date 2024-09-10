import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load ratings data
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=column_names)

# Load movie titles data
movie_titles = pd.read_csv('ml-100k/u.item', sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['item_id', 'title'])

# Merge ratings and movie titles
merged_df = pd.merge(ratings, movie_titles, on='item_id')

# Explore basic stats
print("Basic statistics of ratings:")
print(merged_df.describe())

# Visualize rating distribution
sns.histplot(merged_df['rating'], bins=5, kde=False)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()
