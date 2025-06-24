import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import shutil
import pickle, os
class RecommenderSystem:
    def __init__(self, ratings_path="ml-100k/u.data", movies_path="ml-100k/u.item"):
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self._load_data()

    def _load_data(self):
        self.ratings_df = pd.read_csv(
            self.ratings_path,
            header=0,  
            encoding='latin-1'
        )

        self.ratings_df['rating'] = pd.to_numeric(self.ratings_df['rating'], errors='coerce')
        self.ratings_df.dropna(subset=['rating'], inplace=True)

        self.movies_df = pd.read_csv(
            self.movies_path,
            header=0,
            encoding='latin-1'
        )

        self.data = pd.merge(self.ratings_df, self.movies_df, on='item_id')

        self.user_movie_matrix = self.data.pivot_table(index='user_id', columns='title', values='rating')
        self.user_movie_matrix.fillna(0, inplace=True)

    #Cosine Similarity Recommender
    def get_similar_movies(self, movie_name, n=5):
        if movie_name not in self.user_movie_matrix.columns:
            return ["Movie not found in dataset."]
        if os.path.exists("cosine_matrix.pkl"):
            with open("cosine_matrix.pkl", "rb") as f:
                similarity = pickle.load(f)
        else:
            similarity = cosine_similarity(self.user_movie_matrix.T)
            with open("cosine_matrix.pkl", "wb") as f:
                pickle.dump(similarity, f)
        self.similarity_df = pd.DataFrame(similarity, index=self.user_movie_matrix.columns, columns=self.user_movie_matrix.columns)
        similar_scores = self.similarity_df[movie_name].sort_values(ascending=False)[1:n+1]
        return list(similar_scores.index)
    
    def recommend_cosine_personal(self, user_ratings: dict, n=5):
        similarity = cosine_similarity(self.user_movie_matrix.T)
        similarity_df = self.similarity_df
        if type(list(user_ratings.keys())[0]) is int:
            user_ratings = {self.movies_df[self.movies_df['item_id'] == mid]['title'].values[0]: rating for mid, rating in user_ratings.items()}
        all_movies = set(self.user_movie_matrix.columns)
        rated_movies = set(user_ratings.keys())
        unrated_movies = all_movies - rated_movies

        scores = {}

        for movie in unrated_movies:
            numerator = 0
            denominator = 0
            for rated_movie, rating in user_ratings.items():
                if rated_movie in similarity_df.columns and movie in similarity_df.columns:
                    sim = similarity_df.at[movie, rated_movie]
                    numerator += sim * rating
                    denominator += abs(sim)
            if denominator != 0:
                scores[movie] = numerator / denominator

        top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [title for title, _ in top_movies]

    def load_similarity_matrix(self):
        if os.path.exists("cosine_matrix.pkl"):
            with open("cosine_matrix.pkl", "rb") as f:
                similarity = pickle.load(f)
        else:
            similarity = cosine_similarity(self.user_movie_matrix.T)
            with open("cosine_matrix.pkl", "wb") as f:
                pickle.dump(similarity, f)
        print("Cosine similarity matrix loaded or computed.")
        self.similarity_df = pd.DataFrame(similarity, index=self.user_movie_matrix.columns, columns=self.user_movie_matrix.columns)

    #Matrix Factorization (SVD)
    def train_svd_model(self):
        reader = Reader(line_format='user item rating timestamp', sep=',')
        with open(self.ratings_path, 'r') as original, tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
            next(original) # Skip header
            shutil.copyfileobj(original, temp)
            temp_path = temp.name

        data = Dataset.load_from_file(temp_path, reader=reader)
        self.trainset = data.build_full_trainset()
        self.svd_model = SVD()
        self.svd_model.fit(self.trainset)
        import pickle

        with open("svd_model.pkl", "wb") as f:
            pickle.dump(self.svd_model, f)
    
    def recommend_svd_for_custom_ratings(self, user_ratings_dict, n=5):
        if not hasattr(self, 'svd_model'):
            self.train_svd_model()

        # Step 1: Create new user ID
        temp_user_id = self.data['user_id'].max() + 1

        # Step 2: Create temporary DataFrame with custom ratings
        temp_ratings = pd.DataFrame([
            {'user_id': temp_user_id, 'item_id': mid, 'rating': rating}
            for mid, rating in user_ratings_dict.items()
        ])

        # Step 3: Combine with original data
        combined_data = pd.concat([self.data[['user_id', 'item_id', 'rating']], temp_ratings])

        # Step 4: Prepare Surprise dataset again
        reader = Reader(rating_scale=(0.5, 5))
        surprise_data = Dataset.load_from_df(combined_data[['user_id', 'item_id', 'rating']], reader)
        trainset = surprise_data.build_full_trainset()

        # Step 5: Train new temporary model (can reuse global config if needed)
        temp_model = SVD()
        temp_model.fit(trainset)

        # Step 6: Exclude rated movies
        rated_item_ids = set(user_ratings_dict.keys())
        all_item_ids = set(self.movies_df['item_id'].unique())
        unseen_item_ids = list(all_item_ids - rated_item_ids)

        # Step 7: Predict for unseen movies
        predictions = []
        for mid in unseen_item_ids:
            try:
                pred = temp_model.predict(str(temp_user_id), str(mid)).est
                predictions.append((mid, pred))
            except:
                continue

        predictions.sort(key=lambda x: x[1], reverse=True)
        top_item_ids = [mid for mid, _ in predictions[:n]]

        # Step 8: Map to titles
        id_to_title = dict(zip(self.movies_df['item_id'], self.movies_df['title']))
        recommended_titles = [id_to_title[mid] for mid in top_item_ids if mid in id_to_title]

        return recommended_titles


    def load_or_train_svd(self):
        if os.path.exists("svd_model.pkl"):
            with open("svd_model.pkl", "rb") as f:
                self.svd_model = pickle.load(f)
                self.trainset = self.svd_model.trainset
            print("Loaded saved SVD model.")
        else:
            print("Training new SVD model...")
            self.train_svd_model()

    def recommend_svd(self, user_id, n=5):
        if not hasattr(self, 'svd_model'):
            self.train_svd_model()

        user_movies = self.data[self.data['user_id'] == int(user_id)]['title'].tolist()
        all_movies = self.movies_df['title'].tolist()
        unseen_movies = list(set(all_movies) - set(user_movies))

        predictions = [(movie, self.svd_model.predict(str(user_id), str(self._get_item_id(movie))).est)
                       for movie in unseen_movies if self._get_item_id(movie) is not None]

        predictions.sort(key=lambda x: x[1], reverse=True)
        return [title for title, _ in predictions[:n]]

    def _get_item_id(self, movie_title):
        match = self.movies_df[self.movies_df['title'] == movie_title]
        if not match.empty:
            return match.iloc[0]['item_id']
        return None

    #Evaluation (SVD)
    def evaluate_svd(self):
        reader = Reader(line_format='user item rating timestamp', sep=',')
        data = Dataset.load_from_file(self.ratings_path, reader=reader)
        model = SVD()
        results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        return results

    #Data Visualization Helpers
    def plot_top_rated_movies(self, top_n=10):
        top_movies = self.data.groupby('title')['rating'].mean().sort_values(ascending=False).head(top_n)
        sns.barplot(x=top_movies.values, y=top_movies.index)
        plt.title("Top Rated Movies")
        plt.xlabel("Average Rating")
        plt.show()

    def plot_most_rated_movies(self, top_n=10):
        most_rated = self.data['title'].value_counts().head(top_n)
        sns.barplot(x=most_rated.values, y=most_rated.index)
        plt.title("Most Rated Movies")
        plt.xlabel("Number of Ratings")
        plt.show()
        
rs = RecommenderSystem(ratings_path="data/ml-latest-small/ratings.csv", movies_path="data/ml-latest-small/movies.csv")
rs.load_or_train_svd()
rs.load_similarity_matrix()
#Example usage
if __name__ == "__main__":
    print("Cosine Similarity:", rs.recommend_cosine_personal({"Sabrina (1995)": 5}))
    #rs.train_svd_model()
    custom_user_ratings = {
        1: 5.0,     
        276: 4.0,   
        1022: 3.5   # item_id : rating
    }
    print("SVD Recommendations:", rs.recommend_svd_for_custom_ratings(custom_user_ratings, n=5))
