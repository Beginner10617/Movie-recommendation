import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import os
import time


TEMPLATE_PLOT_DIR = Path("templates/plots")
TEMPLATE_PLOT_DIR.mkdir(parents=True, exist_ok=True)

def skip_if_recently_modified(path, minutes=5):
    if os.path.exists(path):
        modified_time = os.path.getmtime(path)
        current_time = time.time()
        if current_time - modified_time < minutes * 60:
            print(f"[SKIPPED] {path} modified less than {minutes} minutes ago.")
            return True
    return False

def interactive_rating_histogram(df, save_path="templates/plots/ratings_histogram.html"):
    if skip_if_recently_modified(save_path):
        return
    df['rating'] = df['rating'].astype(float)
    fig = px.histogram(df, x='rating', nbins=10, title="Distribution of Ratings")
    fig.write_html(save_path, full_html=False, include_plotlyjs='cdn')

def interactive_avg_ratings(df, save_path="templates/plots/avg_ratings.html", top_n=None):
    if skip_if_recently_modified(save_path):
        return

    df['rating'] = df['rating'].astype(float)
    avg_ratings = df.groupby('title')['rating'].mean().sort_values(ascending=False)
    if top_n:
        avg_ratings = avg_ratings.head(top_n)

    fig = px.line(
        x=avg_ratings.index,
        y=avg_ratings.values,
        title="Average Rating per Movie",
        labels={'x': 'Movie Title', 'y': 'Average Rating'}
    )
    fig.update_layout(
        xaxis_tickangle=-60,
        xaxis_title='Movie Title',
        yaxis_title='Average Rating'
    )
    fig.write_html(save_path, full_html=False, include_plotlyjs='cdn')

def interactive_num_ratings(df, save_path="templates/plots/num_ratings.html", top_n=None):
    if skip_if_recently_modified(save_path):
        return

    rating_counts = df['title'].value_counts()
    if top_n:
        rating_counts = rating_counts.head(top_n)

    fig = px.line(
        x=rating_counts.index,
        y=rating_counts.values,
        title="Number of Ratings per Movie",
        labels={'x': 'Movie Title', 'y': 'Number of Ratings'}
    )
    fig.update_layout(
        xaxis_tickangle=-60,
        xaxis_title='Movie Title',
        yaxis_title='Number of Ratings'
    )
    fig.write_html(save_path, full_html=False, include_plotlyjs='cdn')

def interactive_cosine_heatmap(sim_matrix, save_path="templates/plots/cosine_heatmap.html", top_n=None, shift=(0,0)):
    print(f"Generating cosine heatmap with top_n={top_n} and shift={shift}")
    
    if top_n:
        shiftX, shiftY = shift
        sim_matrix = sim_matrix.iloc[shiftX:shiftX+top_n, shiftY:shiftY+top_n]
    movie_titles_x = sim_matrix.columns.tolist()
    movie_titles_y = sim_matrix.index.tolist()
    print(f"Movie titles for x-axis: {movie_titles_x}")
    print(f"Movie titles for y-axis: {movie_titles_y}")
    fig = px.imshow(
        sim_matrix.values,
        labels=dict(x="Movie Title", y="Movie Title", color="Similarity"),
        x=movie_titles_x,
        y=movie_titles_y,
        title="Cosine Similarity Heatmap (Shifted View)" if top_n else "Cosine Similarity Heatmap (Full)",
        aspect="auto"
    )
    fig.update_xaxes(side="top")
    fig.write_html(save_path, include_plotlyjs='cdn')

def interactive_svd_embeddings(svd_model, trainset, movie_df, save_path="templates/plots/svd_embeddings.html", top_n=None):
    if skip_if_recently_modified(save_path):
        return
    movie_inner_ids = list(trainset.all_items())
    if top_n:
        movie_inner_ids = movie_inner_ids[:top_n]

    movie_factors = np.array([svd_model.qi[i] for i in movie_inner_ids])

    # PCA projection
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(movie_factors)

    movie_titles = []
    for i in movie_inner_ids:
        try:
            raw_id = trainset.to_raw_iid(i)
            title = movie_df[movie_df['item_id'] == int(raw_id)]['title'].values[0]
        except:
            title = f"Movie {raw_id}"
        movie_titles.append(title)

    df_plot = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "title": movie_titles
    })

    fig = px.scatter(df_plot, x="x", y="y", hover_name="title", title="SVD Movie Embeddings (PCA Reduced)")
    fig.update_layout(
        title="SVD Movie Embeddings (Reduced to 2D via PCA)",
        xaxis_title="Latent Feature 1",
        yaxis_title="Latent Feature 2"
    )
    fig.write_html(save_path, full_html=False, include_plotlyjs='cdn')


def generate_cosine_heatmap_fig(top_n=10, shift=(0, 0), sim_matrix=None):
    
    shiftX, shiftY = shift
    sim_matrix = sim_matrix.iloc[shiftX:shiftX+top_n, shiftY:shiftY+top_n]

    fig = px.imshow(
        sim_matrix.values,
        x=sim_matrix.columns,
        y=sim_matrix.index,
        labels=dict(x="Movie Title", y="Movie Title", color="Similarity"),
        title=f"Cosine Similarity Heatmap (top {top_n})"
    )
    fig.update_xaxes(side="top")
    return fig

dataVisualisationFunctions = [
    interactive_rating_histogram,
    interactive_avg_ratings,
    interactive_num_ratings,
    interactive_cosine_heatmap,
    interactive_svd_embeddings
]