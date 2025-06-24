# 🎬 Movie Recommendation System

This is a **movie recommendation web application** built using the [MovieLens](https://grouplens.org/datasets/movielens/latest/) dataset. It includes **data visualisation**, **movie rating**, and **recommendation features** using two models:
- Cosine Similarity
- Matrix Factorization (SVD)

Built with **Python**, **Pandas**, **Flask**, **Plotly**, and **Surprise** library.

---

## 🚀 Features

### 🔍 Interactive Movie Table
- Browse through the entire movie database (30 movies per page)
- Search by title or genre
- Rate movies you've seen using a star system

### 🤖 Movie Recommendations
- Rate any movies you’ve watched
- Choose how many movies you want to be recommended (default is 5)
- Get recommendations via:
  - ✅ **Cosine Similarity**
  - ✅ **SVD Matrix Factorization**
- Results shown side-by-side for comparison

### 📈 Data Visualisation
Visualise the dataset using:
- Histogram of all ratings
- Number of ratings per movie
- Average rating per movie
- Cosine similarity heatmap between movies  
  - You can customise `top_n`, `shiftX`, and `shiftY` by modifying the URL:  
    Example:  
    ```
    http://127.0.0.1:5000/plots/cosine_heatmap?top=10&shiftX=0&shiftY=0
    ```

---

## 🗂 Dataset

This project uses the [MovieLens Latest Small Dataset (ml-latest-small)](https://grouplens.org/datasets/movielens/latest/) provided by [GroupLens Research](https://grouplens.org/), a research group at the University of Minnesota.

- **Total Ratings:** 100,836  
- **Movies:** 9,742  
- **Users:** 610  
- **Rating Scale:** 0.5 to 5.0 stars  

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)* 5, 4: 19:1–19:19.  
> [https://doi.org/10.1145/2827872](https://doi.org/10.1145/2827872)

The dataset includes:
- `movies.csv`: Titles and genres
- `ratings.csv`: User ratings
- `links.csv`: IMDb and TMDb IDs
- `tags.csv` *(optional)*: User-generated tags

⚠️ This dataset is intended for research and educational purposes only. See full license in `README.txt` from the dataset.

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Recommendation Engine**: Surprise (SVD), Scikit-learn (Cosine Similarity)
- **Frontend**: HTML, CSS, JavaScript
- **Visualisation**: Plotly, Pandas
- **Data**: MovieLens Dataset (ml-latest-small)

---

## 🧪 How to Run

1. **Clone this repository**  
   ```bash
   git clone https://github.com/yourusername/movie-recommendation-sys.git
   cd movie-recommendation-sys
   ```

2. **Create and activate a virtual environment**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the application**
    ```bash
    python run.py
    ```

5. **Open in browser**
    ```bash
    http://127.0.0.1:5000/
    ```

## 🧠 Future Enhancements

- 🔄 Enable persistent user sessions and store custom ratings across visits
- 🔍 Add genre-based filtering and advanced search options
- 🎥 Re-enable TMDb API for showing movie posters and trailers (currently disabled due to API limits)
- 🌐 Deploy online using platforms like Heroku, Vercel, or Render
- 📊 Add more recommendation models (e.g., KNN, content-based, hybrid)

---

## 📬 Contact

For feedback, queries, or collaboration, feel free to reach out via:

📧 **Email**: [your_email@example.com]  
🐙 **GitHub**: [https://github.com/yourusername](https://github.com/yourusername)

---

## ⭐ Acknowledgements

- 📚 **[MovieLens Dataset](https://grouplens.org/datasets/movielens/)** by GroupLens Research, University of Minnesota
- 🐍 **Python** and open-source community
- 🔧 Libraries used:
  - Flask
  - Pandas
  - scikit-learn
  - Surprise (for SVD)
  - Plotly
