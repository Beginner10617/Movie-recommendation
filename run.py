from flask import Flask, render_template, redirect, request, Response
import visualisation
from recommender import rs
import pandas as pd
from flask import jsonify
import plotly.io as pio

dataVisualisationFunctions = visualisation.dataVisualisationFunctions

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/all-movies')
def all_movies():
    movies_df = pd.read_csv("data/ml-latest-small/movies.csv")
    links_df = pd.read_csv("data/ml-latest-small/links.csv")
    merged = pd.merge(movies_df, links_df, on="item_id", how="left")

    movies = merged.to_dict(orient='records')  
    return render_template('movies.html', movies=movies)

@app.route("/get-recommendations", methods=["POST"])
def get_recommendations():
    data = request.get_json()
    print("Received Ratings:", data["ratings"])
    print("Recommendation Count:", data["num_recommendations"])
    ratings_list = data["ratings"]
    user_ratings_dict = {entry['item_id']: entry['rating'] for entry in ratings_list}

    model1_output = rs.recommend_cosine_personal(user_ratings_dict, n=data["num_recommendations"])
    model2_output = rs.recommend_svd_for_custom_ratings(user_ratings_dict, n=data["num_recommendations"])
    print("Model 1 Output:", model1_output)
    print("Model 2 Output:", model2_output)
    return jsonify({"message": "Received", 
                    "MODEL 1": model1_output,
                    "MODEL 2": model2_output
                    })

@app.route("/show-recommendations", methods=["POST"])
def show_recommendations():
    data = request.json
    model1_output = data.get("MODEL 1", [])
    model2_output = data.get("MODEL 2", [])

    return render_template("recommendations.html", model1=model1_output, model2=model2_output)

@app.route('/data-visualisation')
def data_visualisation():
    tab_num = request.args.get('tab', '1')
    if tab_num.isdigit():
        tab_num = int(tab_num)
    else:
        tab_num = 1
    if tab_num < 1 or tab_num > len(dataVisualisationFunctions):
        tab_num = 1
    print(f"Rendering tab {tab_num}")
    if tab_num < 4:
        dataVisualisationFunctions[tab_num - 1](rs.data)
    elif tab_num == 4:
        top_n = request.args.get('top', '10')
        shiftX = request.args.get('shiftX', '0')
        shiftY = request.args.get('shiftY', '0')
        try:
            shift = (int(shiftX), int(shiftY))
        except ValueError:
            shift = (0, 0)
        if top_n.isdigit():
            top_n = int(top_n)
        else:
            top_n = 10
        print(f"Rendering cosine heatmap with top_n={top_n} and shift={shift}")
        dataVisualisationFunctions[tab_num - 1](rs.similarity_df, top_n=top_n, shift=shift)
    elif tab_num == 5:
        rs.load_or_train_svd()
        dataVisualisationFunctions[tab_num - 1](rs.svd_model, rs.trainset, rs.data)
    return render_template('visualisation.html', tab=tab_num)

@app.route("/plots/cosine_heatmap")
def serve_cosine_heatmap():
    top_n = int(request.args.get("top", 10))
    shift_x = int(request.args.get("shiftX", 0))
    shift_y = int(request.args.get("shiftY", 0))

    fig = visualisation.generate_cosine_heatmap_fig(top_n=top_n, shift=(shift_x, shift_y), sim_matrix=rs.similarity_df)
    html_str = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    return Response(html_str, mimetype='text/html')


import random
app.jinja_env.globals['random'] = random.random

if __name__ == '__main__':
    app.run()