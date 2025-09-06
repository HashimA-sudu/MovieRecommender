from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

app = Flask(__name__)

movies = pd.read_csv('My_Movie_dataset(cleaned1)_utf8.csv')

movies['combined_features'] = (
    movies['Genre'] + ' ' +
    movies['Director'] + ' ' +
    movies['Actors'] + ' ' +
    movies['Plot']
)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])

#this will comput the sImularity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_recommendations(title, num_recommendations=5):

    title = title.lower()
    movies['lowercase_name'] = movies['Movie Name'].str.lower()

    #this will see if the movie is there 
    if title not in movies['lowercase_name'].values:
        return pd.DataFrame()  # Return an empty DataFrame if not found

    #return index
    idx = movies[movies['lowercase_name'] == title].index[0]

    #this will compute similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations + 1]  #if it is the first match then skip it 

    #this will return the indices of similar movies
    movie_indices = [i[0] for i in sim_scores]

    #this will return the recommended movies as a df
    return movies.iloc[movie_indices][['Movie Name', 'Year', 'Genre', 'imdbRating', 'Tomatometer Score']].reset_index(drop=True)


@app.route('/')
def home():
    #set the options
    genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 
              'Family', 'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 
              'Romance', 'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']

    rated = ['13+', '16+', 'Approved', 'G', 'GP', 'M', 'M/PG', 'MA-17', 'NC-17', 'Not Rated', 'PG', 'PG-13', 
             'Passed', 'R', 'TV-14', 'TV-G', 'TV-MA', 'TV-PG', 'Unrated']

    languages = ['Aboriginal', 'Afrikaans', 'Akan', 'Albanian', 'American Sign', 'Amharic', 'Ancient (to 1453)', 
                 'Apache languages', 'Arabic', 'Aramaic', 'Arapaho', 'Armenian', 'Aromanian', 'Assyrian Neo-Aramaic', 
                 'Azerbaijani', 'Bable', 'Bambara', 'Basque', 'Bemba', 'Bengali', 'Berber languages', 'Bosnian', 
                 'Brazilian Sign', 'British Sign', 'Bulgarian', 'Cantonese', 'Catalan', 'Central Khmer', 'Chaozhou', 
                 'Chechen', 'Cheyenne', 'Chinese', 'Cornish', 'Corsican', 'Cree', 'Croatian', 'Crow', 'Czech', 'Danish', 
                 'Dari', 'Dinka', 'Dutch', 'Dzongkha', 'Egyptian (Ancient)', 'English', 'Esperanto', 'Estonian', 
                 'Filipino', 'Finnish', 'Flemish', 'Fon', 'French', 'French Sign', 'Fur', 'Gaelic', 'Galician', 
                 'Georgian', 'German', 'Greek', 'Greenlandic', 'Guarani', 'Gujarati', 'Haitian', 'Hausa', 'Hawaiian', 
                 'Hebrew', 'Hindi', 'Hokkien', 'Hungarian', 'Ibo', 'Icelandic', 'Indonesian', 'Inuktitut', 'Irish Gaelic', 
                 'Italian', 'Japanese', 'Japanese Sign', 'Kalmyk-Oirat', 'Kikuyu', 'Kinyarwanda', 'Kirundi', 'Klingon', 
                 'Korean', 'Korean Sign', 'Kriolu', 'Kurdish', 'Lao', 'Latin', 'Latvian', 'Lingala', 'Lithuanian', 
                 'Luxembourgish', 'Macedonian', 'Malay', 'Maltese', 'Mandarin', 'Maori', 'Marathi', 'Maya', 'Micmac', 
                 'Middle English', 'Min Nan', 'Mixtec', 'Mohawk', 'Mongolian', 'Nama', 'Navajo', 'Neapolitan', 'Nepali', 
                 'None', 'Norse', 'North American Indian', 'Norwegian', 'Nushi', 'Nyanja', 'Ojibwa', 'Old', 
                 'Old English', 'Osage', 'Papiamento', 'Pashtu', 'Pawnee', 'Persian', 'Peul', 'Polish', 'Polynesian', 
                 'Portuguese', 'Pular', 'Punjabi', 'Quechua', 'Quenya', 'Romanian', 'Romany', 'Russian', 'Saami', 
                 'Samoan', 'Sanskrit', 'Scots', 'Serbian', 'Serbo-Croatian', 'Shanghainese', 'Shoshoni', 'Sicilian', 
                 'Sign', 'Sindarin', 'Sinhala', 'Sioux', 'Slovak', 'Somali', 'Songhay', 'Sotho', 'Spanish', 
                 'Spanish Sign', 'Swahili', 'Swedish', 'Swiss German', 'Tagalog', 'Tamashek', 'Telugu', 'Thai', 
                 'Tibetan', 'Tok Pisin', 'Tonga (Tonga Islands)', 'Tswana', 'Tupi', 'Turkish', 'Ukrainian', 'Urdu', 
                 'Uzbek', 'Vietnamese', 'Wayuu', 'Welsh', 'Wolof', 'Xhosa', 'Yiddish', 'Yoruba', 'Zulu']

    #the year range calculation remains the same
    year_ranges = [f"{year}-{year+9}" for year in range(int(movies['Year'].min()), int(movies['Year'].max()), 10)]

    return render_template('index.html', movies=movies['Movie Name'].tolist(), genres=genres, rated=rated, languages=languages, year_ranges=year_ranges)




@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form.get('movie_name', '').strip()

    #this will handle empty movies if it is not founded
    if not movie_name:
        return render_template('recommendations.html', recommendations=[], movie_name="")
    
    recommendations_df = get_recommendations(movie_name)

    #this will make sure that we only convert only if it’s a df
    recommendations = (
        recommendations_df.to_dict('records') if not recommendations_df.empty else []
    )

    return render_template('recommendations.html', recommendations=recommendations, movie_name=movie_name)

@app.route('/movie/<movie_name>')
def movie_details(movie_name):
    movie = movies[movies['Movie Name'] == movie_name].iloc[0]
    poster_url = get_movie_poster(movie_name)
    return render_template('movie_details.html', movie=movie, poster_url=poster_url)




posters = pd.read_csv('posters.csv')

def get_movie_poster(movie_name):
    #this will search for the movie in the posters dataset
    row = posters[posters['Movie'].str.lower() == movie_name.lower()]
    if not row.empty:
        tmdb_id = row['TMDb ID'].values[0]
        # Retrieve poster from TMDB API
        api_key = '9d7c2e7cfa61f948baee3ebfc456c82b'
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return f"https://image.tmdb.org/t/p/w500{data['poster_path']}"
    return None


@app.route('/advanced_search', methods=['POST'])
def advanced_search():
    # Get form inputs
    genre = request.form.get('genre')
    rated = request.form.get('rated')
    language = request.form.get('language')
    year_range = request.form.get('year_range')

    # Filter the movies dataset
    filtered_movies = movies.copy()

    if genre:
        filtered_movies = filtered_movies[filtered_movies['Genre'].str.contains(genre, na=False)]
    if rated:
        filtered_movies = filtered_movies[filtered_movies['Rated'] == rated]
    if language:
        filtered_movies = filtered_movies[filtered_movies['Language'].str.contains(language, na=False)]
    if year_range:
        start_year, end_year = map(int, year_range.split('-'))
        filtered_movies = filtered_movies[(filtered_movies['Year'] >= start_year) & (filtered_movies['Year'] <= end_year)]

    # Convert to a list of dictionaries for rendering
    filtered_movies = filtered_movies.to_dict('records')

    return render_template('advanced_results.html', movies=filtered_movies)


if __name__ == '__main__':
    app.run(debug=True)
