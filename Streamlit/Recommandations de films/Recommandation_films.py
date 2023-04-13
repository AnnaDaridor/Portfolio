import streamlit as st
import pandas as pd
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process, fuzz


# Charger le jeu de données de films
movies = pd.read_csv('netflix_titles.csv')

# Pour supprimer les virgules
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(",", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(",", ""))
        else:
            return ''

# Pour lemmatiser le texte
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(text)])

# Pour stemmer le texte
def stem_text(text):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(w) for w in word_tokenize(text)])


# Création d'une colonne 'combined_features'
def preprocess_data(movies_df):
    movies_df.fillna(value='', inplace=True)
    movies_df['description'] = movies_df['description'].apply(lemmatize_text)
    movies_df['description'] = movies_df['description'].apply(stem_text)
    movies_df['combined_features'] = movies_df['director']+" "+movies_df['cast']+ " "+ movies_df['rating']+ " "+ movies_df['listed_in']+ " "+ movies_df['description']
    movies_df['combined_features'] = movies_df['combined_features'].apply(clean_data)
    return movies_df


movies = preprocess_data(movies)

# Création d'un Term Frequency-Inverse Document Frequency
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])


# Calcul de la similarité cosinus entre les vecteurs de mots pondérés TF-IDFpip
cosine_sim = cosine_similarity(tfidf_matrix)


# Fonction de recommandation de films
def recommend_movies(title, cosine_sim=cosine_sim, movies=movies):
    recommended_movies = []
    # Utiliser la bibliothèque fuzzywuzzy pour trouver les titres similaires
    title_matches = process.extractBests(title, movies['title'], scorer=fuzz.token_sort_ratio, limit=10)
    # Vérifier s'il y a des titres similaires suffisamment élevés
    if title_matches and title_matches[0][1] >= 60:
        # Parcourir tous les titres similaires et recommander les films correspondants
        for title_match in title_matches:
            if title_match[1] >= 60:
                # Obtenir l'index du film correspondant au titre
                idx = movies[movies['title'] == title_match[0]].index[0]
                # Obtenir la similarité cosinus de ce film avec tous les autres films
                sim_scores = list(enumerate(cosine_sim[idx]))
                # Trier les films en fonction de la similarité cosinus
                sim_scores = sorted(list(enumerate(sim_scores)), key=lambda x: x[1], reverse=True)
                # Récupérer les 10 premiers films
                sim_scores = sim_scores[1:11]
                # Obtenir les titres des films recommandés
                movie_indices = [i[0] for i in sim_scores]
                recommended_movies.extend(list(movies['title'].iloc[movie_indices]))
            else:
                # Si aucun titre similaire n'est trouvé, retourner une liste vide
                recommended_movies = []
    return recommended_movies





# Interface utilisateur
st.title('Système de recommandation de films')
st.subheader('Base de données Netflix')

# Formulaire pour entrer le titre du film
movie_title = st.text_input('Entrez le titre d\'un film que vous avez aimé :')


# Bouton pour obtenir les recommandations de films
if st.button('Obtenir les recommandations'):
    # Convertir le titre en minuscules
    movie_title = movie_title.lower()
    title_match = process.extractOne(movie_title, movies['title'], scorer=fuzz.token_sort_ratio)
    if title_match[1] >= 60:
        # Obtenir l'index du film correspondant au titre
        idx = movies[movies['title'] == title_match[0]].index[0]
        # Obtenir la similarité cosinus de ce film avec tous les autres films
        sim_scores = list(enumerate(cosine_sim[idx]))
        # Trier les films en fonction de la similarité cosinus
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Récupérer les 10 premiers films
        sim_scores = sim_scores[1:11]
        # Obtenir les titres des films recommandés
        movie_indices = [i[0] for i in sim_scores]
        recommended_movies = movies['title'].iloc[movie_indices].values.tolist()
        st.write('Les films recommandés pour vous sont :')
        for i, movie in enumerate(recommended_movies):
            st.write(f"{i+1}. {movie}")
    else:
        st.warning('Le film demandé ne figure pas dans la base de données.')