
import streamlit as st
import pandas as pd
import os
import requests
import joblib

from pathlib import Path
from qdrant_client import QdrantClient, models

# -----------------------------
# Configuration
# -----------------------------

st.set_page_config(
    page_title="Movie Similarity Finder",
    page_icon="🎬",
    layout="wide"
)

DATA_FILE = Path("data/movies.csv")
DETAILS_FILE = Path("movie_details.csv")
VECTORIZER_FILE = Path("description_vectorizer.joblib")

GENRE_COLLECTION = "movie_similarity"
DESCRIPTION_COLLECTION = "movie_descriptions"

st.title("🎬 Movie Similarity Finder")
st.write(
    "Find movies with similar genres and plot descriptions."
)

# -----------------------------
# Load movie catalogue
# -----------------------------

@st.cache_data
def load_movies():
    df = pd.read_csv(DATA_FILE)
    df["movieId"] = df["movieId"].astype(str)
    df["genres"] = df["genres"].fillna("")
    return df


movies_df = load_movies()


@st.cache_data
def load_links():
    links = pd.read_csv("data/links.csv")
    links["movieId"] = links["movieId"].astype(str)
    links["imdbId"] = pd.to_numeric(
        links["imdbId"], errors="coerce"
    )
    return links.set_index("movieId")["imdbId"].to_dict()


imdb_lookup = load_links()


# -----------------------------
# Load movie descriptions
# -----------------------------

@st.cache_data
def load_descriptions():
    df = pd.read_csv(DETAILS_FILE)
    df["movieId"] = df["movieId"].astype(str)
    df["overview"] = df["overview"].fillna("")

    return dict(zip(df["movieId"], df["overview"]))


descriptions = load_descriptions()


@st.cache_resource
def load_vectorizer():
    return joblib.load(VECTORIZER_FILE)


vectorizer = load_vectorizer()


# -----------------------------
# Connect to Qdrant
# -----------------------------

try:
    qdrant_url = st.secrets["QDRANT_URL"]
    qdrant_api_key = st.secrets["QDRANT_API_KEY"]

    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key
    )

except (KeyError, FileNotFoundError):
    client = QdrantClient(url="http://localhost:6333")


# -----------------------------
# Movie posters
# -----------------------------

@st.cache_data(ttl=86400)
def get_poster(movie_id):
    api_key = st.secrets.get("OMDB_API_KEY", os.getenv("OMDB_API_KEY"))
    imdb_value = imdb_lookup.get(str(movie_id))

    if not api_key or pd.isna(imdb_value):
        return None

    imdb_id = f"tt{int(imdb_value):07d}"

    try:
        response = requests.get(
            "https://www.omdbapi.com/",
            params={"apikey": api_key, "i": imdb_id},
            timeout=10
        )

        response.raise_for_status()
        data = response.json()

        poster = data.get("Poster")

        if poster and poster != "N/A":
            return poster

    except requests.RequestException:
        return None

    return None


# -----------------------------
# Create genre vectors
# -----------------------------

genres_list = sorted({
    genre
    for genres in movies_df["genres"]
    for genre in genres.split("|")
    if genre and genre != "(no genres listed)"
})

genre_to_index = {
    genre: index for index, genre in enumerate(genres_list)
}


def create_vector(genres):
    vector = [0.0] * len(genres_list)

    for genre in genres.split("|"):
        if genre in genre_to_index:
            vector[genre_to_index[genre]] = 1.0

    return vector


# -----------------------------
# Prepare genre collection
# -----------------------------

@st.cache_resource
def prepare_collection():
    if not client.collection_exists(GENRE_COLLECTION):
        client.create_collection(
            collection_name=GENRE_COLLECTION,
            vectors_config=models.VectorParams(
                size=len(genres_list),
                distance=models.Distance.COSINE
            )
        )

        points = []

        for _, movie in movies_df.iterrows():
            points.append(
                models.PointStruct(
                    id=int(movie["movieId"]),
                    vector=create_vector(movie["genres"]),
                    payload={
                        "title": movie["title"],
                        "genres": movie["genres"]
                    }
                )
            )

        client.upload_points(
            collection_name=GENRE_COLLECTION,
            points=points,
            wait=True
        )

    return True


with st.spinner("Preparing movie catalogue..."):
    prepare_collection()

st.success(
    f"Ready! {len(movies_df):,} movies loaded, "
    f"{len(descriptions):,} descriptions available."
)


# -----------------------------
# Movie search
# -----------------------------

selected_title = st.selectbox(
    "Search for a movie:",
    options=movies_df["title"].tolist(),
    index=None,
    placeholder="Type a movie title..."
)


# -----------------------------
# Find similar movies
# -----------------------------

if selected_title:

    selected_movie = movies_df[
        movies_df["title"] == selected_title
    ].iloc[0]

    selected_id = selected_movie["movieId"]
    selected_overview = descriptions.get(selected_id, "")

    st.subheader("Selected movie")
    st.write(f"**{selected_movie['title']}**")
    st.write(f"Genres: {selected_movie['genres']}")

    if selected_overview:
        with st.expander("Read plot description"):
            st.write(selected_overview)
    else:
        st.info("No plot description is available for this movie.")

    number_of_results = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=5
    )

    if st.button("Find Similar Movies", type="primary"):

        with st.spinner("Comparing genres and movie descriptions..."):

            # 1. Genre similarity

            genre_vector = create_vector(
                selected_movie["genres"]
            )

            genre_results = client.query_points(
                collection_name=GENRE_COLLECTION,
                query=genre_vector,
                limit=100,
                with_payload=True
            ).points

            genre_scores = {
                str(result.id): result.score
                for result in genre_results
            }

            # 2. Description similarity

            description_scores = {}

            if selected_overview.strip():

                sparse_matrix = vectorizer.transform(
                    [selected_overview]
                )

                sparse_row = sparse_matrix.getrow(0)

                if sparse_row.nnz > 0:

                    sparse_query = models.SparseVector(
                        indices=sparse_row.indices.tolist(),
                        values=sparse_row.data.tolist()
                    )

                    description_results = client.query_points(
                        collection_name=DESCRIPTION_COLLECTION,
                        using="description",
                        query=sparse_query,
                        limit=100,
                        with_payload=True
                    ).points

                    description_scores = {
                        str(result.id): result.score
                        for result in description_results
                    }

            # 3. Combine both recommendation lists

            candidate_ids = (
                set(genre_scores.keys())
                | set(description_scores.keys())
            )

            recommendations = []

            for movie_id in candidate_ids:

                if movie_id == selected_id:
                    continue

                genre_score = genre_scores.get(movie_id, 0.0)
                description_score = description_scores.get(
                    movie_id, 0.0
                )

                if selected_overview.strip():
                    combined_score = (
                        0.5 * genre_score
                        + 0.5 * description_score
                    )
                else:
                    combined_score = genre_score

                movie_match = movies_df[
                    movies_df["movieId"] == movie_id
                ]

                if movie_match.empty:
                    continue

                movie_info = movie_match.iloc[0]

                recommendations.append({
                    "movie_id": movie_id,
                    "title": movie_info["title"],
                    "genres": movie_info["genres"],
                    "genre_score": genre_score,
                    "description_score": description_score,
                    "combined_score": combined_score
                })

            recommendations.sort(
                key=lambda x: x["combined_score"],
                reverse=True
            )

        # 4. Display recommendations

        st.subheader("Similar movies")

        if not recommendations:
            st.warning("No recommendations were found.")

        else:
            for movie in recommendations[:number_of_results]:

                poster = get_poster(movie["movie_id"])
                overview = descriptions.get(
                    movie["movie_id"], ""
                )

                image_col, info_col = st.columns([1, 3])

                with image_col:
                    if poster:
                        st.image(poster, width=130)
                    else:
                        st.write("Poster unavailable")

                with info_col:
                    st.markdown(f"### {movie['title']}")

                    st.write(f"Genres: {movie['genres']}")

                    st.write(
                        f"**Overall similarity: "
                        f"{movie['combined_score']:.3f}**"
                    )

                    st.caption(
                        f"Genre: {movie['genre_score']:.3f} | "
                        f"Description: "
                        f"{movie['description_score']:.3f}"
                    )

                    if overview:
                        with st.expander("Read plot description"):
                            st.write(overview)
                    else:
                        st.caption(
                            "No plot description available."
                        )

                st.divider()
