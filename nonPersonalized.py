#IMPORT 
import os
import pandas as pd

#LOAD DATA FROM PICKLE 

# Define directory for MainData
DIR = os.path.join(os.getcwd(), "Data")
MAIN_PICKLE_DIR = os.path.join(DIR, "MainData")

# load all pickle datasets from MainData
def load_main_pickles(pickle_dir=MAIN_PICKLE_DIR):
    datasets = {}
    for file in os.listdir(pickle_dir):
        if file.endswith(".pkl"):
            name = file.replace(".pkl", "")
            datasets[name] = pd.read_pickle(os.path.join(pickle_dir, file))
            print(f"Loaded {name} from {file}")
    return datasets

# dictionary 
main_datasets = load_main_pickles()

# RENAME FOR CONVIENCE
reviews = main_datasets["reviews"]

user_interaction = main_datasets["user_interaction"]

books_joined_clean = main_datasets["books_joined_clean"]

reviews_started_clean = main_datasets["reviews_started_clean"]
reviews_added_clean   = main_datasets["reviews_added_clean"]
reviews_read_clean    = main_datasets["reviews_read_clean"]

# Genre-specific books datasets
books_children    = main_datasets["books_children"]
books_comics      = main_datasets["books_comics"]
books_fantasy     = main_datasets["books_fantasy"]
books_history     = main_datasets["books_history"]
books_mystery     = main_datasets["books_mystery"]
books_poetry      = main_datasets["books_poetry"]
books_romance     = main_datasets["books_romance"]
books_young_adult = main_datasets["books_young_adult"]

#print(books_joined_clean.columns)
#print(user_interaction.columns)
#print(reviews.columns)
#load and rename the remap if needed

#GENERAL NON-PERSONALIZED 
# Recommendation title (THE MOST POPULAR BOOKS)
# baseline method using the most rated, which is the metric goodreads users use to evaluate general popularity
# note that it doesn't take quality into account
def recommend_popular_books(dataset=books_joined_clean, n=10):
    """
    Top-n globally popular books by ratings_count. 
    - used on books dataset.
     """
    return dataset.sort_values("ratings_count", ascending=False).head(n)


# Recommendation title (THE MOST POPULAR BOOKS WITHIN THE LAST 2 YEARS)
# Trending books within the last 2 years of the dataset (recent + popular)
# adds time factor
def recommend_trending_books(dataset=books_joined_clean, years=2, n=10, max_year=2017):
    """
    top-n trending books published in the last `years` relative to dataset end year.
    -used for book datasets 
    """
    recent_books = dataset[dataset["publication_year"] >= (max_year - years)]
    return recent_books.sort_values("ratings_count", ascending=False).head(n)


# Recommendation title (THE MOST INTERACTED WITH NOW)
# Trending books within the last X days of the dataset (recent + popular + interactions)
# adds time factor and interaction
def recommend_trending_by_interactions( dataset=reviews, books_dataset=books_joined_clean, date_col="date_added",lastdays=90,n=10):
    """
    top-n trending books interacted with within the last 90 days.
    - used with reviews dataset to garuntee interaction.
    - interaction type: review.
    - mapped with books_joined_clean to retrieve book names.
    - recommendations with no names or with invalid join are discarded.
    """
    # copy to avoid chaning the original datset
    dataset = dataset.copy()

    # parse dates and 
    dataset[date_col] = pd.to_datetime(dataset[date_col], errors="coerce", utc=True)
    dataset = dataset.dropna(subset=[date_col]) # drop rows with missing dates 
    
    #if the dataset is empty return it as is signaling no recommenations for this datset 
    #we know this dataset has dates but if another one is sent this will ouput nothing 
    if dataset.empty:
        return pd.DataFrame(columns=["title", "name", "recent_interactions"])

    # define cutoff window 
    latest_date = dataset[date_col].max() #find the lastest date since the dataset was done in 2017
    cutoff = latest_date - pd.Timedelta(days=lastdays)

    # filter recent interactions and tell user that there are no recommendations 
    recent = dataset[dataset[date_col] >= cutoff].copy()
    if recent.empty:
        return f"Sorry! No trending books within the last {lastdays} days."

    # count interactions per book => interaction type = reviews 
    counts = recent.groupby("book_id").size().reset_index(name="recent_interactions")
    counts["book_id"] = counts["book_id"].astype(str) #ftype change or merge with books dataset

    # join with the metadata-dataset (since it has both best_book_id and book_id it matches with both)
    if "best_book_id" in books_dataset.columns:
        # create copy to avoid any issues 
        books_dataset= books_dataset.copy()
        books_dataset["best_book_id"] = books_dataset["best_book_id"].astype(str) #type str maintained
        merged = counts.merge(books_dataset, left_on="book_id", right_on="best_book_id", how="inner") 

    elif "book_id" in books_dataset.columns:
        #same thing but with book id in case best_book_id is not avaliable 
        books_dataset = books_dataset.copy()
        books_dataset["book_id"] = books_dataset["book_id"].astype(str)
        merged = counts.merge(books_dataset, on="book_id", how="inner")

    else:
        # this is in case the dataset used is not the big merged one
        raise ValueError("ERROR! book_dataset must contain either 'best_book_id' or 'book_id'")

    # to solve issues with dataset naming inconsistency (author vs name)
    if "authors" in merged.columns:
        if "name" not in merged.columns: #rename authors to name 
            merged = merged.rename(columns={"authors": "name"})
        else:
            merged["name"] = merged["name"].fillna(merged["author unknown"]) # we still want the book title 

    # drop rows without titles for cleaner output
    merged = merged.dropna(subset=["title"]) # we can have books without author names mentioned 
    #but can't do  the same with book titles

    # sort and return top-n
    merged = merged.sort_values("recent_interactions", ascending=False)
    merged = merged.drop_duplicates(subset=["title"]) #drop duplicates

    return  merged.head(n)



# Recommendation title (Community Favorites)
# balance rating score and popularity in the conventional way used by goodreads => Weighted Scoring
def recommend_weighted_books(dataset=books_joined_clean, w1=0.2, w2=0.8, n=10):
    """
    Weighted scoring: popularity + average rating.
    - w1 = popularity 
    - w2 = rating 
    - Applied to books datasets
    - default is to focus on quality, thus the default weights assigned
    """
    #work on a copy 
    dataset = dataset.copy()
    # we normalize both popularity and rating
    dataset["norm_popularity"] = dataset["ratings_count"] / dataset["ratings_count"].max()
    dataset["norm_rating"] = dataset["average_rating"] / 5.0 #max average rating is always 5

    dataset["weighted_score"] = (w1 * dataset["norm_popularity"]) + (w2 * dataset["norm_rating"])
    return dataset.sort_values("weighted_score", ascending=False).head(n)


# Recommendation title (Top Picks)
# use bayesian scoring to compare with the traditional goodreads scoring
def recommend_bayesian_books(dataset=books_joined_clean, C=None, n=10):
    """
    Bayesian scoring: (C*m + ratings_count*avg_rating) / (C + ratings_count).
    - m = global average but weighted to better reflect distribution 
    - C= confidence factor 
    """
    dataset = dataset.copy()
    global_avg = (dataset["average_rating"] * dataset["ratings_count"]).sum() / dataset["ratings_count"].sum()
    if C is None: # you can set C but here I am following what we did in class
        C = dataset["ratings_count"].mean() 
    
    dataset["bayesian_score"] = ((C * global_avg) + (dataset["ratings_count"] * dataset["average_rating"])) / (C + dataset["ratings_count"])
    return dataset.sort_values("bayesian_score", ascending=False).head(n)


# applying to genre => default is weighted 
def recommend_genre_books(genres_dataset, method="weighted", n=10, interactions_dataset=None, **kwargs):
    """
    Recommend books from a genre-specific dataset (e.g. books_romance.pkl).
    Default method = weighted since it is the default for Goodreads.
    - interactions dataset for Trending interactions ONLY
    """
    if method == "popular":
        return recommend_popular_books(genres_dataset, n=n)

    elif method == "trending":
        return recommend_trending_books(genres_dataset, years=kwargs.get("years", 2), 
                                        n=n, max_year=kwargs.get("max_year", 2017))

    elif method == "weighted":
        return recommend_weighted_books(genres_dataset, w1=kwargs.get("w1", 0.2), 
                                        w2=kwargs.get("w2", 0.8), n=n)

    elif method == "bayesian":
        return recommend_bayesian_books(genres_dataset, C=kwargs.get("C", None), n=n)

    elif method == "trending_interactions":
        if interactions_dataset is None:
            raise ValueError("interactions_df must be provided for trending_interactions method.")
        return recommend_trending_by_interactions(interactions_dataset, genres_dataset, 
                                                  date_col=kwargs.get("date_col", "date_added"),
                                                  days=kwargs.get("days", 90), n=n)

    else:
        raise ValueError("Unknown method. Choose from: popular, trending, weighted, bayesian, trending_interactions.")


#USING THE CLEANED REVIEWS DATASETS 

#BUZZING REVIEWS RELATED (ALL THIS USING SORTING )
# Recommendation title (WHAT OTHERS ARE READING)
def recommend_what_others_are_reading(dataset=reviews_started_clean, n=10):
    """
    Non-personalized recommendation: books users have started reading.
    """
    # count how many times each book was started
    counts = dataset.groupby("book_id").size().reset_index(name="started_count")
    merged = counts.merge(books_joined_clean, left_on="book_id", right_on="best_book_id", how="left")
    return merged.sort_values("started_count", ascending=False).head(n)

#BUZZING 
# Recommendation title (Buzzing Books)
def recommend_buzzing_books(dataset=reviews_added_clean, n=10):
    """
    Non-personalized recommendation: books users are actively reviewing.
    """
    counts = dataset.groupby("book_id").size().reset_index(name="review_count")
    merged = counts.merge(books_joined_clean, left_on="book_id", right_on="best_book_id", how="left")
    return merged.sort_values("review_count", ascending=False).head(n)

#FULLY FINISHED NOT STOPPED READING 
# Recommendation title (Page Turners)
def recommend_page_turners(dataset=reviews_read_clean, n=10):
    """
    Non-personalized recommendation: books users finished reading.
    """
    counts = dataset.groupby("book_id").size().reset_index(name="finished_count")
    merged = counts.merge(books_joined_clean, left_on="book_id", right_on="best_book_id", how="left")
    return merged.sort_values("finished_count", ascending=False).head(n)

#to use in main
def get_recommendations(user_request, n=10):
    """
    Main caller function that interprets user request and returns recommendations.
    user_request example: {"type": "genre", "genre": "fantasy", "method": "popular"}
    """
    req_type = user_request.get("type")

    if req_type == "popular":
        return recommend_popular_books(n=n)

    elif req_type == "trending":
        return recommend_trending_books(years=user_request.get("years", 2), n=n)

    elif req_type == "interactions":
        # dataset timeline cutoff
        # explicitly use reviews dataset
        return recommend_trending_by_interactions( df=reviews, 
            date_col=user_request.get("date_col", "date_added"), days=user_request.get("days", 90), n=n)

    elif req_type == "weighted":
        return recommend_weighted_books(w1=user_request.get("w1", 0.2),
            w2=user_request.get("w2", 0.8),n=n)

    elif req_type == "bayesian":
        return recommend_bayesian_books(C=user_request.get("C", None),n=n)

    elif req_type == "genre":
        genre = user_request.get("genre")
        method = user_request.get("method", "weighted")

        # map genre string to dataset variable
        genres_loaded = {
            "children": books_children,
            "comics": books_comics,
            "fantasy": books_fantasy,
            "history": books_history,
            "mystery": books_mystery,
            "poetry": books_poetry,
            "romance": books_romance,
            "young_adult": books_young_adult
        }
        genres_df = genres_loaded.get(genre)
        if genres_df is None:
            return f"Genre '{genre}' not available."

        #remove dupliaction
        user_request = {k: v for k, v in user_request.items() if k != "method"}

        return recommend_genre_books(genres_df, method=method, n=n, **user_request)

    else:
        return "ERROR! Unknown request type."



#display for main
def genre_recommender_interface():
    genres = [
        "children", "comics", "fantasy", "history",
        "mystery", "poetry", "romance", "young_adult"
    ]
    print("Available genres:")
    for g in genres:
        print(f" -> {g}")

    selected_genre = input("\nEnter a genre from the list above: ").strip().lower()
    if selected_genre not in genres:
        print(f"ERROR! Genre '{selected_genre}' not recognized.")
        return

    # build request with default method = weighted
    request = {
        "type": "genre",
        "genre": selected_genre,
        "method": "weighted"
    }

    print("\nTop recommended books:")
    result = get_recommendations(request, n=5)
    if "name" in result.columns:
            print(result[["title", "name"]])
    elif "authors" in result.columns:
            print(result[["title", "authors"]])
