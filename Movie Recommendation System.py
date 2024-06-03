# Databricks notebook source

## Movie Recommendation system
Ananya Gopalakrishna (AXG220262)



# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 Search Engine for Movie Plot Summaries

# COMMAND ----------

!pip install nltk
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from math import log

# COMMAND ----------

plot = sc.textFile("dbfs:/FileStore/shared_uploads/axg220262@utdallas.edu/plot_summaries.txt")

# COMMAND ----------

new_line= "\n"
movieskv= plot.flatMap(lambda line: line.split(new_line))

# COMMAND ----------

moviessplit=movieskv.map(lambda line: line.split("\t"))
#moviessplit.take(5)
num_movies= moviessplit.keys().distinct().count()

# COMMAND ----------

display(num_movies)

# COMMAND ----------

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def process_summary(summary):
    tokens = word_tokenize(summary.lower())
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.lower() not in stop_words]
    return ' '.join(processed_tokens)

processed_summaries = moviessplit.map(lambda line: [line[0], process_summary(line[1])])
#processed_summaries.take(5)


# COMMAND ----------

tokenized_summaries = processed_summaries.flatMapValues(lambda x: x.split())
#tokenized_summaries.take(20)

# COMMAND ----------

# ((term, movieID), tf)
term_frequency = tokenized_summaries.map(lambda x: ((x[1], x[0]), 1)).reduceByKey(lambda x, y: x + y)
term_frequency.cache()
#term_frequency.take(50)

# COMMAND ----------

#IDF ((term, movieID), idf)
inverse_document_frequency = term_frequency.map(lambda x: ((x[0][0], x[0][1]), log(num_movies / (x[1] + 1)) + 1))
inverse_document_frequency.cache()
#inverse_document_frequency.take(10)

# COMMAND ----------

# join term_frequency and inverse_document_frequency  ((term, movieID), (idf, tf))
tfidf = inverse_document_frequency.join(term_frequency)
tfidf.cache()
#tfidf.take(10)

# COMMAND ----------

# TF-IDF values ((term, movieID), TF-IDF)
tf_idf = tfidf.mapValues(lambda x: x[1] * x[0])
tf_idf.cache()
#tf_idf.take(10)

# COMMAND ----------

movies = sc.textFile("dbfs:/FileStore/shared_uploads/axg220262@utdallas.edu/movie_metadata.tsv").map(lambda x: x.split("\t"))
metadata= movies.map(lambda x: (x[0], x[2]))
#metadata.take(10)

# COMMAND ----------

# (movieId, (word, TF, IDF, TF-IDF))
tfidf2 = tfidf.map(lambda x : (x[0][1], (x[0][0], x[1][1], x[1][0], x[1][1]*x[1][0])))
tfidf2.cache()
#tfidf2.take(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## a) User enters a single term

# COMMAND ----------

search_terms= sc.textFile("dbfs:/FileStore/shared_uploads/axg220262@utdallas.edu/search.txt").collect()

# COMMAND ----------

# For each search term, find the top 10 movie IDs with the highest TF-IDF values
for term in search_terms:
    # Filter tf_idf RDD for the current term
    term_rdd = tf_idf.filter(lambda x: x[0][0] == term)
    
    # Sort the filtered RDD by the tf-idf values in descending order
    sorted_term_rdd = term_rdd.sortBy(lambda x: -x[1])
    
    # Extract the top 10 movie IDs from the sorted RDD
    top_10_movie_ids = sorted_term_rdd.map(lambda x: x[0][1]).take(10)
    
    # Join the tf_idf RDD with the metadata RDD on the movie ID
    joined_rdd = term_rdd.map(lambda x: (x[0][1], x[1])).join(metadata.map(lambda x: (x[0], x[1])))

    print(f"Top 10 movies for the search term: {term}")
    # Print the top 10 movie names for the current search term
    for movie_id in top_10_movie_ids:
        movie_name = joined_rdd.lookup(movie_id)[0][1]
        print(f" Movie Name: {movie_name}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## b) User enters a query consisting of multiple terms

# COMMAND ----------

# the multi term queries are in queries rdd
queries = sc.textFile("dbfs:/FileStore/shared_uploads/axg220262@utdallas.edu/multi_term_queries.txt").collect()

# COMMAND ----------

df_metadata = metadata.toDF(["MovieID", "movieName"])
movies_data_rdd = df_metadata.rdd

# COMMAND ----------

for multi_term_query in queries:
    multi_terms = multi_term_query.lower().split()
    term_tf = sc.parallelize(multi_terms).map(lambda x : (x, 1)).reduceByKey(lambda x, y : x+y)
    
    tf_idf_modified = tfidf2.map(lambda x :  (x[1][0], (x[1][1], x[1][3]))) # (movieID, (IDF, TF))
    joined_data = term_tf.join(tf_idf_modified)
    joined_data_modified = joined_data.map(lambda x : (x[0], x[1][1][1]))
    tf_data = tfidf2.map(lambda x : (x[1][0], (x[0], x[1][3]))).join(joined_data_modified).map(lambda x : (x[1][0], x[1][1], x[1][0][1]))

    cos_num = tf_data.map(lambda x : (x[0], (x[1] * x[2], x[2] * x[2], x[1] * x[1]))).reduceByKey(lambda x,y : ((x[0] + y[0], x[1] + y[1], x[2] + y[2])))

    #similarity(doc1,doc2)=cos(θ)=(doc1*doc2)/|doc1||doc2|
    cossine_score = cos_num.map(lambda x : (x[0], x[1][0]/(sqrt(x[1][1]) * sqrt(x[1][2]))))
    similar_movies = cossine_score.sortBy(lambda x : -x[1]).map(lambda x : x[0][0])

    similar_movies_rdd = similar_movies.map(lambda x : (x, 1)).reduceByKey(lambda x, y : x+y)
    movie_names_rdd = similar_movies_rdd.join(movies_data_rdd).map(lambda x : (x[0], x[1][1]))

    movie_names_score = cossine_score.map(lambda x : (x[0][0], x[1]))
    result_score_movies = movie_names_score.join(movie_names_rdd).distinct().sortBy(lambda x : -x[1][0]).map(lambda x: (x[1][1], x[1][0]))


    print("Top 10 movies for the multi-term query", multi_term_query)
    result_score_movies.toDF(["MovieName ", "CosSine Score"]).show(10, truncate=False)

