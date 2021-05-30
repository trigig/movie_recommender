## Movie Recommendation Project

### Project scripts are in project/project_code

  * application.py
  * interface.py
  * recommender.py

### How to use?
run application.py

### Data
All models are trained on movielens dataset, educational version.
https://grouplens.org/datasets/movielens/


### Model Information
#### Kmean 
Use sklearn Kmean with 348 clusters.
#### NMF
Use sklearn NMF with 50 Components, max_iter = 500.
#### User Cosine Similarity
Use sklearn cosine similarity.
#### Random
Use numpy sample method.

### Web Application
For this first project I use python Flask framework with HTML and CSS.

### Deployment
Deployment on Heroku: https://trigig.herokuapp.com/

# movie_recommender
