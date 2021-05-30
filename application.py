from sys import argv
from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import redirect
from interface import movies_genres, ratings, movies, user_item_matrix
from recommender import select_random_movies, recommend_with_NMF, recommend_from_same_cluster, recommend_with_user_similarity


app = Flask(__name__) #This is where everything starts. Look from this point
#template - html /flamework
#statis - css - thing to make it beautiful
# here we construct a Flask object and __name__ sets the script to the root


#first landing place
@app.route('/',methods = ['POST','GET'])
def index():  
    if request.method == 'POST':
        title = request.form.getlist('movie_title')
        rating = request.form.getlist('movie_rating')
        user_rating =dict(zip(title, rating))    
        if request.form.get('action1') == 'Kmeans':            
            recs = recommend_from_same_cluster(user_rating, movies)
            return render_template('recommender.html',recs=recs)                                
        elif  request.form.get('action2') == 'NMF':
            nmf_recs = recommend_with_NMF(user_rating,user_item_matrix) 
            return render_template('nmf.html',nmf_recs= nmf_recs)   
        elif request.form.get('action3')== 'User Cosine Similarity':
            cosim_recs = recommend_with_user_similarity(user_rating, user_item_matrix)
            return render_template('cosim.html', cosim_recs=cosim_recs)        
        elif request.form.get('action4')== 'Random':
            random_recs = select_random_movies(movies, user_rating)
            return render_template('random_rec.html', random_recs=random_recs)
           
    return render_template('index.html')
    

  

@app.route('/movie/<int:movieid>/<imdbid>')
def movie_info(movieid, imdbid):
    return render_template('movie_info.html', movieid=movieid,imdbid=imdbid)
  

@app.route('/imdb/<id>')
def link_to_imdb(id): 
    imdb_link= "https://www.imdb.com/title/tt"+str(id)
    return redirect(imdb_link)


if __name__ == '__main__':
    app.run(debug=True, port=5000)   

