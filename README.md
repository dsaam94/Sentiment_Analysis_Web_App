# Sentiment_Analysis_Web_App
A sample ML web app designed on Flask

For starters please ensure you have following dependecies installed:

<ul>pandas</ul>
<ul>numpy</ul>
<ul>sklearn</ul>
<ul>nltk</ul>
<ul>pyprind</ul>
<ul>flask</ul>

Clone project to a folder and extract all files. Open cmd in that folder and run <i>pip install</i> 'dependencyname'

# Once all dependencies are installed run: 

<i>python create_movie_csv.py</i>

This will download data from http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz and extract it to folder named 'data'.
It's a time consuming processing depending on your internet speed and hardisk read/write speed.
After extracting data a new csv file will be created named, 'movie_data.csv'. This csv file will have all reviews from test and train folders in first column and respective sentiment labels in second column.

<ul>'0 denotes negative sentiment'</ul>
<ul>'1 denotes positive sentiment'</ul>

# Once done run next command:

<i>python train_model.py</i>

This will read csv file and train a logistic classification model in mini batches of data. Once done it will be stored as a serialized object in movieclassifier/pkl_objects folder.

Next is where we will run our flask app.
For Windows users:

Please run:
<i> $env:FLASK_APP="vectorize.py"</i>

For linux users type the following command in terminal:
<i>export FLASK_APP="vectorize.py"</i>

Once your server is up, open localhost:5000 in browser. Enter the sample review in text field. Click submit and check the results.
