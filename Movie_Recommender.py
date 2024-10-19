#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[6]:


import os
print(os.getcwd())


# In[7]:


movies = pd.read_csv('/Users/nishtha/Desktop/Personal/movies-recommender-system/archive/tmdb_5000_movies.csv')
credits = pd.read_csv('/Users/nishtha/Desktop/Personal/movies-recommender-system/archive/tmdb_5000_credits.csv')


# In[8]:


movies.head()
movies = movies.merge(credits,on='title')
movies.head(1)


# In[9]:


# genres
# id
# keywords
# overview
# title
# cast
# crew

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[10]:


movies.info()


# In[11]:


movies.isnull().sum()


# In[12]:


movies.dropna(inplace = True)


# In[13]:


movies.duplicated().sum()


# In[14]:


movies = movies.drop_duplicates()


# In[15]:


import ast
def convert(obj):
    obj = ast.literal_eval(obj)  # To convert string of list into string
    l = []
    for i in obj:
        l.append(i['name'])
    return l


# In[16]:


movies['genres']= movies['genres'].apply(convert)
movies


# In[17]:


movies['keywords'] = movies['keywords'].apply(convert)
movies


# In[18]:


import ast
def convert3(obj):
    obj = ast.literal_eval(obj)  # To convert string of list into string
    l = []
    counter = 0
    for i in obj:
        if(counter!=3):
            l.append(i['name'])
        else:
            break;
        counter +=1
    return l


# In[19]:


movies['cast'] = movies['cast'].apply(convert3)
movies


# In[20]:


import ast
def findDirector(obj):
    l = []
    for i in ast.literal_eval(obj):
        if(i['job']=='Director'):
            l.append(i['name'])
            break;
    return l


# In[21]:


movies['crew'] = movies['crew'].apply(findDirector)
movies


# In[22]:


movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies


# In[23]:


movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['overview'] = movies['overview'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
movies


# In[24]:


movies['tags'] = movies['genres'] + movies['overview'] + movies['keywords'] + movies['cast'] + movies['crew']
movies.head(5)


# In[25]:


movies.drop(['genres','overview','keywords','cast','crew'], axis = 1 , inplace = True)
movies


# In[26]:


movies['tags'] = movies['tags'].apply(lambda x : " ".join(x))
movies


# In[27]:


movies['tags'][0]


# In[28]:


movies['tags'] = movies['tags'].apply(lambda x : x.lower())
np.dtype(movies['tags'])
movies['tags']


# In[48]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000 , stop_words = 'english' )
vector = cv.fit_transform(movies['tags']).toarray()


# In[49]:


vector[0]


# In[50]:


cv.get_feature_names_out()


# In[32]:


import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[44]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[46]:


movies['tags'] = movies['tags'].apply(stem)


# In[51]:


movies['tags']


# In[54]:


from sklearn.metrics.pairwise import cosine_similarity


# In[58]:


similarity = cosine_similarity(vector)
similarity


# In[75]:


sorted(enumerate(similarity[1360]),reverse = True, key = lambda x:x[1])[1:6]


# In[105]:


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distance = similarity[movie_index]
    movies_list = sorted(enumerate(distance),reverse = True, key = lambda x:x[1])[1:6]

    for i in movies_list:
        print(movies.iloc[i[0]].title)


# In[106]:


recommend('Batman')


# In[89]:


import pickle
pickle.dump(movies.to_dict(), open('movies_dict.pkl','wb'))
pickle.dump(similarity, open('similarity.pkl','wb'))


# In[90]:


pickle.dump(similarity, open('similarity.pkl','wb'))


# In[92]:


movies['movie_id']


# In[ ]:




