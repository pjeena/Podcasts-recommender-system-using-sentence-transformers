
from mpi4py import MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()


from bs4 import BeautifulSoup
import requests
import pandas as pd 
import numpy as np
import warnings
import time
import re
import pickle
pd.set_option('display.max_columns', None)


def url_extraction(url):
        """
        Takes an url of a category in the website allrecipes.com and returns the urls of all the recipes on the page
        """
        data = requests.get(url)
        soup = BeautifulSoup(data.content,'html.parser')
        s = soup.find_all('a')
    #  s2 = soup.find_all('a',{'class':'comp card--image-top mntl-card-list-items mntl-document-card mntl-card card card--no-image'})
        urls=[]
        for i in s:
            urls.append(i.get('href'))  
                
        return(urls)
    
    
cat_urls = url_extraction('https://podcasts.apple.com/us/genre/podcasts/id26')
cat_urls = cat_urls[27:137]

working_urls = []

for i in cat_urls:
    data = requests.get(i)
    soup = BeautifulSoup(data.content,'html.parser')
    s1 = soup.find_all('div',{'class': 'column first'})[0].find_all('a')
    s2 = soup.find_all('div',{'class': 'column'})[0].find_all('a')
    s3 = soup.find_all('div',{'class': 'column last'})[0].find_all('a')
    s = s1+s2+s3
    urls_within_cat = []
    for ii in s:
        urls_within_cat.append(ii.get('href'))
    working_urls.extend(urls_within_cat)
    
   

delta = 1000
urls = working_urls[my_rank*delta:(my_rank+1)*delta]



    
class Extract():

    def __init__(self,url):
        self.url = url
 
    def get_id(self):
        id_pod = self.url.split('/')[-1]
        return id_pod
    
    
    def get_title(self):
        data = requests.get(self.url)
        soup = BeautifulSoup(data.content,'html.parser')
        s = soup.find_all('span',{'class' : 'product-header__title'})
        if list(s):
            title = s[0].get_text().strip()
            return title
        else:
            return np.nan
    
    def get_studio(self):
        data = requests.get(self.url)
        soup = BeautifulSoup(data.content,'html.parser')
        s = soup.find_all('span',{'class' : 'product-header__identity podcast-header__identity'})
        if list(s):
            studio = s[0].get_text().strip()
            return studio
        else:
            return np.nan
    
    def get_category(self):
        data = requests.get(self.url)
        soup = BeautifulSoup(data.content,'html.parser')
        s = soup.find_all('li',{'class' : 'inline-list__item inline-list__item--bulleted'})
        if list(s):
            category = s[0].get_text().strip()
            return category
        else:
            return np.nan
    
    
    def get_avg_rating_and_volume(self):
        data = requests.get(self.url)
        soup = BeautifulSoup(data.content,'html.parser')
        s = soup.find_all('figcaption',{'class' : 'we-rating-count star-rating__count'})
        if list(s):
            avg_rating = s[0].get_text().split('•')[0].strip()
            ratings_volume = s[0].get_text().split('•')[1].strip() 
            return avg_rating, ratings_volume
        else:
            return np.nan, np.nan
    
    
    def get_episode_count(self):
        data = requests.get(self.url)
        soup = BeautifulSoup(data.content,'html.parser')
        s = soup.find_all('div',{'class' : 'product-artwork__caption small-hide medium-show'})
        if list(s):
            num_episodes = s[0].get_text().strip()  
            return num_episodes
        else:
            return np.nan

    def get_description(self):
        data = requests.get(self.url)
        soup = BeautifulSoup(data.content,'html.parser')
        s = soup.find_all('section',{'class' : 'product-hero-desc__section'})
        if list(s):
            description = s[0].get_text().strip()
            return description
        else:
            return np.nan
    
    
    
    

import time
start = time.time()

ids = []
titles = []
studios = []
categories = []
avg_ratings = []
ratings_volume = []
episodes_count = []
description = []



for url in urls:
    p = Extract(url)
    
    ids.append(p.get_id())
    titles.append(p.get_title())
    studios.append(p.get_studio())
    categories.append(p.get_category())   
    avg_ratings.append(p.get_avg_rating_and_volume()[0])    
    ratings_volume.append(p.get_avg_rating_and_volume()[1]) 
    episodes_count.append(p.get_episode_count())
    description.append(p.get_description())
    
    print(url)
    
end = time.time()
print(end - start)





df = pd.DataFrame(
        {'id': ids,
        'name' : titles,
        'url': urls,
        'studio': studios,
        'category' : categories,
        'episode_count' : episodes_count,
        'avg_rating' : avg_ratings,
        'total_ratings' : ratings_volume,
        'description' : description 
        })


df.to_csv('podcasts_%s.csv' %(my_rank),index=False)
    
    
    