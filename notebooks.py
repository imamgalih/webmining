#!/usr/bin/env python
# coding: utf-8

# Crawling data abstrak dari PTA Universitas Trunojoyo Madura

# In[1]:


get_ipython().system('pip install beautifulsoup4')


# In[2]:


from bs4 import BeautifulSoup
import requests
import csv


# In[3]:


dataAbstract = []
dataFix = []


# In[4]:


def crawlAbstract(src):
    # inisialisasi beautifulsoup4     
    global c
    tmp = []
    page = requests.get(src)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # mengambil data judul     
    title = soup.find(class_="title").getText()
    tmp.append(title)
    
    # mengambil data abstrak   
    abstractText = soup.p.getText()
    tmp.append(abstractText)
    
    return tmp


# In[5]:


def getLinkToAbstract(src):
    # inisialisasi beautifulsoup4
    global c
    page = requests.get(src)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    # mendapatkan semua link menuju halaman detail
    items = soup.find(class_="items").find_all('a')
    # looping setiap link untuk mendapatkan nilai href, dimana link tersebut digunakan sebagai parameter function crawlAbstract agar mendapat data judul dan abstract
    for item in items:
        if item.get('href') != '#':
            tmp = crawlAbstract(item.get('href'))
            # dataAbstract menampung data sementara hasil crawl
            dataAbstract.append(tmp)


# In[6]:


link = "https://pta.trunojoyo.ac.id/c_search/byprod/10"
# mengambil data sampai halaman 19
for i in range(1, 11):
    # memindah halaman menuju halaman selanjutnya     
    src = f"https://pta.trunojoyo.ac.id/c_search/byprod/10/{i}"
    # counter untuk melihat progress berapa persen proses crawling
    print(f"Halaman ke-{i}")
    # memanggil function getLinkToAbstract untuk mendapatkan setiap link ke halaman detail
    getLinkToAbstract(src)


# In[7]:


# looping berikut bertujuan menambahkan kolom index di setiap baris, lalu disimpan di list dataFix
for i in range(1, len(dataAbstract)+1):
    dataAbstract[i-1].insert(0, i)
    dataFix.append(dataAbstract[i-1])

# menyimpan data hasil crawl dengan format csv
header = ['index', 'title','abstract']
with open('dataAbstrak.csv', 'w', encoding="utf-8") as f:
    write = csv.writer(f)
    write.writerow(header)
    write.writerows(dataFix)
# akan ada file dataHasilCrawl.csv berisi id, judul dan abtrak dari pta trunojoyo 
# proses crawling selesai


# In[8]:


get_ipython().system('pip install -U scikit-learn')
get_ipython().system('pip install Sastrawi')


# In[9]:


#import library yg diperlukan untuk proses preprocessing
import re
import nltk
import string 
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
nltk.download('punkt')


# In[10]:


mystring = "&lrm;Some Time W&zwnj;e"
mystring  = re.sub(r"&lrm;", "", mystring)
mystring  = re.sub(r"&zwnj;", "", mystring)


# In[11]:


df = pd.read_csv("dataAbstrak.csv")
df.head(10)


# In[12]:


df['abstract'] = df['abstract'].str.lower()

print('Case Folding Result : \n')
print(df['abstract'].head(5))


# In[13]:


list_review = []
list_review.append(df['abstract'].values.tolist())

print(list_review)


# In[14]:


for i in list_review: #mengambil librari yang sebelumnya
    for j in i:
        df_token = word_tokenize(j)
        for k in df_token:
            kecil = k.lower()  #memanggil fugsi lower
            print(kecil) #hasil dari data yang telah di lower kan / di kecilkan


# In[15]:


stopword = ('abstrak','(',')',',','.',':','%')


# In[16]:


filtering = [] #deklarasi variabel proses filtering
for a in df_token: #perulangan untuk melakukan tokenisasi
        if a not in stopword:
            filtering.append(a)
print(filtering) #mencetak hasil filtering


# In[17]:


factory = StemmerFactory()
stemmer = factory.create_stemmer()

for x in filtering:
    print (x, " : ",stemmer.stem(x))


# In[18]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


# In[19]:


#untuk menghitung jumlah kata yang telah di steming
cv = CountVectorizer()
term_fit = cv.fit(filtering)

print (len(term_fit.vocabulary_))


# In[20]:


print (term_fit.vocabulary_) #mengurutkan berdasarkan urutab abjad kata 


# In[21]:


print (term_fit.get_feature_names()) #mengurutkan berdasarkan urutkan nama 


# In[22]:


#kolom pertama ini berarti jumlah dokumen
#kolom kedua berarti letak katanya
#kolom ketiga hasil dari tf

term_frequency_all = term_fit.transform(filtering)
print (term_frequency_all)


# In[23]:


komen_tf = filtering #memanggil kata
print (komen_tf)


# In[24]:


#term_frequency = term_fit.transform([komen_tf]) #hanya menampilkan hasil document 1
#print (term_frequency)


# In[25]:


dokumen = term_fit.transform(filtering) #hasil perhitungan tf idf dalam 1 doc
tfidf_transformer = TfidfTransformer().fit(dokumen)
print (tfidf_transformer.idf_)

#tfidf=tfidf_transformer.transform(term_frequency)
#print (tfidf) #hasil manual dengan sistem pyhton

