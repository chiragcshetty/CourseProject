# CourseProject

Student: Chirag C. Shetty (cshetty2)

Paper: ChengXiang Zhai, Atulya Velivelli, and Bei Yu. 2004. A cross-collection mixture model for comparative text mining. In Proceedings of the 10th ACM SIGKDD international conference on knowledge discovery and data mining (KDD 2004). ACM, New York, NY, USA, 743-748. DOI=10.1145/1014052.1014150 [[link](http://sifaka.cs.uiuc.edu/czhai/pub/sigkdd04-ctm.pdf)]

## Introduction

The paper explores a further improvement like PLSA in mining topics. In PLSA, k topics are mined from the entire collection. However, collection may have subset and we may be interested in knowing the topics within a collection while also comparing across different collections. The paper adds one more level of generative variable (lambda_c) and tries to achieve this.

## Data

The original paper from 2004 had used a set on news articles and reviews fro  epionions.com. The site is no longer active  and the dataset wasn't archived anywhere. So I decided to write a scraper, starting with the codes used in the MP's. I chose CNN, which has a search feature on its webpage. So I scrap the webpage resulting from searching a topic of interest and extract the news articles. This mostly involved handcrafting the extraction process.

### Procedure for scraping

The main python file is called scrap.py

1. Edit the 'name' variable to indicate the topic. Files extracted will be stored with this name
2. no_pages: Number of pages to search. Each page has 10 articles
3. Run scrap.py (tested for python3.5), by setting dir_url to a topic search page on cnn webpage
Example: For example this webpage shows for the search 'election': [https://www.cnn.com/search?q=election](https://www.cnn.com/search?q=election)
4. run python (3.5 used) scrap.py. The extracted docs will be stored in the folder 'cnn'
5. You can run it for as many topics as you wish

## Baseline model

For baseline, the paper uses the standard PLSA model. Starting with PLSA code from MP3, background model was added. Thus complete PLSA was implemented at plsa_proj.py. 

## Cross-Collection Mixture Model

The model is implemented at ccmix.py.  Following at the EM update equations from the paper

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/531cce09-14ce-46c7-b6d9-c0923a2e9784/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/531cce09-14ce-46c7-b6d9-c0923a2e9784/Untitled.png)

Procedure:

1. Run scrap.py, by setting dir_url to a topic search page on cnn webpage. Set appropritae variables as described in scrap.py
2. Set N - number of docs of each kind in the collection
3. name_set=list of names of each collection eg: ['elon','bezos']
4. Set number_of_topics
5. Run the code
6. The output displays top_n words in each distribution

## Important notes

1) In calculating c(w,d) that count of word w in doc d across all words and docs, smoothing must be applied. No c(w,d) should be exactly 0. Esle it'll cause divison by zero problem. In the code, term_doc_matrix stores c(w,d)

2) In the EM update steps given in the paper, observe the update for P(w/theta j,i) i.e the collection specific word distributions. Since both numerator and denominator are summed over the entire collection, P(w/theta j,i) will not capture features specific to the sub-collections. They will all behave similarly. Hence in implementation, the summations are only taken over the docs in collection concerned


