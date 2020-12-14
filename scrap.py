#################################
# Author: Chirag Shetty
# Course: CS410, Fall 2020
# Adopted from: MP 2
# Purpose: Scrap news articles from CNN

#Procedure:
# 1) Run scrap.py, by setting dir_url to a topic search page on cnn webpage
# Example: For example this webpage shows for the search 'election': https://www.cnn.com/search?q=election
# 2) Edit the 'name' variable to indicate the topic. Files extracted will be stored with this name
# 3) no_pages: Number of pages to search. Each page has 10 articles
# 4) run python (3.5 used) scrap.py. The extracted docs will be stored in the folder 'cnn'
# 5) You can run it for as many topics as you wish
#################################

from bs4 import BeautifulSoup
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
import re 
import urllib.request
import time
import requests
import re

#create a webdriver object and set options for headless browsing
options = Options()
options.headless = True
driver = webdriver.Chrome('./chromedriver',options=options)


def get_js_soup(url,driver):
    driver.get(url)
    res_html = driver.execute_script('return document.body.innerHTML')
    soup = BeautifulSoup(res_html,'html.parser') #beautiful soup object to be used for parsing html content
    return soup

################Scrapes news listings on CNN search page###########################
# Mostly hand-crafted by observing CNN webpages
# url's are written to cnn_url.txt
# extracted news articles are stored in cnn/
def scrape_cnn_page(dir_url,driver, name, no_pages):
    print ('-'*20,'Scraping page','-'*20)
    art_no = 0
    f_url = open("cnn_url.txt", "w")
    for page_no in range(0,no_pages):           # Controls how many pages you want to crawl. Each page has 10 articles
        print(dir_url+str(page_no*10))
        soup = get_js_soup(dir_url+str(page_no*10),driver)     
        for row in soup.find_all("h3", {"class": "cnn-search__result-headline"}):
            if row:
                url = 'https:'+row.find('a').get('href')
                print(url)
                f_url.write(url)
                soup2 = get_js_soup(url,driver)
                div = soup2.find("div", {"itemprop": "articleBody"})
                if div:
                    for sec in div.findAll('section'):
                        if sec.has_attr("class"):
                            if 'zn-body-text' in sec['class']:
                                div2  = sec.find("div", {"class": "l-container"})
                                if div2:
                                    div3  = div2.findAll("div")
                                    if div3:
                                        art_no = art_no+1
                                        f = open("cnn/"+name+"_"+str(art_no)+".txt", "w")
                                        for row in div3:
                                            #print(row.name)
                                            #print(" ")
                                            if row and row.name=='div':
                                                if row.has_attr('class'):
                                                    if row['class']:
                                                        if 'zn-' in row['class'][0]:
                                                            div4  = row.find("div")
                                                            if div4:
                                                                for heading in div4.findAll(["h3","em","a",'div','strong','span']):
                                                                    heading.decompose()
                                                                if div4.contents:
                                                                    for line in div4.contents:
                                                                        if line:
                                                                            l = str(line)
                                                                            l=l.replace('</li>', '').replace('<li>','').replace('<ul class="cnn_rich_text">','').replace('</ul>','')
                                                                            print(l)
                                                                            f.write(str(l))
                                                            else:
                                                                for heading in row.findAll(["h3","em","a",'div','strong','span']):
                                                                    heading.decompose()
                                                                if row.contents:
                                                                    for line in row.contents:
                                                                        if line:
                                                                            l = str(line)
                                                                            #print(l)
                                                                            l=l.replace('</li>', '').replace('<li>','').replace('<ul class="cnn_rich_text">','').replace('</ul>','')
                                                                            print(l)
                                                                            #print("*************************")
                                                                            f.write(str(l))
                                        f.close()
                print("\n\n")
    f_url.close()

    return 0


def main():
    try:

        #dir_url='https://www.cnn.com/search?size=10&q=jeff%20bezos%20%22amazon%22'
        #name = 'bezos'

        dir_url='https://www.cnn.com/search?size=10&q=elon%20musk%20%22tesla%22'
        name = 'elon' #for file name while storing
        no_pages = 2  # Number of pages to search. Each page has 10 articles
        scrape_cnn_page(dir_url+'&type=article&sort=newest&from=',driver,name, no_pages)
        
    finally:
        print('Closing Driver')
        driver.close()

if __name__=='__main__':
    main()



#######################IGNOORE##############################################
#########################Other USEFUL Variants################################
# def scrape_oan_page(dir_url,driver):
#     print ('-'*20,'Scraping page','-'*20)
#     faculty_links = []
#     #execute js on webpage to load faculty listings on webpage and get ready to parse the loaded HTML 
#     soup = get_js_soup(dir_url,driver)     
#     #print(table.prettify())
#     f_url = open("oan_url.txt", "w")
#     art_no = 0
#     for row in soup.find_all("article"):
#         col = row.find_all("h3")
#         if col:
#             url = col[0].a.get("href")
#             f_url.write(url)
#             print(url)
#             soup2 = get_js_soup(url,driver)
#             div = soup2.find("div", {"class": "entry-content clearfix"})
#             art_no = art_no+1
#             f = open("oan/"+str(art_no)+".txt", "w")
#             for row in div.findAll('p'):
#                 for script in row.findAll("script", "video", 'iframe','br'):
#                     script.decompose()
#                 if row.contents:
#                     for line in row.contents[0]:
#                         f.write(str(line))
#             f.close()
#             print("-------------------------------------------------------------------------------")

#             print("\n\n")
#     return 0


# def scrape_cnn_page(dir_url,driver):
#     print ('-'*20,'Scraping page','-'*20)
#     #execute js on webpage to load faculty listings on webpage and get ready to parse the loaded HTML 
#     soup = get_js_soup(dir_url,driver)     
#     #print(table.prettify())
#     f_url = open("cnn_url.txt", "w")
#     art_no = 0
#     for row in soup.find_all("h3", {"class": "cnn-search__result-headline"}):
#         if row:
#             url = 'https:'+row.find('a').get('href')
#             print(url)
#             f_url.write(url)
#             soup2 = get_js_soup(url,driver)
#             div = soup2.find("div", {"itemprop": "articleBody"})
#             if div:
#                 for sec in div.findAll('section'):
#                     if sec.has_attr("class"):
#                         if 'zn-body-text' in sec['class']:
#                             div2  = sec.find("div", {"class": "l-container"})
#                             if div2:
#                                 div3  = div2.find("div", {"class": "zn-body__read-all"})
#                                 if div3:
#                                     div4  = div3.find_all("div", {"class": "zn-body__paragraph"})
#                                     art_no = art_no+1
#                                     f = open("cnn/"+str(art_no)+".txt", "w")
#                                     for row in div4:
#                                         for heading in row.findAll(["h3","em","a"]):
#                                                 heading.decompose()
#                                         if row.contents:
#                                             for line in row.contents:
#                                                 print(line)
#                                                 f.write(str(line))
#                                     f.close()
#             print("\n\n")
#     f_url.close()

#     return 0

# for row in soup.find_all("h3", {"class": "cnn-search__result-headline"}):
#         if row:
#             url = 'https:'+row.find('a').get('href')
#             soup2 = get_js_soup(url,driver)
#             sec = soup2.find("section", {"class": "zn zn-body-text zn-body zn--idx-0 zn--ordinary zn-has-multiple-containers zn-has-18-containers"})
#             if sec:
#                 div = sec.find("div", {"class": "l-container"})
#                 if div: 
#                     print(div.findALL)
#                     print("-------------------------------------------------------------------------------")

#                     print("\n\n")

# url = 'https://www.oann.com/pro-trump-groups-to-march-and-pray-to-protest-presidents-election-loss/'
# soup = get_js_soup(url,driver)

# div = soup.find("div", {"class": "entry-content clearfix"})
# for row in div.findAll('p'):
#     if row.contents:
#         print(row.contents[0])
#         print(" ")



# headers = {
#     'Access-Control-Allow-Origin': '*',
#     'Access-Control-Allow-Methods': 'GET',
#     'Access-Control-Allow-Headers': 'Content-Type',
#     'Access-Control-Max-Age': '3600',
#     'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0'
#     }
# req = requests.get(url,headers)
# soup = BeautifulSoup(req.content, 'html.parser')
# print(soup.prettify())



