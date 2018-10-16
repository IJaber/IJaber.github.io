---
title: "An Aspect-Based Sentiment Analysis of Customer Reviews"
date: 16-10-2018
tags: [machine learning, data science, sentiment analysis, customer reviews, lda, text mining, natural language processing]
header:
  image:
excerpt:
mathjax:
---

## Project Aim
In this project, we proposed a novel approach to Aspect-Based Sentiment Analysis (ASBA) that is capable of accurately retrieving customer sentiments regarding specific aspects through analysing customer reviews. An ABSA is a technique that attempts to discover the most important aspects of a textual document and classify the sentiment polarity of the discovered aspects. Numerous ABSA methods have been proposed in the past, however, the majority of the models proposed were not scalable, and were mainly domain specific. The approach needs to be robust and versatile with the ability to perform across domains and languages, this is because customer reviews are being produced for a variety of products and services. ASBA is generally composed of two tasks, Aspect Detection, and Sentiment Analysis [Schouten and Frasincar, 2016]. Aspect Detection is the process of detecting aspects of an entity in a textual document. There are two main approaches to Aspect Detection – supervised and unsupervised. Sentiment Analysis captures the opinions and attitudes conveyed in text [Liu and Zhang, 2012].
## NOA LDA
By examining relevant literature, it was determined that previous Aspect Detection methods were lacking in either coherence or versatility. Thus, we proposed a NOA to the topic modelling technique LDA (NOA-LDA) for Aspect Detection. We showed through our testing that a NOA-LDA system is a superior method compared with a raw corpus LDA for Aspect Detection; it produced more accurate and coherent aspects. This was demonstrated with two examples: Hotel reviews, and Headphone reviews. Where both examples had considerably greater coherence with a NOA-LDA system than LDA with all POS. This confirms the work by Hu and Liu (2004) whose Aspect Detection approach found frequently occurring nouns and noun phrases, due to their assumption that vocabulary tends to converge when various aspects of a product are discussed. Our approach successfully extended upon the work by Titov and McDonald (2008), and Lin and He (2009) who used topic modelling to discover aspects. We managed to separate aspect words from opinion words through the NOA-LDA, whereas previous methods included both opinion words and aspect words in their approach, resulting in the topic model not being very accurate.  The NOA-LDA system was also significantly more computationally efficient. In terms of the scalability, run time for the NOA-LDA system was significantly faster than with a raw LDA. This was also demonstrated in both examples, the Hotel reviews performed approximately twice as fast with the NOA-LDA system, and the Headphone reviews performed approximately 15 times faster. The NOA-LDA system was also capable of performing across various domains. We showed this by selecting two examples from different domains.
### NOA preprocessing approach
Adequate pre-processing is necessary to ensure the LDA model results in coherent aspects. The Figure below gives an outline of the NOA-LDA pre-processing method for Aspect Detection.
image: "/images/NOApre.jpg"
R code block:
```library(RTextTools)
library(topicmodels)
library(dplyr)
library(tm)
library(lexRankr)

# Import text data
reviews <- reviews_df
# Sentence parse and Isolate text
reviews <- sentenceParse(reviews$comments)
reviews_text <- reviews$sentence
# average length of sentence
mean(sapply(reviews$sentence,function(x)length(unlist(gregexpr(" ",x)))+1))
# Make a Source object
review_source <- VectorSource(reviews_text)
# Make a corpus
review_corpus <- Corpus(review_source)
# Print out
review_corpus
# cleaning
review_corpus <- review_corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("english"))
# tokenize the corpus
review_corpus <- lapply(review_corpus, scan_tokenizer)
# concatenate tokens by document, create data frame
Reviews <- data.frame(text = sapply(review_corpus, paste, collapse = " "), stringsAsFactors = FALSE)
# Remove sentences with less than three words
Reviews <- as.data.frame(Reviews[sapply(strsplit(as.character(Reviews$text)," "),length)>2,])
colnames(Reviews)[colnames(Reviews) == 'Reviews[sapply(strsplit(as.character(Reviews$text), " "), length) > 2, ]'] <- 'text'
# Add sentence ID
Reviews$ID <- seq.int(nrow(Reviews))
str(Reviews)
Reviews$text <- as.character(Reviews$text)
save(Reviews, file = "Reviews.Rda")
# Annotate POS with udpipe
library(udpipe)
udmodel <- udpipe_download_model(language = "english")
udmodel <- udpipe_load_model(file = 'english-ud-2.0-170801.udpipe')
x <- udpipe_annotate(udmodel, x = Reviews$text, doc_id = Reviews$ID)
x <- as.data.frame(x)
str(x)
x$topic_level_id <- unique_identifier(x, fields = c("doc_id"))
save(x, file = "x.Rda")
## Get a data.frame with 1 row per id/lemma and extract only nouns
dtf <- subset(x, upos %in% c("NOUN"))
dtf <- document_term_frequencies(dtf, document = "topic_level_id", term = "lemma")
head(dtf)
## Create a document/term/matrix for building a topic model
dtm <- document_term_matrix(x = dtf)
## Remove words which do not occur that much
dtm_clean <- dtm_remove_lowfreq(dtm, minfreq = 50)
```
## PLSS
In this project, we also presented an intuitive system, PLSS for Sentiment Analysis that successfully assigned ratings to aspects through a Lexicon-based approach. Our approach employed a customer review orientated Lexicon built to resolve the flaws of previous methods which did not consider how customers used opinionated text to assign ratings. For instance, in the AFINN lexicon, the reviewer needs to use words such as “breathtaking” and “outstanding” in order to allocate 5 stars. This sort of vocabulary is common amongst professional critics, but not amongst the majority of customer reviews. We found that extracting adjectives through POS tagging from customer review titles could allow us to assign a fair and realistic polarity rating to opinionated words. Our intuition behind this is the assumption that adjectives used in review titles typically convey the overall customer sentiment towards a product or service, allowing us to retrieve a sentiment score for each adjective. However, the limitation to our PLSS is that we could not quantifiably measure the accuracy of our method. This potentially weakens our approach until further testing is done.

Here's some basic text.

And here's some *italics*

Here's some **bold** text.

What about a [link](https://github.com/dataoptimal)?

Here's a bulleted list:
* First item
+ Second item
- Third item

Here's a numbered list:
1. First
2. Second
3. Third

Python code block:
```python
    import numpy as np

    def test_function(x, y):
      z = np.sum(x,y)
      return z
```

R code block:
```r
library(tidyverse)
df <- read_csv("some_file.csv")
head(df)
```

Here's some inline code `x+y`.

Here's an image:
<img src="{{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg" alt="linearly separable data">

Here's another image using Kramdown:
![alt]({{ site.url }}{{ site.baseurl }}/images/perceptron/linsep.jpg)

Here's some math:

$$z=x+y$$

You can also put it inline $$z=x+y$$
