## Topic 2 Slides

- Tokenization
    - tokenization = task of preprocessing language s.t. input to NLP applications is a sequence of tokens
    - splits text into words but meaning of "words" in this sense is ambiguous
    - wordform = specific form of a word (ex. pluralized noun, conjugated verb, etc.)
    - thus tokenization is kinda like splitting natural langauge into a sequence of instances of wordforms

- lemmatization
    - lemma = base form of a word (ex. sing, sang, sung, singing all share the same lemma)
    - set of words that share the same lemma must all have the same major part of speech
    - it is common to use a resource to look up the lemma (wordnet)

- stemming
    - simpler than lemmatization
    - involves use of a sequnece of rules to conver wordform to a simpler form
    - ex. ational -> ate (relational -> relate)
    - in theory words with same lemma should map to same stem and words that don't share same lemma shouldn't

- sentence segmentation
    - split a document into sentences
    - may want to split then tokenize but tokenizing aids sentence segmentation
    - end of markers are one complication as decimal points aren't just used to end sentences
    - text normalization = process of tokenization optionally followed by lemmatization or stemmeing and/or sentence segmentation

- chomsky hierarchy
    - four types of formal grammars
        - unrestricted (type 0)
        - context-sensitive (type 1)
        - context-free (type 2)
        - regular (type 3)
    - numbered from most powerful / least restrictive (type 0) to least powerful / most restrictive (type 3)
    - often useful to use the most restrictive grammar that suits the purpose
    - regular grammars are generally powerful enough to specify rules for allowable tokens
    - define regular grammars with regular expressions
        - each regular expression is a formula in a special language that specifies simple classes of strings
        - there's a bunch of different operators in regular expressions which define how to interpret text and a hiearchy
        - can be used to define rules for allowable tokens
        - important for word-based tokenizers
        - also useful for implementing lexical analyzers, search/replace operations, and input validation

- morphology = study of the way words are built up from smaller meaning-bearing units called morphemes
    - morpheme = smallest part of a word that has semantic meaning
    - ex. going => "VERB-go + GERUND-ing"
    - morphological parsing = take wordform and split it into its component morphemes
    - sometimes played an important role for part of speech tagging
    
    - stem = the main morpheme of a word
    - affix = additional meanings of various kinds (prefixes, suffixes, etc.)

    - inflection = combining word stem with a grammatical morpheme resulting in word of the same basic POS as the stem
        - ex. pluralizing a noun or changing the tense of a verb
    - derivation = combination of a word stem with a morpheme, usually resulting in word of a different class
        - ex. appointee from appoint and clueless from clue and happiness from happy
    - compounding = combination of multiple word stems together
        - ex. doghouse
    - cliticization = combation of a word stem with a clitic
        - acts syntactically like a word but is reduced in a form and attached to another word
        - ex. I've = I + have
    
    - morphologically complex languages = polysynthetic languages

- the language instinct
    - a book written by steven pinker
    - humans remember stems, rules related to inflection, rules related to derivation, and irregular words
    - stems generally must be memorized seperately
    - listeme = each unit of the memorized list

- words that have not been seen in a training set = unknown words or out-of-vocabulary (OOV) words
    - one possibility is to convert rare words in training set to a special token and replace those rare words in later docuemtns with the same token
    - new words are frequently introduced to the language or new terms or misspellings so it's not a bad idea to include OOV words

- after word-based tokenization, text normlaization steps are often applied
- some problems tho
    - word based tokenization makes it difficult to deal with unknown words (OOV)
        - in some cases they can be ignored or hacky solutions such as replacing them with special character is common
    - in some languages words are not separated by spaces making it difficult to seperate words algorithmically

- subwords
    - most modern NLP systems tokenize based on smaller pieces of words known as subwords
    - can use morphemes as subword tokens but this is expensive
    - individual characters can be tokens which isn't that bad
    - subword embeddings known as fasttext exist
    - an algorithm to learn a vocabulary of subwords from a training set can be applied

- byte-pair encoding = originally a compression algorithm but it's adapted for tokenizers to generate subword embeddings
    - includes a preprocessing step in which a document is split into words and an extra end-of-word symbol is added to each word
    - leads to subwords that do not cross word boundaries
    - algorithm:
        - starts with a list of words with counts each split into acharacters and ending in the end-of-word symbol
        - also has a seperate vocabulary symbol for each character in the text being processed and EOW character at the end of each word in corpus
        - then iteratively combintes the most frequeny sequential pair of symbols to form a new symbol
            - new symbol then gets added to vocabulary
            - words contianing the new symbol are updated in the corpus being processed
            - algorithm only considers pairs of symbols within words
        - can then apply result of the training to tokenize future text as follows:
            - start with all characters represented by seperate symbols
            - then apply the merges greedily in the order that they were learned
        - note: when tokenizing, the frequencies in the data beign tokenized to not play a role
    - runs for a fixed number of iterations k
    - determines the size of the vocabulary which is typically significantly smaller than the size of a conventional vocabulary
    - most words will be represented as full symbols and only very rare words will have to be represented by their parts
    - can achieve something similar to morpholofical parsing with a much simpler algorithm
    - OOV words are likely to be tokenized into meaningful chunks
    - works well across many languages


## Topic 3 Slides
    
- information retrieval (IR)
    - task of returning or retrieving relevant documents in response to a particular natural language query
    - can be text, photographs, audio, video, etc.
    - collection = set of documents beign used to satisfy user requests
    - term = lexical item (typically a word or token but optionally a phrase) that occurs in the collection

    - conventional IR
        - assume that meaning of a document resides solely in the set of terms that it contains
        - bag of words approach = ignore syntactic (and arguably semantic) information

- vector space model
    - documents and queries are represented as vectors
    - each location in each vector corresponds to one term that occurs in the collection
    - each term's value = term weight
        - often a function of its frequency but can have other factors
    - can generally represent a document vector d(j) = <w(1, j), w(2, j), ... , w(N, j)>
    - N = number of dimensions in the vector = number of distinct terms that occur in the collection
    - w(i, j) = term weight that each term i is assigned in document j
    - most documents will not contain most of the possible terms that can occur so most of the weights are zero and the vectors are sparse
    - document vectors often include weights of more general features = terms or words in a document

- use cosine similarity metric to measure the similarity between two vectors
    - pretty much the same thing as a normalized dot product
- term-document matrices
    - can consider the entire collection of documents to be represented as a sparse matrix of weights
    - w(i, j) represents the weight of term i in document j
    - each column represents a document in the collection and each row represents a term
    - most term weights would be 0

- in practice, do not store the entire matrix since it is huge and sparse
    - rather use inverted index = efficient maps each term in the vocabulary to a list of document IDs
    - each entry represents a document in which the term appears at least once
    - each entry can also optimally include the term's count or weight in the doucment and/or a list of the term's positions in the document
    - including positions is important if you want the user to be able to search for exact phrases
    - often implemented as a hash table

- term frequency (TF) = one component of the term weight which reflects the intuition that terms that occurs frequently of a docuemnt should have higher wieght
    - TF for a term t, in a document d, could just be the count of the word in the document
    - one problem using TF as a word weight si that words that are frequent in general (ex. the, of, and, etc.) should not usually have high weights
- inverse document frequency (IDF) = another component of term weight that reclects the intuition that more discriminant words (occur in less overall docs) should have higher weight
    - IDF typically defined as idf(i) = log(10) N/df(i)
    - N = total number od documents in the collection
    - i = index of a term from the vocabulary
    - df(i) = term's document frequency = number of docts in collection in which term i occurs at least once

- TF*IDF = combination of TD and IDF by multiplication
    - highest weights are assigned to words that are frequent in the current document but rare throughout the collection as a whole

- other issues
    - need to decide whether an IR system should apply stemming (common) or lemmatization (not common)
    - whichever text normalization techniques were applied to the colletion must be applied to the query
    - most conventional IR systems use stop lists containing stop words = lists of very common words to exclude computation
         - doesn't change results much since these words tend to have low IDF values but could make system more efficient
    - some systems applied query expansion which adds synonyms of query terms or other related words to queries to help locate relevant documents that don't use overlapping terms

- text categorization (TC) = automatical labeling of documents based on text contained in or associated with documents in one or more classes/categories
    - some TC tasks assume that categories are independent
    - each document can be assigned to zero, one, or more categories
    - other TC tasks assume mutually exclusive and exhaustive categories s.t. each document is assigned to exactly one category
    - can be used for spam filtering, classification of news into topics, websites, etc.

- machine learning terminology
    - parameters of ML models are learned based on a training set
    - an ML model can be evaluated using a test set
    - there should be no overlap between training and test set
    - hyperparameters = external config variables that are set before training to control how it learns from data
        - use validation set to tune these
        - could also use cross-validation as well
    - supervised machine learning = learning a function that maps inputs to outputs based on a training set that includes labeled examples
    - unsupervised machine learning = learning patterns from a training set without labels

- supvervised ML for TC
    - all TC methods we discuss are supervised machine learning examples
        - if outputs are discrete and have a finite domain this is called classifcation or categorization and each output can be called a label
        - if the outputs have an infinite domain (either discrete or continuous) this is called regression
    - input for categorization task is usually in the form of a vector of features
        - for TC each input represents a text document
    - must deciside what constitutes a term aka what tokenization and text normalization strategies to use
        - ex. whether or not to be case sensitive, use stemming or lemmatization, use a stop list, etc.
        - most conventional approaches that rely on a vector space model normlaize document vectors; some normalize category vectors
    - many TC methods rely on a bag-of-words approach to represent documents

- rocchio / tf*idf
    - stems from rocchio's approach to reelevance feedback
    - uses a vector space model to represent not only documents but also categories
        - each category = sum of docs that belongs to category
        - generally category vectors are normalized
        - each category vector can be thought of as a centroid for the category
    - assuming mutually exclusive and exhaustive categories, method chooses category with highest similarity to document c = argmax(sim(c, d))
    - similarity can be computed using the cosine metric
    - if category vectors are normlaized, a simple dot product is guaranteed to return the same result
    
    - is more complicated when dealing with independent binary categories
        - system needs to convert similarity scores to YES/NO decisions
        - could define thresholds for training
        - this ML technique doesn't work well for binary categories

- K-nearest neighbors
    - example of instance based learning
    - training consists of determining a representation of each training document and recording which instances belong to each category
    - when a new document arrives, it is compared ot those in the training set
    - assuming docs are represented as numerical vectors the "closeness" is just its euclidian distance
    - once all distances are computed, you find the K closest documents to the new document from the training set
    - k can be manually coded or chosen via cross validation or validation set
    - the categories of these k nearest neighbors will be used to select the category or categories of the new document

    - choosing a category
        - if categories are mutually exclusive and exhaustive, the simplest approach is to choose the most common category among the k closest documents
        - if categories are binary, the samplest approach is to choose all categories that are assigned to over half of the k closest documents
    - for conventional text categorization, document vectors would often be TF*IDF vectors
    - better to use similarity (ex. cosine similarity) rather than distance to find and weight k-NN
 
    - efficiency is a major problem as it is computationally intensive
        - training is fast but that doesn't matter as it only needs to be trained once
        - classifying a document requires it to be compared against every other document in the training set this it's slow af
        - there are ways to speed this up but at the cost of slight accuracy

- Naive Bayes
    - probabilistic approach based on Bayes' theorem which calculates conditional probabilities
    - c = argmax[P(d | c) * P(c)]
        - assumes that categories in C are mutually exclusive and exhaustive
        - attempting to predict most likely category c_hat for a document d
        - P(c) is prior probability of a category c
            - estimate based on training set as frequency of training documents falling into category c
        - d can be represented as a set of features with values

    - still considered bag-of-word approach but does not use a vector space model nor rely on TF*IDF word weights
    - features are the words of the vocabulary and are typically considered boolean features
        - aka all that matters is whether each word does or does not appear in a document
    - "naive" assumption is that the probability of seeing a word given its category is not affected by the other words in the document
    - estimate P(t | c) which is the estimated probability of seeing each possible term t in each possible category c
    - probabilitiy estimates are so small so we use log probabilties instead of probabilities

    - P(t | c) is often estimated as the percentage of training docs of category c that contain term t
        - example of a maximum likelihood estimate
    - P(t | c) = # training docs in category c that contain word t / # training docs in category c
        - this works better in our case though can also just do it based on frequency of words
        - aka distinct terms or all term instances
        - looping through all terms gives more weight to words that appear multiple times
    
    - smoothing is very important to avoid 0 probabilties 

- evaluating TC systems
    - evaluating a TC system is generally easier than for IR as there is typically a labeled test set
    - overall accuracy is ogten reasonable metric especially if no single category is domincant and none are extremely small
        - not really reliable for binary categories
        - most documents generally do not belong to most categories
    - confusion matrix:
                         actual = yes   |  actual = no
        prediction = yes      A        _|_     B
        prediction = no       C         |      D

    - overall accuracy = (A + D) / (A + B + C + D)
    - precision = A / (A + B)
    - recall = A / (A + C)
    - F1 = 2 * precision * recall / (precision + recall)

    - micro-averaging = combine all confusion matrices (for all categories) to obtain global counts for A, B, C, D
    - macro-averaging = average together values of overall accuracy, precision, or recall for the individual categories

    - micro averaging weights each devision equally; macro averaging weights each category equally

- publication bias = literature gives us an exaggerated notion of the goodness of at lesat some of these methods
- no free lunch theorem = no machine learning methodology is good for all possible ML tasks

## Topic 3 Slides

- POS can be divided into two broad categories: open classes and closed classes
