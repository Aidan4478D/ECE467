# Quiz 2 Notes

- questions
    - when it says that perceptrons are generally allowed to contain multiple nodes that share the same inputs, does that mean it's fully connected
    - in a deep neural network, is "forward propagation" just using the previous node's output as that node's input
    - can you maybe go over how a neural language model can learn word embeddings specifcally like what's going on with the input and projection layer
    - how exaclty are SGNs more computationally efficient than training skip-grams as a neural network


- predictions:
    - static word embeddings = single word embedding is learned for each word in the training set; doesn't take context into account
    - dense vectors work better in every NLP task than sparse vectors
        - it is easier to use dense vectors as features for ML systems as they lead to fewer weights
        - they may help avoid overfitting
    - skip-gram algorithm or continuous skip-gram model
        - general goal is to predict context words based on a current, or center word
    - continuous bag-of-words (CBOW) model
        - general goal is to predict the current, or center word based on context words
    - how are SGNS more efficent using negative sampling rather than using a neural network
        - Instead of calculating probabilities for every word in the vocabulary, SGNS only updates the weights for a few randomly selected negative samples.
    - values at hidden layers and output nodes are changing but weight matrices are not
        - weights change during training but NOT DURING FORWARD INFERENCE
    - how does the two-pass algorithm for training weights in RNNs work
        - first perform forward inference, computing all the h and y values at every time step
        - second "process sequence in reverse" computing the required error terms and gradient
    - RNNs are not limited to a fixed number of prior words when predicting the next word
        - all the words in the sequence so far can affect the prediction of the next word, in theory
    - how do named entity recognition systems work?
        - B = begin and tokens labeled B begin the name of an entity
        - I = inside and tokens labeled I continue the name of a named entity
        - O = outsude and tokens labeled O are not part of a named entity
    - end-to-end training = all parts of the system are trained at once, based on training examples
        - used in training stacked RNNs
        - using SGD and backpropagation
    - what is the vanishing gradient problem and how can it be mitigated
        - multiplication typically reduces the gradients, thus, the further back we go the "less significant" the layers "seem to be"
        - can use ReLUs as an activation function (for CNNs and FF NNs but doesn't really work for RNNs)
        - leads to only very "local" context being "significant"
    - it is the hidden state, not the cell state, that also potentially serves as the output of the cell


## slide deck 1

- in more recent times, neural networks have dominated machine learning
    - most of the work is focused on feedforward neural networks

- neural networks
    - composed of units
    - each unit accepts inputs and computes a weighted sum of the inputs where the weights are adjustable parameters
    - a bias weight is also added
    - an activation function is applied to the weighted sum including the bias weight
    - the result of the activation function is the output, or activation, of the unit
    - weights, including the bias weights, are learned during the training of a neural network

- activation functions
    - introduces a non-linearity which is necessary to represent functions that are not linearly seperable
    - two popular ones are thereshold and the sigmoid function (also = logistic function)
        - threshold not common in deep neural networks
    - other common ones are tanh and rectified linear units (ReLUs) 
        - tanh is like sigmoid but from -1 to 1
        - ReLU is like 0 if negative but value if positive (works well ig)

- perceptrons
    - neural networks which output binary values and do not contain any hidden layers
    - generally allowed to contain multiple nodes that share the same inputs but each node functions completely independent of the other
    - only capable of representing functions that can be represented by individual nodes
    - can only represent linearly seperable functions
        - can't compute XOR

- feedforward neural networks
    - represent functions that are not linearly seperable
    - typically organized into layers
    - output nodes are the far left layer (output layer)
        - output nodes will also include bias weights connected to fixed inputs
        - common type of output layer is the softmax layer
            - pretty much just normalizes all of the outputs
            - often interpreted as a probability distribution
    - there are one or more hidden layers consisting of hidden units
        - each node in the hidden layer has its own bias weight
        - could vectorize each layer and compute all outputs at the same time
        - repreesnt weights between layers as matrices
    - when the nodes in one layer connect with every node from the previous layer, the later layer is said to be fully connected
    - inputs do not have an activation function applied to them and are fed as scalars to the nodes in the first hidden layer
    

- deep neural network 
    - feedforward neural network with more than one hidden layer
    - can implement "forward propagation" by using the previous node's output as that node's input

- creating a neural network
    - choose an architecture (number of nodes, types, sizes of layers, etc.)
    - set hyperparameters
    - obtain a training set, train the network to learn the weightsm and test it on a test set
    - loss function = how far predicted output of nn is from true output
    - generally initialize all the weights to small random values
    - could use stochastic gradient descent to learn the weights of a neural network
    - it's common to loop through the entire training set multuple times -> each pass = epoch

- adjusting weights
    - deriving forumlas to update weights into hidden layers is a little more complicated
    - procedure = backpropagation which computes the partial derivative of the loss function w.r.t each weight into hidden layers
    - once gradients for all weights are calculated they are multipled by a learning rate that controls the size of the adjustments
    - modern neural networks have adaptive learning rates (speficic to SGD i think)
    - reduce overfitting by methods of regularization
        - add an extra term to penalize large weights
        - dropout randomly drops units or weights during training

- cool stuff
    - can make logic gates with neural networks (more specifically perceptrons)
    - can represent neural networks as computation graphs

## slide deck 2

- word embeddings = representaion of words as vectors 

- latenet semantic analysis (LSA)
    - decades old technique that produces vectors representing abstract concepts
    - all text-based sequences can be representd as wegihted sums of those concepts
    - is very interesting but didn't seem to lead to great results for various NLP tasks
    - going back to a term-document matrix
        - can think of each column as being bag-of-word representation of the document
        - can think of each row as being a representation of a word
        - seems reasonable to assume that similar words will occur in many of the same documents
    - LSA inlvolves the use of singular vlaue decomposition SVD applied to the term-document matrix

- semi-modern word embeddings
    - idea is to create a d-dimensional vector, with a fixed d, for each word in a vocabulary
    - typically d ranges from 50-500
    - word embeddings are learned from a corpus using an unsupervised learning approach
    - we will learn about static word embeddings = single word embedding is learned for each word in the training corpus
        - doesn't take context into account
    - embeddings can also be learned to represent subwords and that methods to produce contextual word embeddings can be learned

- pre-word-embedding neural networks
    - conventionally, an input node for every word in the vocabulary would be used
    - typically a large input layer
    - could either be boolean balues, word counts, or TF*IDF weights
    - other input features could be included in additon to the words
    - was still a bag-of-word approach
    - problems:
        - a lot of weights between inputs and first hidden layer led to a lot of overfitting
        - no simple way to incorporate word order into the methodology
        - incorporating bigrams or N-grams would blow up the number of input nodes much further
        - two very similar words would be represented by entirely different nodes

- word embeddings for neural networks
    - number of inputs is related to d, the dimension of the word embeddings
    - might be one word embedding at a time or a fixed number of word embeddings at a time
    - consider task such as sentiment analysis:
        - convolutional neural networks (CNNs): input consists of all the word embeddings from one padded sentence at a time
        - recurrent neural networks (RNNs): typically one word embedding at a time is used as input and words are traversed in a sequence
        - transformers: input typically consists of all the word embeddings from one padded sentence at a time
    - similar (but non-identical) words will have similar word embeddings

- language models & N-grams
    - language model = model that assigns a probability to a sequence of text
    - N-grams were typically used for this purpose  
        - is a sequence of N consecutive tokens 
        - computes estimates of the probabilities of each possible final token of an N-gram given N-1 tokens
    - natural language generation (NLG) is an important component of several NLP applications
        - with N grams, pretty much just consider previous N words and predict next work
        - if trigrams, do bigrams and then predict next word and same for 4-grams
    
- neural language models (NLMs)
    - will be mainly looking at feedforward neural language models
    - will consider a NN architecture that conisders three sequential words at a time and predicts next word

    - projection/embedding layer consists of 3*d nodes where d = dimension of each word embedding vector
    - embedding layer = input layer to NN
    - it's possible to learn word embeddings for current task or to use contextual word embeddings
    - output layer consists of |V| output nodes where V is the set of vocabulary words
        - since output is a softmax layer, the output of the i'th output node is interpreted as the probability that the i'th vocabulary word is the next word

- training a first NLM
    - assuming a static mapping between words and embeddings,
    - train a network using stochastic gradient descent and backpropagation
    - before trainign weights of the NLM are initialzied to small random values
        - in theory could then compute the ideal probability model for each N-gram and then train the model
        - in practice loop through a large training corpus and for each N-gram in the corpus:
            - map the first n-1 words to embeddings and concat these to form the NN input
            - for output, treat the probability of the actual word as 1 and all the other probabilities as 0
        - looping through all the N-grams in the entire training set would be one epoch of training
        - multiple epochs would be applied until there is some sort of convergence
    - no smoothing necessary when training as softmax never outputs a 0 exactly
    - the NN has a good chance of generalizing based on similar words to the current words
    - NN has a good chance of predicting the next word after trigrams that have never been seen
        - can generally handle longer N-grams compared to conventional language models
        - in general NLMs (LTSMs or transformers) make much better predictions than conventional N-gram models

- using NN to learn embeddings
    - by adding one additonal layer to our network it can learn the word embeddings as it learns how to predict the probabilties of the next words
    - such a netwrok is learning embeddings specifically for the task of serving as a neural language model
    - when static word embeddings are learned separately, in practice, training a NN is not the actual method used to learn such embeddings

    - in this model we have an input layer which are all "one-hot vectors"
        - i think input = actual word and it's a 1 when it is the acutal word and 0 when it isn't? I'm lowkey kinda confused ngl
        - okay yeah i was right
    - to get from input layer to embedding layer, a shared set of weights is used to convert each one-hot vector to a word embedding vector
        - each one-hot vector at the input layer is being multiplied by the same weight matrix E to produce a word embedding in the embedding layer
        - columns of E matrix (weight matrix for embeddings) represent the words that are being learned
        - E is being learned along with the rest of the network's weights
    - training of the updated network can proceed in a similar fashion to the last one using SGD and back propagation

- advantages of word embeddings in general
    - dense vectors work better in every NLP task than sparse vectors
        - it is easier to use dense vectors as features for ML systems as they lead to fewer weights
        - they may help aboid overfitting
        - they may do a better job a capturing synonymy aka related words have similar vectors

- word2vec
    - group of related models for producing word embeddings
    - they train a classifier to predict whether a word will show up close to specific other words
        - learned weights become word embeddings
        - these embeddings seem to capture something about the semantics of words
        - embeddings can be used to compute the similarity between words
    - implenentation using two methods:
        - skip-gram algorithm or continuous skip-gram model
            - general goal is to predict context words based on a current, or center word
        - continuous bag-of-words (CBOW) model
            - general goal is to predict the current, or center word based on context words
    - two models are similar and create similar embeddings but one will always turn out to be better choice for any particular task
    - will focus mainly on skip-gram method which is arguably more popular


- skip-gram model
    - window size = how many words around target you're considering for context
    - learns two embeddings for each word, w
        - one is called the target embedding, t, which basically represents w when it is the current/center word surrounded by other context words
        - other is the context embedding, c, which basically represents w which it appears a context word around another target word
    - target matrix, T, is a matrix with |V| (len of vocab) rows and d (dimenions) columns that contains all the target embeddings
    - context matrix, C, is a matrix with d rows and |V| columns that contains all context embeddings

    - during training, only consider context words within a small window of some specified size, L, of each target word
        - target embeddings of center words and context embeddings of nearby words are pushed closer together
        - the target embeddings of words and context embeddings of all other words are pushed further apart
    - probability of seeing w(j) in the context of a target word w(i) is P(w(j) | w(i))
        - this prob is related to the dot product of the target vector for w(i) and the context vector for w(j), aka t(i) * c(j)
    - after training, it's possible to just use target embeddings (rows of T matrix) as the final embeddings
    - it's more common to sum, average, or concat the target embeddings and the context embeddings (columns of C) to product the final vectors

    - overall, model tries to shift embeddings s.t. target embeddings are closer to (have a higher dot product with) context embeddins for nearby words
      and are further away from (have a lower dot product with) context embeddings for words that don't occur nearby
    - in theory could be implemented as a simple feedforward NN 
        - to train, every epoch could loop through every target word / context word pair treating the prob of the context word as 1 and all others as 0
        - training the networks leanrns the target embeddings and context embeddings
        - not implemented like this for efficiency reasons as it would have to compute the dot product of each target embedding with every context embedding for every update
    - instead implement with skip-gram with negative sampling (SGNS)
        - choose k negative sampled words
        - typically choose 5-20 and have probabilties proportional to their unigram frequencies raised to the power of 0.75 (wtf?)
        - rasing to such a power gives rare words a high chance of being selected compared to sampling words based on their frequencies directly
        - choose k negatively sampled words for each context word (ex. if k = 2 and window size = 2, have 4 context words thus 8 negatively sampled words)
        - no longer viewing the model as a neural network and no longer using the softmax function
            - typical to compute probability estimates with the sigmoid function
            - want probabilities of actual context words to be high (close to 1) and probabilties of negative sampled words to be low (close to 0)
    - for each target/context pair (t, c) with k negatively sampled words, we have an objective function that we want to MAXIMIZE
    - will not cover training anymore in depth except randomly initialze T and C matrices and use SGD to maximize the object objection function
        - proceed through multiple epochs over the training set
        - training negative sampling is much more efficient and leads to word embeddings that are approximately as effective

- embeddings for word similarity
    - word2vec word embeddings haved been specifically trained for the purpose of predicting nearby words
    - they are useful for many additonal purposes as well like computing word-to-word similarity
        - can do so by computing dot product between two embeddings to measure their similarity
    - can also search for closed embeddings in d-dimensional embedding space to any specified word (is this how thesaurus works?)

- for visualization, d-dimensional vectors can be mapped to two dimensions  
    - use t-SNE plots to see differences between embeddings
    - cool as differences between differently conjugated words are consistent amongst any word in the corpus
    - thus can be used to solve stuff like analogies lol

- evaluating word embeddings
    - intrinsic evaluation = evaluated on the exact task they are trained for 
        - ex. how well we can predict nearby words
    - extrinsic evaluation = evaluated on techniques that use it for complex tasks to evalute word embeddings
        - ex. text categorization, machine translation, question answering, etc.


## slide deck 3 - RNNs & LSTMs

- recurrent neural network (RNN)
    - any network that contains a cycle within its network connections
    - simple RNN has a single hidden layer with outputs that lead back to its inputs = recurrent link
    - layers can be vectors and weights between layers can be matrices
    - output of one hidden layer (recurrent one) also goes to every other hidden layer node
    - activation value of the hidden layer depends on current input as well as activation value of hidden layer from previous time step

    - the recurrent link layer (h at time i - 1) has it's own weight matrix U
    - thus, h(t) = g(U * h(t(i - 1)) + W * x(t) + b(h))
            y(t) = f(V * h(t) + b(y))

            where g is an activation function and f is something like a softmax layer
            b are bias terms/weights

    - common to depict an RNN as unrolled
        - each time step is drawn seperately and each hidden layer and output layer is drawn seperately
        - values at hidden layers and output nodes are changing but weight matrices are not
            - weights change during training but NOT DURING FORWARD INFERENCE

- forward inference / forward propagation
    - similar to forward propagation in a NN
        - with a FF NN, all the input is fed to the NN at once
        - with an RNN a series or sequence of inputs is fed to the NN across multiple time steps
    - values of hidden nodes and output nodes change at each time step (h and y change)
    - values of weights DO NOT chance

- training an RNN
    - can train a simple RNN usid SGD and backpropagation
    - need a training set and need to define a loss function

    - for simple RNN, use three sets of weights to update
        - W represents weights betwene input and hidden layer
        - V represents weights between the hidden and output layer
        - U represents the weights between the output of the hidden layer (at one time step) to the input of the hidden layer (at the next time step)
        - might also include bias weights that also need to be udpated

    - z(i) typically refers to the weighted sum of the inputs to layer i
    - a(i) typically refers to the activation value from a layer i which applies activation value to z

    - to update V, need to compute the gradient of the loss L w.r.t V using the chain rule
    
    - pretty much just define error terms to compute dL/dV, dL/dW, and dL/dU
    - two pass weight training:
        - first perform forward inference, computing all the h and y values at every time step
        - secod "process the sequence in reverse" computing the error terms and gradients
        - gradients during the backward pass are accumulated and the sum is used to adjust the weights
        - type of training is sometimes called backpropagation through time

- recurrent neural language models (RNLM)
    - can use a simple RNN as a RNLM
    - previous hidden state and current word are used to calculate the current hidden state
    - current hidden state is fed to a softmax which created prob distribution to predict the next word
    - then combine probabilities to evaluate the model

    - RNNs are not limited to a fixed number of prior words when predicting the next word
        - all the words in the sequence so far can affect the prediction of the next word, in theory

- autoregressive generation
    - automatically generates random text
    - once a simple RNN is trained as a RNLM, can apply it to generate random text
        - the probabilties of each possible next word are used to randomly choose a word
        - inputs are pre-trained word embeddings
        - hidden state can be interpreted as a semantic representation of all content that has been processed so far
        - processing ends either after a fixed # of tokens or end of sentence </s> marker

- sequence labeling
    - refers to any task that involves categorizing every item in a sequence
    - one exmaple is part of speech (POS) tagging
    - pretrained word embeddings are inputs and a softmax layer provides prob distribution over POS tags as output at each time step

- named entity recognition (NER) 
    - involves detecting spans of text representing names of people, places, organization, etc.
    - can also include additional concepts such as times, dates, domain specific entities, etc.
    - sometimes the first phase of other tasks like information extraction
    - systems are typically trained using supervised machine learning
    - words in training set are labeled with BIO tags
        - B = begin and tokens labeled B begin the name of an entity
        - I = inside and tokens labeled I continue the name of a named entity
        - O = outsude and tokens labeled O are not part of a named entity

- text categorization
    - RNNs have mostly been successfuly for caregorizing short sequences of text like tweets or individual sentence 
        - could also be applied to longer documents
    - common approach is to have the final hidden state become the input to a FF neural net
    - end-to-end training = all parts of the system are trained at once, based on training examples

- stacked RNNs
    - stacked RNN uses the hidden states produced by one RNN as the inputs to the next
    - can refer to each RNN as a 'layer'
    - the final RNN in the stack produces the final output for the stack
    - the hidden states of the top layer can be used as outputs of the stack
        - can be sent as input to another type of layer such as a softmax layer
    - stacked RNNs outperform single-layer RNNs for many tasks
    - optimal number of RNN layers varies according to the task and the training set
    - the entire stack is trained at once using end-to-end training

- bidirectional RNNs
    - if all input is available at once, we can create another RNN that process the inputs in the opposite or backward direction
    - combining the two RNNs (forward and backward) results in a bidirectional RNN (Bi-RNN or BRNN)
    - at each time step, it is typical to concat the hidden states from each direction although other methods of combining them are possible

    - when a BRNN is used for text categorization, typically only the final states produced by the forward and backward RNNs are concated then fed to a FF NN
    - it's also possible to use a stacked bidirectional RNN

    - seperate models are trained in the forward and backward directions with the output of each model at each time point concatenated to represent the bidirectional state at that time point

- vanishing gradient problem
    - during backpropagation, for each layer or time step that the error is propagated there is a multiplication that takes place
    - multiplication typically reduces the gradients, thus, the further back we go the "less significant" the layers "seem to be"
        - in terms of how they affect the measured loss during training
    - ReLUs mitigate the problem to some extent but are not used for RNNs
    - there also exists the exploding gradient problem which is when multiplication INCREASES gradients
        - capping gradient to some fixed max provides an adequate solution to the exploding gradient problem
    - there is no simple solution to the vanishing gradient problem
    
    - leads to only very "local" context being "significant"
    - in a simple RNN, hidden states are only significantly influenced by previous 2 or 3 words

- Long Short-Term Memory Units (LSTMs)
    - provide one solution for mitigating the vanishing gradient problem
    - often the network as a whole, or each layer of a stacked network is referred to as an LSTM
    - common to depict LSTMs graphically in terms of cells
        - such depictions show the LSTM unrolled - cell is the guy that repeats

    - cells have two sets of values that are passed between other cells
        - really being passed as feedback between one time step and the next
        - one set of values is typically referred to as the cell state (cell context)
        - other set of values is the cell's hidden state
        - textbook refers to two sets of values as seperate layers of an LSTM network but that's not common usage

    - certain components within the cell are typically referred to as gates
    - forgot gate:
        - uses previous hidden states and the current input to decide how much of (and which parts of) the cell state to forget/remember
        - can also incorporate a bias weight into this computation
        - gets multiplied by the cell state element-wise determining which information will be kept/forgotten
        - use f = sigmoid(previous hidden layer * U + current input * W)
        - then update (the forgetting) is k = c(t-1) * f(t) (where * is element wise)
    - new cell content
        - determines how the input, combined with previous hidden state might be used to update the cell state
        - g = tanh(previous hidden layer * U + input * W)
    - input gate
        - used to decide how much of (and what parts of_ new cell content is added to the cell state
        - i = sigmoid(previous hidden layer * U + input * W)
        - update (the adding) is j = g(t) * i(t)
    
    - cell state is simply updated as c(t) = j(t) + k(t)
    - value of the cell state will be passed out of the cell to the next time step

    - output gate
        - used to decide how much of (and which parts of) updated cell state is passed on as the hidden state from the cell
        - formula for this is o = sigmoid(previous hidden later * U + current input * W)
        - output gate does not get applied to cell state directly
        - rather, it is element-wise multiplied by the result of putting the cell state through a tanh function
            - h = o(t) * tanh(c(t))
        - it is the hidden state, not the cell state, that also potentially serves as the output of the cell
            - hidden state might be used to make predictions, serve as an input to another stacked LSTM, etc.

    - in summary:
        - LSTM accepts previous cell's state (context), previous cell's hidden state, and the current input (also a vector) as input
        - cell generates an updated cell state and an updated hidden state which are passed to next cell
        - hidden state could also serve as cell's output
            - is visible outside of cell
            - can be used for classification, make predictions, as input to another stacked LSTM layer, etc.
        - gates, and process of calculating new candidate values, all ultimately involve ordinary nodes and weights in a neural net
        - can also learn weights (aka train the LSTM) via end-to-end training using SGD and backpropagation

- complex neural networks
    - LSTMs can be bi-directional and/or stacked
    - they can be applied to any of the previously discussed tasks
    - outputs from recurrent structures can be fed as input to feedforward networks for categorization
    - modern DL libraries make it relatively easy to build complex networks out of standard layers
    - regardless of architecture, NNs can be trained end-to-end using SGD and backpropagation

## slide deck 4

