## Quiz 3 Notes

### predicted questions:
- what is attention related to? (as in what are the parameters for calculating it)
- how is multi-head attention different from normal? why do we use it?
- why are positional encodings important? what do they tell us and why is it okay if they're only added to the input?
- how does BERT differ from ELMo? 

### Transformers

pre-transformer era
    - through 2017 variations of RNNs (like LSTMs) dominated NLP literature
    - input sequences would be sequences of word embeddings or subword embeddings

    - encoder-decoder networks were used for sequence-to-sequence (seq2seq) tasks like machine translation (MT)
        - encodwer processes the input sequence and the decoder generates an output sequence
        - attention = allows decoder to focus on the portion of the encoder output htat seems most relevant at each time step
            - imprioves results significantly
            - cross-attention = applied to an encoder and used by a decoder

        - most systems used static word embeddings like those produced by word2vec or GloVe
        - ELMo used contextual word embeddings

    - shortcomings of RNNs and CNNs
        - RNNs (and LSTMs) must process input sequentially
        - this menas they can't be efficiently parallelized which prevents them from getting the full benefits of GPUs
        - leads to very slow training

    - CNNs have problems detecting long-distance dependencies
        - refers to relationships between elements that exist relatively far apart in an input sequence
        - in NLP it is often case that proper word requires knowledge of words from much earlier in sentence or previous sentences
        - with enough layers CNNs can handle this in theory
            - in pracitce they don't work work well when long-distance dependencies are significant

transformers
    - "Attention is All You Need" paper presented at 2017 conference developed by Google researchers
        - introduces a novel neural architecture called a transformer
        - relies solely on attention mechanisms dispensing with recurrence and convolutions entirely
        - discusses the transformer as an encoder-decoder model

        ![image](images/transformer.png)

        - transformer is an example of an encoder-decoder network
            - encoder and decoder both rely on stacked layers
            - each layer has sublayers and residual connections (skip connections)
            - some of the sublayers rely on a concept called self-attention
            - others are point-wise fully-connected feedforward neural networks
            - decoder also uses cross attention applied to the encoder output
            - several later works relying on transformers only use the encoder portion (ex. BERT)
            - others only use the decoder portion (ex. GPT)

    - attention
        - related to queries, keys, and values
            - all of these are vectors and at least the queries and keys have the same dimension d(k)
        - queries and keys are the things beoing compared 
        - the values are the things being combined based on the results of the comparisons
        - in different contexts, two or even all three of these may be identical
        - transformer architecture in artcle uses scaled dot-product attention
            - without the scaling factor, additive attention (involved a learned weight matrix) performes better than a dot procut attention when d(k) is large
            - as the dot prouct grows large in magniture pushing the softmax function into regions where it has extremely small gradients
            - scaling factor alleviates this which is important since dot product attention is more efficient

    - multi-head attention
        - attention mechanisms typically learn to focus on one aspect of the input
        - but for NLP tasks there's various aspects of the input that we need to pay attention to for different reasons
        - use multi-head attention
            - first queries, keys, and values are linearly projected using learned mappings
            - next, attention is applied in parallel to each result
            - outputs of the various attention mechanisms are then concatenated and again projected
        - pretty much consists of several dot-product attention layers running in parallel

    - encoder
        - accepts the input which is a sequence of input embeddings combined with positional embeddings
        - in the original paper, it consists of 6 identical layers, each containing two sublayers
            - first sublayer = multi-head self-attention
                - paper didn't invent self-attention
                - each encoding is compared to every other and the weighted encodings are added together to form an output representation
                - a residual connection adds the input of the sublayer to its output & the sum is normalized
            - second sublayer = position-wise fully-connected feedforward neural network
                - consists of "two linear transformations with a ReLU actication in between"
                - weights are shared across positions within a layer but the weights differ between layers
                - a residual connection adds the input of the feedforward sublayer to its output and the sum is normalized

    - positional encodings
        - that little like sine wave that gets combined with the input
        - without these, there is no way for a transfoormer to make use of the order of the input
        - therefore, it's necessary to inject some information about the relative or absolute position of the tokens in the sequence
        - original paper uses sine and cosine functions of different frequencies producing vectors of the same dimension as the embeddings
            - are only added to the input embeddings directly
        - due to residual connections they have an effect throughout the stack
    
    - decoder
        - also accepts input sequence which is a sequence of embeddings + positional embeddings
        - during training the input sequence to  the decoder is the diseired output sequence of the seq2seq model shifted to the right
            - when we apply a transformer, each predicted value is used as the next input
        - like the encoder, it also consists of 6 identical layers
            - each layer consists of three sublayers
            - first sublayer = multi-head self-attention layer
                - sublayer is masked
                - pretty much same as encoder multi-head self-attention layer
            - second sublayer = performs multi-head attention over the output of the encoder stack
                - same sort of attention we learned about sometimes called cross-attention
            - third sublayer = position-wuse fully-connected feedforward neural network
                - same as second sublayer of encoder
        - decoder output is fed through a linear transformation layer and then a softmax layer to predict outputs
            - during training the predicted outputs would be compared to the desired outputs to calculate a loss function
    

- transformers for machine translation (MT)
    - network can be trained end-to-end using stochastic gradient descent and backpropagation
        - using a parallel corpus 
            - sentences from source language are fed to encoder
            - sentences from target language are fed to decoder
        - target sentences are "shifted right"
        - start of sentence marker is inserted as the first new token
        - masking = ensures that predictions for position i can depend only on the known outputs at positions less than i
            - self-attention sub-layer is modified in the decoder stack to prevent positions from attending to subsequent positions
    - during training each sentence pair can be trained in a single pass taking advantage of parallelization

    - applying a trained transformer to MT
        - need to run the decoder sequentially
        - each run of the system predicts one additional token (word or subword in this case)
        - during the first run, you feed in the source sentence to the encoder and only a start-of-sentence symbel to the decoder
            - decoder predicts the first word or subword of the target sentence
            - system remembers the first predicted token of the output
        - during the second run you feed in the source sentence to the encoder, and the start-of-sentence symbol + the first predicted token to the decoder
            - decoder predicts the second token of the target sentence -> system remembers it
            - system repeats this process until an end-of-sentence symbol is predicted

    - comparing architectures
        - he's got this big ass table with complexities, number of operations, max path length
        - n = length of sequence
        - d = size of each vector
        - k = kernel size (for convolutions)
        - r = window size for restricted self-attention
        - self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d
    
    - original results
        - transformers beat all previous architectures for both translation tasks
        - achieved state-of-the-art results at "a fraction of the training cost"
        - emphasis is more about the efficiency of transformers as opposed to the accuracy

### BERT and BERT variations

Bidirectional Encoder Representations from Transformers (BERT)
    - based on transformers, not LSTMs (as ELMo was) to produce contextual embeddings
    - BERT uses only the encoder of a transformer
        - in general can be used for sequence labelling tasks and sequence classification tasks
    - BERT uses only one specific architecture for many NLP tasks
        - elmo produces contextual embeddings that are fed to other architectures

- tokenization
    - BERT implementation used WordPiece tokens
        - like BPE, the WordPiece algorithm leads to subword tokens that do not cross word boundaries
        - system considers the prediction for the word to be the label assigned to its first subword
    - position embeddings are necessary for the system to get any information that is dependent on word order
        - segment embedding indicates which sentence the corresponding token is a part of
    - original BERT paper fed two sentences at a time as input during pretraining
    - both the position embeddings and the segment embeddings are learned as a part of pre-training
    - the final input embedding for teach token is the sum of the WordPiece embedding, the position embeddings, abd the segment embedding

- pre-training
    - the developers pretrained their system for two tasks using unsupervised machine learning
    - pretrained a single transformer encoder for two seperate tasks at the same time
        - one task = language modelling 
    
