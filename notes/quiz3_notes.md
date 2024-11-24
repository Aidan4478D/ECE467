## Quiz 3 Notes

### predicted questions:
- what is attention related to? (as in what are the parameters for calculating it)
- how is multi-head attention different from normal? why do we use it?
- why are positional encodings important? what do they tell us and why is it okay if they're only added to the input?
- how does BERT differ from ELMo? 
- what is BERT's first training objective 
    - involves randomly masking some tokens and then system tries to predict them based on their context
- what is BERT's second pre-training objective/task
    - system learns to predict whether the second sentence is the actual follow up sentence to the first
- why do we not replace all masked tokens with [MASK]
    - creates a mismatch between pre-training and fine-tuning since the [MASK] token does not appear during fine tuning
- how is BERT different than ELMo in fine-tuning?
    - unlike ELMo, the parameters learned during pretraining are not frozen so during fine-tuning, all BERT parameters are adjusted
- why is fine tuning preferable?
    - less expensive to train as you just take the big ass pretrained shit and then tune it to what you need
- what is one of the BERT subsidiaries and how does it differ from BERT?
    - RoBERTa uses 10x more pretraining data, trains for more epochs, and gets rid of the next sentence prediction (NSP) objective

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

tokenization
    - BERT implementation used WordPiece tokens
        - like BPE, the WordPiece algorithm leads to subword tokens that do not cross word boundaries
        - system considers the prediction for the word to be the label assigned to its first subword
    - position embeddings are necessary for the system to get any information that is dependent on word order
        - segment embedding indicates which sentence the corresponding token is a part of
    - original BERT paper fed two sentences at a time as input during pretraining
    - both the position embeddings and the segment embeddings are learned as a part of pre-training
    - the final input embedding for teach token is the sum of the WordPiece embedding, the position embeddings, abd the segment embedding

pre-training
    - the developers pretrained their system for two tasks using unsupervised machine learning
    - pretrained a single transformer encoder for two seperate tasks at the same time
        - one task = language modelling task that ELMo and other systems are pretrained for
            - it's important to use context from both the left and the right of words at the same time
            - first training objective used for bert involves randomly masking some tokens and system tried to predict them
            - thus BERT = masked language model and objective = masked language modeling
        - second task = next sentence prediction (NSP)
            - input = two concatenated sentences with a sentence separator symbol in between
            - for training:
                - half the time the second sentence follows the first in the training corpus
                - other half of time the second sentence is a random sentence from the training corpus
            - system learns to predict whether the second sentence is the actual follow up sentence to the first
    - some later works found that NSP isn't actually that important to include as a pre-training objective

    - masking procedure
        - 15% of the WordPiece tokens from the training data, chosen randomly, are masked
        - allows to obtain a bidirectional pre-trained model
            - creating a mismatch between pre-training and fine-tuning since [MASK] token doesn't appear in fine tuning
            - thus don't always replace "masked" words with [MASK] token
        - 80% of masked tokens replaced with [MASK]
        - 10% of masked tokens replaced with a random token
        - 10% of masked tokens are not changed

    - input
        - consists of components concatenated together  
        - a special token [CLS] + input embeddings from sentence 1 (wordpiece token embedding + position embedding + segment embedding)
        - a sentence seperation token [SEP] + embeddings from sentence 2 (again adding all 3 components)
        - again 15% of tokens are masked
        - 50% of the time sentence 2 follows sentence 1 and the other 50% sentence 1 follows sentence 2
        - "sentences" are actaully more general spans of text, chosen s.t. the input sequence never exceeds 512 tokens
        - if the input sequence is shorter than 512 tokens, it must be padded (as it accepts fix-sized input)

    - process
        - training corpus consists of BookCorpus and English Wikipedia (abount 3.3 billion words in total)
        - loss functions were computed based on the following predictions
            - final hidden state corresponding to the [CLS] token is considered the "aggregate sequence representation for classification tasks"
            - during pretraining, this is used for NSP objective
            - final hidden states corresponding to masked tokens were used for the masked language modeling task
        - original paper created two systems:
            - BERT (base) = 12 transformer encoder layers, vectors of 768, and 12 self-attention heads (110 million trainable params)
            - BERT (large) = 24 transformer encoder layers, vectors of 1024, and 24 self-attention heads (340 million trainable params)
        - there's also other hyperparameters like batch size, learning rate, epochs, etc. which are described in the paper

fine-tuning
    - unlike ELMo the purpose of BERT is not to produce contextual embeddings that can be fed into other architectures
    - the same architecture that has been used for pretraining can be **fine-tuned** for other tasks
    - is an example of **transfer-learning**
        - information learned for one task can also be useful for other related tasks
        - use of static word embeddings probabily fits the definition of transfer learning
        - term more often used when the model being used either stays the same or is slightly modified

    - SQuAD = reading comprehension dataset
        - task usually tackled is as a sequence labelling task applied to the appropriate passage
        - instead of feeding the system sentence 1 and sentence 2, the system is fed the question and the corresponding passage or paragraph that contains (or might contain) the answer
        - final hidden state corresponding the passage tokens are fed to two softmax layers
            - one predicts the prob that each token is the start of the answer
            - the other predicts the prob that each token is the end of the answer
        - for SQuAD 2.0 the system treats answers to unanswerable questions as starting and ending at the [CLS] token
    - for sequence labeling tasks that do not involve sentence pairs (POS tagging for ex.) the second sentence is left out
    - unlike ELMo, the parameters learned during pretraining are not frozen
        - during fine-tuning, all BERT parameters are adjusted
    - compared to pre-training, fine tuning is relatively inexpensive

BERT experiments
    - the GLUE benchmark = collection of datasets related to tasks that seem to rely on natural language understanding
        - GLUE = general language understanding evaluation
        - a collection of diverse natural language understanding tasks
        - there are 9 datasets that comprise the GLUE benchmark
            - multi-genre natural language inference
            - quora question pairs (from the site lmfao)
            - question natural language inference
            - the stanford sentiment treebank
            - corpus of linguistic acceptability
            - semantic textual similarity benchmark
            - microsoft research paraphrase corpus
            - recognizing textual entailment
    - also used SQuAD dataset and SWAG (sittuations with adversarial generations dataset)
        - SWAG contains 113k sentence-pair completion examples that evaluate grounded commonsense inference

    - results
        - for all of the 11 tasks, BERT achieved state-of-the-art results
        - for some results it was a significant improvement over the last
        - this was achieved with a single common architecture
            - based on a transformer encoder BERT is pre-trained using unlabaled data for masked language modeling
            - in the original paper for next sentence prediction
            - same architecture is then fine-tuned using a smaller training set

BERT variations
    - RoBERTa = a robustly optimized BERT pretraining approach
        - primarily uses the same architecture as BERT but it is pretrained differently
            - uses about 10x as much pretraining data
            - uses larger mini-batches
            - trains for more epochs
            - every few epochs it randomly geenrates masked tokens
            - does not use next sentence prediction as pretraining objective
        - achieved new state of the art results for everything BERT was also tested on
    - SpanBERT
        - masks out spans of text as opposed to individual tokens
        - pretraining task is to predict the masked spans based on the context
        - used the same pretraining dataset as BERT
        - did not use the next sentence prediction as an additional training objective
        - achieved better results than BERT on 14 of the 17 tasks

### LLMs, GPT, and RLHF

