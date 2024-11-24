## Quiz 3 Notes

### predicted questions:


### Transformers

#### pre-transformer era
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

#### transformers
    - "Attention is All You Need" paper presented at 2017 conference developed by Google researchers
        - introduces a novel neural architecture called a transformer
        - relies solely on attention mechanisms dispensing with recurrence and convolutions entirely
        - discusses the transformer as an encoder-decoder model

        ![image info](./transformer.png)
