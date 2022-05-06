# CS753-LatticeInputs

## Team Name
Speech Hackers

## Team Members
Malhar Kulkarni (19D070032) and Kunal Chhabra (19D070031)

## Paper Used
Self-Attentional Models for Lattice Inputs - Matthias Sperber, Graham Neubig, Ngoc-Quan Pham, Alex Waibel

## Link to Paper
https://aclanthology.org/P19-1115.pdf

## Link to PPT

## Brief abstract on Paper
The paper Self-attentional models for Lattice Inputs utilizes various self-attentional models used in multiple different models, such as the Encoder-Decoder seq2seq model given in the paper "Attention is all you need" - Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
Here, in this paper, the authors propose a similar such encoder-decoder architecture using LSTMs, however, with a Lattice input encoded into a sequence at the input of the encoder. This is done by implementing extra features such as binary/probability masking (which are used in the multi-head attention block of the aforementioned model); along with input embedding.

## Brief abstract on Code Implemented
The code we run here, compiles the architecture for only the aforementioned encoder, as the decoder can be used directly from the seq2seq Encoder-Decoder architecture; or as mentioned in et. al. Sperber, the decoder can be built using multiple layers of LSTMs.
For the encoder, however, we make two extra changes. We add an extra method to generate the probability masks for the lattice, using the method of topological sort, given in the above paper. This is a computationally expensive method as its time complexity is O(V^3), where V is the number of nodes in the lattice.
We also include an input positional embedding method, which is specifcally used to find the largest distances (ldists) of each node from the start node, through which we can embed the position of each input node wrt one another; and use this embedding in the input Dropout layer of the encoder. We utilize the DFS algorithm for this method, the time complexity for which is O(E), where E is the number of edges in the lattice, which we assume to be less than V^2.
