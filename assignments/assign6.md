# Assignment 6

## Reading 

- Read the following tutorials 
  - [Sequence Modeling With CTC](https://distill.pub/2017/ctc/)
  - [CTC forced alignment API tutorial](https://docs.pytorch.org/audio/stable/tutorials/ctc_forced_alignment_api_tutorial.html)
  - [ASR Inference with CTC Decoder](https://docs.pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html)

## Implementation 

Due 12/5. 

**CTC for Word Recognition**

As was learned in the lecture by Eng. Ossama Ghandour, CTC learns the alignment between sequences of acoustic features to sequences of phonemes or words **implicitly**, unlike HMM which requires phoneme segmentation and alignment. We need to replicate the previous assignment but using CTC. 

To replicate your previous HMM-based assignment using CTC for word recognition on the TIMIT dataset, the workflow shifts from modeling explicit state transitions (phoneme boundaries) to directly mapping sequences of acoustic features to sequences of phonemes or words, without requiring precise frame-level alignment.

- Experiment Setup:
  - Use the same TIMIT dataset, extracting audio files and their phoneme transcriptions.
  - Segment audio into utterances (not phonemes), since CTC operates on variable-length sequences.
  - Extract features (MFCCs or Mel filter bank features) for each utterance, resulting in a sequence of feature vectors.
  - Use LSTM with a softmax output layer for each time step, where the output dimension should match the number of phonemes plus the blank symbol.
  - The CTC loss function computes the probability of the correct phoneme sequence, marginalizing over all possible alignments.

- Task:
  - Load TIMIT utterances and phoneme labels.
  - Extract MFCCs for each utterance.
  - Prepare training, development, and test sets.
  - Implement a simple LSTM with CTC loss. Implement the loss and decoding yourself for educational purposes. Train the model on the training set.
  - Implement your own CTC forward and decoding functions. Implement the forward pass for CTC to compute the probability of a given phoneme sequence given the acoustic features. This involves computing forward probabilities for all valid alignments. Implement the Viterbi-like decoding for CTC (often called “CTC beam search”) to find the most likely phoneme sequence. This requires dynamic programming over possible alignments
  - Evaluate on the test set by computing the likelihood of phoneme sequences and the most likely decoded sequence Monitor phoneme error rate (PER) or word error rate (WER) for evaluation.
  - Visualize the learned alignment (e.g., by plotting the alignment probabilities or the most likely path).

