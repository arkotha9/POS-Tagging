# POS-Tagging
## Python version: 3.10.12
## Packages used: numpy and json
import numpy as np
import json
For other instructions, please check the how-to-run file.
# TASK 1 - Data Preprocessing

In Task 1, the training and development data are processed, and a vocabulary is created.

## Data Processing

- Training and development data are in the form of a list of dictionaries, each containing 'index,' 'sentence' (list of words - tokenized), and 'labels' corresponding to each word.
- The training data is read and stored in a list.

## Vocabulary Creation

- Created a separate list and a corresponding dictionary called `word_freq` to store the frequencies of each word.
- Also created a list called `train_labels` to store all the unique training labels (45 tags).

## Unknown Word Handling

- Choose a threshold of 2. Any word with a frequency < 2 is considered '< unk >' (unknown tagged).
- Replaced all the words in the train and dev datasets having a frequency less than the threshold (2) as '< unk >'.

## Vocabulary Sorting

- The dictionary is sorted in descending order based on occurrences, excluding the frequency of the '< unk >' tags.
- The sorted dictionary with '< unk >' as the first tag and the rest in descending order of frequency is written into a 'vocab.txt' file.

### Results

- **Vocabulary Size:** 23183 (including '< unk >')
- **Frequency of '< unk >' in Train Dataset:** 20011

# TASK 2: HMM Probability Calculation

In Task 2, probabilities for Hidden Markov Model (HMM) are calculated based on the training data.

## Counting Frequencies

- Calculated the initial, transition, and emission probabilities by iterating over the lines in the training data.
- Found the count of all possible states in initial states by iterating over the first word of each sentence.
- For transmission and emission probabilities, iterated over the possible combinations of (prev_state, state) and (state, word), keeping track of counts.

### Three New Dictionaries

1. **transition_counts:**
   - Stores tuples of (prev_state, state) as keys and the number of times this combination occurs in the training set as values.

2. **emission_counts:**
   - Stores tuples of (word, state) as keys and the number of times this combination occurs in the training set as values.

3. **initial_state_counts:**
   - Stores the state as keys and the number of times this state occurs at the beginning of the sentence as values.

4. **overall_state_counts:**
   - Stores the state as keys and the frequency of the state in the training data as values.

## Probability Calculation

- Calculated initial, emission, and transition probabilities using the counts.

### Dictionaries:

1. **initial_probabilities:**
   - Stores initial probabilities when the given state is the first tag in the sentence (no prior tag to depend on).
   - Calculated as `initial_state_counts / total_no_of_sentences`.

2. **transition_probabilities:**
   - Stores transition probabilities.
   - Calculated as `transition_counts / overall_state_counts of that state`.

3. **emission_probabilities:**
   - Stores emission probabilities.
   - Calculated as `emission_counts / overall_state_counts of that state`.

## Results

- Number of Transition Parameters: 1351
- Number of Emission Parameters: 30303

The dictionaries `initial_probabilities`, `transition_probabilities`, and `emission_probabilities` are written into a 'hmm.json' file, which is a dictionary having the above dictionaries as elements.

# TASK 3: Greedy Decoding Algorithm

The algorithm focuses on determining the most probable part-of-speech tag for each word in a sentence using greedy decoding.

### Initialization for First Word:

- Determines the most probable part-of-speech tag (stored in `most_probable_s`) for the first word in each sentence.
- Iterates through a list of possible part-of-speech tags (`train_labels`).
- Calculates the initial probability of each tag for the first word in the sentence.
- Probability is computed by multiplying the initial probability of the tag (`i_prob.get(s, 1e-6)`) with the emission probability of the tag for the first word in the sentence (`e_prob.get((s, sentence[0]), 1e-6)`).
- Tracks the most probable tag (`most_probable_s`) by comparing calculated probabilities.

### Decoding for Words Beyond the First Word:

- For words beyond the first word (i.e., `i > 0`), calculates the most probable part-of-speech tag given the previous tag (`prev_tag`).
- Initializes `final_prob_of_s` to negative infinity to track the maximum probability encountered.
- Iterates through all possible pairs of part-of-speech tags (`train_labels`) to calculate the probability of the current tag (`s`) being the correct tag for the current word in the sentence.
- Probability is calculated as the product of transition probability (`t_prob.get((prev_tag, s), 1e-6)`) from the previous tag to the current tag and emission probability (`e_prob.get((s, sentence[i]), 1e-6)`) of the current tag emitting the current word.
- Selects the tag with the highest probability as the predicted tag for the current word.
- This process is repeated for each word in the sentence as part of the greedy decoding algorithm.

### Accuracy Calculation:

- Calculates accuracy using the formula: `accuracy = correctly_predicted_labels / total_labels`.
- Achieves an accuracy of 0.93479 on dev data after replacing unknown words with `<unk>` tags and using a smoothing of 1e-6 for all probabilities.

## Function for Predictions on Test Data

- Implements a function `greedy_on_test_data` to create predictions for the `test.json` dataset.
- Stores the predicted labels, along with the sentence and index information, in the file `greedy.json`.

# Task 4: Viterbi Decoding Algorithm

- Utilizes dynamic programming to compute state probabilities for each position in a sentence.
- Iterates through possible part-of-speech tags (states) in a sentence.

### Initialization for Initial State Probabilities:

- Iterates through possible tags (S), computing the initial probabilities of each tag at the beginning of the sentence.

### Iteration through Words for Maximum Probability:

- Iterates through each word (O) in the sentence, calculating the maximum probability of reaching a particular tag (S[s]) at each position.
- Considers all possible previous tags (S[k]) for the calculation.
- Calculates the probability of transitioning from S[k] to S[s], combined with the probability of the current word having the tag S[s].

### Updating Trellis Matrix and Pointers Matrix:

- Updates the trellis matrix with these maximum probabilities.
- Maintains the pointers matrix to record the previous state indices that lead to the highest probabilities.

### Viterbi Decoding Process:

- Determines the optimal sequence of part-of-speech tags (POS tags) for a given input sentence.
- Initializes an empty list called `best_path` to store the sequence of tags.
- Identifies the index of the best final state (POS tag) in the last position of the trellis matrix using `np.argmax`.
- Iterates in reverse through the positions (words) in the sentence, retrieving the index of the previous state (POS tag) from the pointers matrix.
- Constructs the `best_path` list, containing the sequence of state indices representing the most likely sequence of POS tags.

### Accuracy Calculation:

- Calculates accuracy using the formula: `accuracy = correctly_predicted_labels / total_labels`.
- Achieves an accuracy of 0.94754 on dev data after replacing unknown words with `<unk>` tags and using a smoothing of 1e-6 for all probabilities.

## Function for Predictions on Test Data

- Implements a function `viterbi_on_test_data` to create predictions for the `test.json` dataset.
- Stores the predicted labels and sentence/index information in the file `viterbi.json`.

