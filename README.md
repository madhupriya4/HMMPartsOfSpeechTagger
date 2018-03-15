# HMM_Pos_Tagging

# Programs

hmmlearn.py will learn a hidden Markov model from the training data, and hmmdecode.py will use the model to tag new data. 

The learning program will be invoked in the following way:

> python hmmlearn.py /path/to/input

The argument is a single file containing the training data; the program will learn a hidden Markov model, and write the model parameters to a file called hmmmodel.txt.

The model file contains sufficient information for hmmdecode.py to successfully tag new data.


The tagging program will be invoked in the following way:

> python hmmdecode.py /path/to/input

The argument is a single file containing the test data; the program will read the parameters of a hidden Markov model from the file hmmmodel.txt, tag each word in the test data, and write the results to a text file called hmmoutput.txt in the same format as the training data.
