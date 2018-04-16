# NER_system

This repository includes the implementation of a named entity recognition system for Diseases and Treatments.
The ner.txt file includes the training data , each line consists of a token and a label(D,T,O) corresponding to it.
The crf_nlu.py file consists of the CRF based NER tagger for the give data and this script will automatically divide the data into train and test set and will output a output.txt file for test data and its labels in same format as that of the training file.
The bigru_nlu.py is implementation of a deep sequence tagger based on Bidirectional Gated Recurrent unit.
The NER_system_report is a report file which includes all the implementation details and the results for this NER task.
