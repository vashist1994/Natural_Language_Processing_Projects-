# Natural_Language_Processing_Projects-1
1: Domain Specific Spell-Checker
For this project I have used the an library call autocorrect for in general spell correction of terms other than the Domain for which your are building it , after that I used NLTK library of python along with a random expression library for the processing of the user input.
To make it Domain specific I have created a dataset of all the verious terms used in IT as I am building it for IT Domain But you can use it for any Domain you want
Then I have used the concept of Edit distance and calculate the edit distance of the words user have entered with the term i have in my dataset and the term which is having the less edit diatance value Have replaced it with that IT Domain term.

2: Word Sense Disambiguity
Word sense Disambiguity is simply extracting the correct sense of some words based on the context as there are some words that have different meaning when they are used in different context. 
So for this I have used following dependencies:
1: NLTK
2:Random Expression (re) : For preprocessing of user input
3:pywsd library : for Lesk algorithm
The program is like, user enter the input sentence and after that which word they want to disambiguate then the user enterd words meaning or sense based on the context is given as the output

3: Custom Named Entity Recognizer
Following are the Dependensies
1: Plac
2: spacy
3: path
4: tqdm
This project is basically about how you can train the Named Entity Recognizer of the Spacy which is a library for Natural language processing. I have trained it using only small dataset but you can trained it with your own dataset to improve its acurracy
the programing is quite simple if you know spacy.
