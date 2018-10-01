#importing libraries
import nltk
import re
from pywsd.lesk import simple_lesk

#Downloading the The stopwords and populars
nltk.download('popular')

#Taking the user input
sent = input("Enter the sentence")

#Tokenizing the inut into word
sent2 = nltk.word_tokenize(sent)

#Tagging parts of speech
tagged_word = nltk.pos_tag(sent2)
print(tagged_word)
#Ask your to which word sense the want to know and collecting all that word with there POS in a list
ambiguous = input('Enter the word want to disambiguate:')
sense_word_list = []
for i in range(len(tagged_word)):
    if tagged_word[i][0].lower() == ambiguous:
        synset=simple_lesk(sent,tagged_word[i][0],tagged_word[i][1][0].lower())
        sense = synset.lemmas()[0].name()
        sense = re.sub(r"_"," ",sense)
        print("Sense of",tagged_word[i][0],"is: ",sense)
        print("Definition of",tagged_word[i][0],"is:",synset.definition())
