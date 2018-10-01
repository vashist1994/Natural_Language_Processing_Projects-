from autocorrect import spell
import nltk
import re

sentence = input("Enter the input: ").lower()
sentence = re.sub(r'(?<!\d)\.(?!\d)', '', sentence)

word = nltk.word_tokenize(sentence)

#Check the spelling mistake in the user input
for i in range(len(word)):
    word[i]=spell(word[i])

spellchecked_sentence = ' '.join(word)

#Preprocessing the IT dataset
with open('IT_Domain_words.txt','r') as bg:
    content = bg.read()
    word_list =[word.lower() for word in content.splitlines()]
word2 = ' '.join(word_list)
it_words=word2.split()

# Replacing with the It Domain Word
correct_word = {}
mistake = "wife"
for words in it_words:
    ed = nltk.edit_distance(mistake, words)
    correct_word[words] = ed
    
predict_word=min(correct_word, key=correct_word.get)
final_sentence = [w.replace(mistake,predict_word) for w in word]
print('correct sentence: ',' '.join(final_sentence))
    




