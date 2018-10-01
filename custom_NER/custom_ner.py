# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 21:48:52 2018

@author: vashist
"""
#importing libraries
#from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import spacy
from tqdm import tqdm

#Training data
TRAIN_DATA = [
    ('can i get a loan on a property in amsterdam?', {
        'entities': [(11, 15, 'Loan'),(21,29, 'Property'),(33,42,'Overseas')]
    }),
    ('overseas property loan', {
        'entities': [(0, 8, 'Overseas'), (8, 16, 'Property'),(17,21,'Loan')]
    }),
    ('will you lend for property in israel?', {
        'entities': [(8, 12, 'Loan'), (17, 25, 'Property'),(29,35,'Oversease')]
    }),
    ('hi, do you guys grant home loans for property in brisbane, queensland?', {
        'entities': [(21, 25, 'Home'), (26, 31, 'Loan'),(36,44,'Property'),(48,56,'Overseas'),(58,68,'Overseas')]
        
    }),
    ('overseas loan for australia property', {
        'entities': [(0, 8, 'Overseas'),(8,12, 'Loan'),(17,26,'Overseas'),(27,35,'Property')]
    }),
    (' just want to know the house loan package', {
        'entities': [(27, 31, 'Loan'),(21,26, 'Home')]
    }),
    ('home loans?', {
        'entities': [(4, 8, 'Loan'),(0,4, 'Home')]
    }),
    ('do u want to get a home loan with me', {
        'entities': [(11, 15, 'Loan'),(18,22, 'Home')]
    }),
    ('can i use housing loan to pay for bsd and absd ?', {
        'entities': [(17, 21, 'Loan'),(9, 16, 'Home')]
    }),
    ('i need a home loan', {
        'entities': [(13, 17, 'Loan'),(8, 12, 'Home')]
    }),
     ("hello emma! can i find out what's the maximum home loan i can borrow?", {
        'entities': [(50, 54, 'Loan'),(45, 49, 'Home'),(33,42,'Overseas')]
    }),
    ('tenor home loans', {
        'entities': [(10, 15, 'Loan'),(5, 9, 'Home')]
    }),
    ('hey are pursuing a general program of renovation to the entire property',{
        'entities': [(37, 47, 'Renovation'),(62,70, 'Property')]
    }),
    ('a major overhaul of the healthcare system was proposed',{
      'entities':[(7, 15, 'Renovation')]
     })
 ]
#difining the variables
model=None 
output_dir="C:\\cognicor coding question\\custom_NER" 
n_iter=100

"""Load the model, set up the pipeline and train the entity recognizer."""
if model is not None:
    nlp = spacy.load(model)
    print("Loaded model '%s'" % model)
else:
    nlp = spacy.blank('en')  # create blank Language class
    print("Created blank 'en' model")

# create the built-in pipeline components and add them to the pipeline
# nlp.create_pipe works for built-ins that are registered with spaCy
if 'ner' not in nlp.pipe_names:
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
else:
    er = nlp.get_pipe('ner')
# add labels
for _, annotations in TRAIN_DATA:
    for ent in annotations.get('entities'):
        ner.add_label(ent[2])

# disable other pipes during the training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):  # only train NER
    optimizer = nlp.begin_training()
    for itn in range(n_iter):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in tqdm(TRAIN_DATA):
            nlp.update(
                [text],  
                [annotations],  
                drop=0.5, 
                sgd=optimizer,
                losses=losses)
            print(losses)

# test the trained model
for text, _ in TRAIN_DATA:
    doc = nlp(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])

# save model to output directory
if output_dir is not None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

# test the saved model
print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
for text, _ in TRAIN_DATA:
    doc = nlp2(text)
    print('Entities', [(ent.text, ent.label_) for ent in doc.ents])
    print('Tokens', [(t.text, t.ent_type_, t.ent_iob) for t in doc])
