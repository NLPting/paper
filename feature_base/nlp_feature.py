import re
from nltk import sent_tokenize
import math
from pyphen import Pyphen
import string
import spacy
import numpy as np

nlp = spacy.load('en', disable=['ner'])
vocab_level  = np.load('my_level_vocab.npy').item()
connect = [i.replace('\n','') for i in open('connect.txt').readlines()]

def differe_level_feature(org_array ,  w =vocab_level  , delete_key=['A1','A2']):
    feature_map = {'A1':0,'A2':0,'B1':0,'B2':0,'C1':0,'C2':0}
    for word in org_array:
        if word in w:
            if w[word]=='A1':feature_map['A1']+=1
            if w[word]=='A2':feature_map['A2']+=1
            if w[word]=='B1':feature_map['B1']+=1
            if w[word]=='B2':feature_map['B2']+=1
            if w[word]=='C1':feature_map['C1']+=1
            if w[word]=='C2':feature_map['C2']+=1
    F = [feature_map[key] for key in feature_map.keys() if key not in delete_key]
    return F  

def corpus_word_level_score(org_array ,  w = vocab_level ):
    score = 0
    for word in org_array:
        if word in w and w[word]!='A1' and w[word]!='A2':
            if w[word]=='B1':count=1
            if w[word]=='B2':count=2
            if w[word]=='C1':count=3
            if w[word]=='C2':count=4
            score+=count
    try:
        word_level_score = score / len([word for word in org_array if word in w and w[word]!='A1' and w[word]!='A2'])
    except ZeroDivisionError:
        return 0.0
    return word_level_score


def get_sum_heights(paragraph):
    def tree_height(root):
        """
        Find the maximum depth (height) of the dependency parse of a spacy sentence by starting with its root
        Code adapted from https://stackoverflow.com/questions/35920826/how-to-find-height-for-non-binary-tree
        :param root: spacy.tokens.token.Token
        :return: int, maximum height of sentence's dependency parse tree
        """
        if not list(root.children):
            return 1
        else:
            return 1 + max(tree_height(x) for x in root.children)
    """
    Computes average height of parse trees for each sentence in paragraph.
    :param paragraph: spacy doc object or str
    :return: float
    """
    if type(paragraph) == str:
        doc = nlp(paragraph)
    else:
        doc = paragraph
    roots = [sent.root for sent in doc.sents]
    return np.sum([tree_height(root) for root in roots])

def count_connectives(text):
    text = text.split(' ')
    return len([i for i in connect if i in text])

def punct_feature(text):
    punc_count = text.count("?") + text.count("!")
    return punc_count


def remove_punctuation(text):
    return ''.join(ch for ch in text if ch not in string.punctuation)

def char_count(text):
    text = delete_mask_return_sen(text)
    text = remove_punctuation(text)
    text = text.replace(" ", "")
    return len(text)

def legacy_round(number, points=0):
    p = 10 ** points
    return float(math.floor((number * p) + math.copysign(0.5, number))) / p

def lexicon_count(text):
    text = remove_punctuation(text).strip()
    count = len(text.split())
    return count

def sentence_count(text):
    sen_count = len(sent_tokenize(text))
    return sen_count

def delete_mask_return_sen(text):
    return re.sub(r'@+\w+\s?', '', text)

def avg_sentence_length(text):
    try:
        asl = float(lexicon_count(text) / sentence_count(text))
        return legacy_round(asl, 1)
    except ZeroDivisionError:
        return 0.0
    
def avg_word_length(text):
    try:
        asl = float(char_count(text) / lexicon_count(text))
        return legacy_round(asl, 1)
    except ZeroDivisionError:
        return 0.0

    
def syllable_count(text, lang='en_US'):
    """
    Function to calculate syllable words in a text.
    I/P - a text
    O/P - number of syllable words
    """
    text = text.lower()
    text = delete_mask_return_sen(text)
    text = remove_punctuation(text).strip()
    if not text:
        return 0
    dic = Pyphen(lang=lang)
    count = 0
    for word in text.split(' '):
        if word:
            word_hyphenated = dic.inserted(word)
            count += max(1, word_hyphenated.count("-") + 1)
    return count

def avg_syllables_per_word(text):
    syllable = syllable_count(text)
    words = lexicon_count(text)
    try:
        syllables_per_word = float(syllable) / float(words)
        return legacy_round(syllables_per_word, 1)
    except ZeroDivisionError:
        return 0.0
    
def polysyllabcount(text):
    count = 0
    for word in text.split():
        wrds = syllable_count(word)
        if wrds >= 3:
            count += 1
    return count
    
    
def flesch_reading_ease(text):
    sentence_length = avg_sentence_length(text)
    syllables_per_word = avg_syllables_per_word(text)
    flesch = (
        206.835
        - float(1.015 * sentence_length)
        - float(84.6 * syllables_per_word)
    )
    return legacy_round(flesch, 2)


def smog_index(text):
    sentences = sentence_count(text)
    if sentences >= 20:
        try:
            poly_syllab = polysyllabcount(text)
            smog = (
                (1.043 * (30 * (poly_syllab / sentences)) ** .5)
                + 3.1291)
            return legacy_round(smog, 1)
        except ZeroDivisionError:
            return 0.0
    else:
        return 0.0
def automated_readability_index(text):
    chrs = char_count(text)
    words = lexicon_count(text)
    sentences = sentence_count(text)
    try:
        a = float(chrs)/float(words)
        b = float(words) / float(sentences)
        readability = (
            (4.71 * legacy_round(a, 2))
            + (0.5 * legacy_round(b, 2))
            - 21.43)
        return legacy_round(readability, 1)
    except ZeroDivisionError:
        return 0.0
