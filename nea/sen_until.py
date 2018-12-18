import nltk
import re
import numpy as np



def sen_split(data):
    sen = []
    for item in data:
        sentences = [sent.strip() for sent in nltk.sent_tokenize(item)]
        sen.append(sentences)
    return sen

def tokenize(string):
	tokens = nltk.word_tokenize(string)
	for index, token in enumerate(tokens):
		if token == '@' and (index+1) < len(tokens):
			tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
			tokens.pop(index)
	return tokens

def is_number(token):
	return bool(num_regex.match(token))
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')

def sentence_level(data_sen , vocab , max_sen , max_sen_lg):
    all_data = []
    for corpus in data_sen[:]:
        tmp_corpus = []
        for index , sen in enumerate(corpus):
            tmp_sen = []
            for token in tokenize(sen):
                if token in vocab:
                    tmp_sen.append(vocab[token])
                elif is_number(token):
                    tmp_sen.append(vocab['<num>'])
                else: tmp_sen.append(vocab['<unk>'])
            if len(tmp_sen)<max_sen_lg:
                for i in range(max_sen_lg-len(tmp_sen)):
                    tmp_sen.append(0)
            tmp_corpus.append(tmp_sen)
        if len(tmp_corpus)<max_sen:
            for i in range(max_sen-len(tmp_corpus)):
                tmp_corpus.append(np.zeros(max_sen_lg))
        all_data.append(tmp_corpus)
    return all_data