# feature extraction code
# extracting character based lexical features
from __future__ import division
from collections import defaultdict
import re
import nltk



def extract_char_lex_feature(text):
    lex_char_features = defaultdict(float)
    no_of_chars = len(text)
    lex_char_features["lex_char_count"] = no_of_chars
    alpha_count = 0
    upper_case_count = 0
    digit_count = 0
    space_count = 0
    special_characters = ['~', '@', '#', '$', '%', '^', '&', '*', '-', '_', '=',
                          '+', '>', '<', '[', ']', '{', '}', '/', '\\', '|']
    special_character_freq = defaultdict(int)
    alpha_character_freq = defaultdict(int)

    for character in text:
        if character.isalpha():
            alpha_count += 1
            if character.isupper():
                upper_case_count += 1
            alpha_character_freq["lex_char_"+character.lower()] += 1
        elif character.isdigit():
            digit_count += 1
        elif character.isspace():
            space_count += 1
        elif character in special_characters:
            special_character_freq["lex_special_"+character] += 1
    alpha_ratio = alpha_count / no_of_chars
    upper_case_ratio = upper_case_count / no_of_chars
    digit_ratio = digit_count / no_of_chars
    space_ratio = space_count / no_of_chars
    lex_char_features["lex_alpha_distribution"] = alpha_ratio
    lex_char_features["lex_upper_case_distribution"] = upper_case_ratio
    lex_char_features["lex_digit_distribution"] = digit_ratio
    lex_char_features["lex_space_distribution"] = space_ratio
    lex_char_features.update(special_character_freq)
    lex_char_features.update(alpha_character_freq)
    return lex_char_features



def extract_syntactic_features(text):
    syntactic_features = defaultdict(float)
    #tokenize into words using nltk word_tokenizer
    tokens = nltk.word_tokenize(text)
    punctuation_list = ["\"", ".", "?", "!", ":", ";", "\'"]
    punctuation_freq = defaultdict(int)
    functional_word_list = ['a', 'between', 'in', 'nor', 'some', 'upon',
                            'about', 'both', 'including', 'nothing', 'somebody',
                            'us', 'above', 'but', 'inside', 'of', 'someone',
                            'used', 'after', 'by', 'into', 'off', 'something',
                            'via', 'all', 'can', 'is', 'on', 'such', 'we',
                            'although', 'cos', 'it', 'once', 'than', 'what',
                            'am', 'do', 'its', 'one', 'that', 'whatever',
                            'among', 'down', 'latter', 'onto', 'the', 'when',
                            'an', 'each', 'less', 'opposite', 'their', 'where',
                            'and', 'either', 'like', 'or', 'them', 'whether',
                            'another', 'enough', 'little', 'our', 'these',
                            'which', 'any', 'every', 'lots', 'outside', 'they',
                            'while', 'anybody', 'everybody', 'many', 'over',
                            'this', 'who', 'anyone', 'everyone', 'me', 'own',
                            'those', 'whoever', 'anything', 'everything',
                            'more', 'past', 'though', 'whom', 'are', 'few',
                            'most', 'per', 'through', 'whose', 'around',
                            'following', 'much', 'plenty', 'till', 'will', 'as',
                            'for', 'must', 'plus', 'to', 'with', 'at', 'from',
                            'my', 'regarding', 'toward', 'within', 'be', 'have',
                            'near', 'same', 'towards', 'without', 'because',
                            'he', 'need', 'several', 'under', 'worth', 'before',
                            'her', 'neither', 'she', 'unless', 'would',
                            'behind', 'him', 'no', 'should', 'unlike', 'yes',
                            'below', 'i', 'nobody', 'since', 'until', 'you',
                            'beside', 'if', 'none', 'so', 'up', 'your']
    func_word_freq = defaultdict(int)
    for word in tokens:
        if word in punctuation_list:
            punctuation_freq["synt_punct_"+word] += 1
        elif word in functional_word_list:
            func_word_freq["synt_funct_word_"+word] += 1
    syntactic_features.update(punctuation_freq)
    syntactic_features.update(func_word_freq)
    return syntactic_features

def extract_structural_features(text):
    structural_features = {}
    line_count = len(filter(None, re.split(r'\n', text)))
    structural_features["struct_line_count"] = line_count
    # A sentence is defined by us as a sequence of words ending with a (.)
    # or (!) or (?) or (:)
    sentence_count = len(filter(None,re.split(r'[:.!?]+', text)))
    structural_features["struct_sentence_count"] = sentence_count
    # A paragraph is one which is delimited by two \n
    paragraph_count = len(filter(None,re.split(r'\n\n',text)))
    structural_features["struct_paragraph_count"] = paragraph_count
    greetings = ["hi", "hello", "dear", "hey", "respected"]
    lowercase_words = [i.lower() for i in text.split()]
    intersection = len(set(greetings) & set(lowercase_words))
    if intersection > 0:
        has_greeting = True
    else:
        has_greeting = False
    structural_features["struct_has_greeting"] = has_greeting
    return structural_features

