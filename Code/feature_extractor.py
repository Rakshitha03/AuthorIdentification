# feature extraction code
# extracting character based lexical features
from __future__ import division
from collections import defaultdict
import re
import nltk
import math


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
            alpha_character_freq["lex_char_" + character.lower()] += 1
        elif character.isdigit():
            digit_count += 1
        elif character.isspace():
            space_count += 1
        elif character in special_characters:
            special_character_freq["lex_special_" + character] += 1
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
    # tokenize into words using nltk word_tokenizer
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
            punctuation_freq["synt_punct_" + word] += 1
        elif word in functional_word_list:
            func_word_freq["synt_funct_word_" + word] += 1
    syntactic_features.update(punctuation_freq)
    syntactic_features.update(func_word_freq)
    return syntactic_features


def extract_structural_features(text):
    structural_features = {}
    line_count = len(filter(None, re.split(r'\n', text)))
    structural_features["struct_line_count"] = line_count
    # A sentence is defined by us as a sequence of words ending with a (.)
    # or (!) or (?) or (:)
    sentence_count = len(filter(None, re.split(r'[:.!?]+', text)))
    structural_features["struct_sentence_count"] = sentence_count
    # A paragraph is one which is delimited by two \n
    paragraph_count = len(filter(None, re.split(r'\n\n', text)))
    structural_features["struct_paragraph_count"] = paragraph_count
    greetings = ["hi", "hello", "dear", "hey", "respected"]
    lowercase_words = [i.lower() for i in text.split()]
    intersection = len(set(greetings) & set(lowercase_words))
    if intersection > 0:
        has_greeting = 0
    else:
        has_greeting = 1
    structural_features["struct_has_greeting"] = has_greeting
    return structural_features


def extract_word_lex_features(text):
    lex_word_features = defaultdict(float)
    word_count = defaultdict(float)
    word_freq_dist = defaultdict(float)
    freq_word_count = defaultdict(float)
    unique_word_set = set()
    words = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    C = len(text)
    no_of_words = len(words)
    no_of_sentences = len(sentences)
    no_of_short_words = 0
    no_of_chars_in_words = 0
    simpsons_diversity_index_numerator = 0
    for word in words:
        word = word.lower()
        word_count[word] += 1
        freq_key = len(word)
        if freq_key < 20:
            word_freq_dist["lex_"+str(freq_key)] += 1
        if len(word) < 4:
            no_of_short_words += 1
        no_of_chars_in_words += len(word)
        unique_word_set.add(word)
    m1 = no_of_words
    m2 = 0
    for word in word_count:
        freq_word_count[word_count[word]] += 1
        simpsons_diversity_index_numerator += word_count[word] * (word_count[word] - 1)
    for freq in freq_word_count:
        m2 += (freq ** 2) * freq_word_count[freq]
    no_of_chars_by_C = no_of_chars_in_words / C
    average_word_length = no_of_chars_in_words / no_of_words
    averag_sentence_length_in_words = no_of_words / no_of_sentences
    averag_sentence_length_in_chars = C / no_of_sentences
    no_of_unique_words = len(unique_word_set)
    hapax_legomena = freq_word_count[1]
    hapax_dislegomena = freq_word_count[2]
    if m2 == m1:
        yules_k_measure = 0
    else:
        yules_k_measure = (m1 * m2) / (m2 - m1)
    try:
        simpsons_diversity_index = 1 - (simpsons_diversity_index_numerator / (no_of_words * (no_of_words - 1)))
    except ZeroDivisionError:
        simpsons_diversity_index = 0
    try:
        sichels_s_measure = freq_word_count[2] / no_of_unique_words
    except ZeroDivisionError:
        sichels_s_measure = 0
    brunets_w_meaure = math.log(no_of_words) * (no_of_unique_words - 0.17)
    try:
        honores_r_measure = 100 * math.log(no_of_words) / (1 - (freq_word_count[1] / no_of_unique_words))
    except ZeroDivisionError:
        honores_r_measure = 0
    lex_word_features["lex_hapax_legomena"] = hapax_legomena
    lex_word_features["lex_hapax_dislegomena"] = hapax_dislegomena
    lex_word_features["lex_yules_k_measure"] = yules_k_measure
    lex_word_features["lex_brunets_w_meaure"] = brunets_w_meaure
    lex_word_features["lex_sichels_s_measure"] = sichels_s_measure
    lex_word_features["lex_honores_r_measure"] = honores_r_measure
    lex_word_features["lex_simpsons_diversity_index"] = simpsons_diversity_index
    lex_word_features["lex_no_of_words"] = no_of_words
    lex_word_features["lex_no_of_short_words"] = no_of_short_words
    lex_word_features["lex_no_of_chars_by_C"] = no_of_chars_by_C
    lex_word_features["lex_average_word_length"] = average_word_length
    lex_word_features["lex_averag_sentence_length_in_chars"] = averag_sentence_length_in_chars
    lex_word_features["lex_averag_sentence_length_in_words"] = averag_sentence_length_in_words
    lex_word_features["lex_no_of_unique_words"] = no_of_unique_words
    lex_word_features.update(word_freq_dist)
    return lex_word_features
