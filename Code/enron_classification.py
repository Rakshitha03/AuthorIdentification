from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import svm
from sklearn.cross_validation import cross_val_score
import numpy as np
import os
import feature_extractor
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# feature list for lexical, syntactic and structural features.
lexical_char_features = ["lex_char_count", "lex_alpha_distribution",
                         "lex_upper_case_distribution", "lex_digit_distribution",
                         "lex_space_distribution", "lex_char_a", "lex_char_b",
                         "lex_char_c", "lex_char_d", "lex_char_e", "lex_char_f",
                         "lex_char_g", "lex_char_h", "lex_char_i", "lex_char_j",
                         "lex_char_k", "lex_char_l", "lex_char_m", "lex_char_n",
                         "lex_char_o", "lex_char_p", "lex_char_q", "lex_char_r",
                         "lex_char_s", "lex_char_t", "lex_char_u", "lex_char_v",
                         "lex_char_w", "lex_char_x", "lex_char_y", "lex_char_z",
                         "lex_special_~", "lex_special_@", 'lex_special_#',
                         'lex_special_$', 'lex_special_%', 'lex_special_^',
                         'lex_special_&', 'lex_special_*', 'lex_special_-',
                         'lex_special__', 'lex_special_=', 'lex_special_+',
                         'lex_special_>', 'lex_special_<', 'lex_special_[',
                         'lex_special_]', 'lex_special_{', 'lex_special_}',
                         'lex_special_/', 'lex_special_\\', 'lex_special_|']
lexical_word_features = ["lex_hapax_legomena", "lex_hapax_dislegomena", "lex_yules_k_measure", "lex_brunets_w_meaure",
                         "lex_sichels_s_measure", "lex_honores_r_measure", "lex_simpsons_diversity_index",
                         "lex_no_of_words", "lex_no_of_short_words", "lex_no_of_chars_by_C", "lex_average_word_length",
                         "lex_averag_sentence_length_in_chars", "lex_averag_sentence_length_in_words",
                         "lex_no_of_unique_words", "lex_1", "lex_2", "lex_3", "lex_4", "lex_5", "lex_6", "lex_7",
                         "lex_9", "lex_10", "lex_11", "lex_12", "lex_13", "lex_14", "lex_15", "lex_16", "lex_17",
                         "lex_18", "lex_19", "lex_20"]
syntactic_features = ["synt_punct_\"", "synt_punct_.", "synt_punct_?",
                      "synt_punct_!", "synt_punct_:", "synt_punct_;",
                      "synt_punct_\'", 'synt_funct_word_a',
                      'synt_funct_word_between', 'synt_funct_word_in',
                      'synt_funct_word_nor', 'synt_funct_word_some',
                      'synt_funct_word_upon',
                      'synt_funct_word_about', 'synt_funct_word_both',
                      'synt_funct_word_including', 'synt_funct_word_nothing',
                      'synt_funct_word_somebody',
                      'synt_funct_word_us', 'synt_funct_word_above',
                      'synt_funct_word_but', 'synt_funct_word_inside',
                      'synt_funct_word_of',
                      'synt_funct_word_someone',
                      'synt_funct_word_used', 'synt_funct_word_after',
                      'synt_funct_word_by', 'synt_funct_word_into',
                      'synt_funct_word_off',
                      'synt_funct_word_something',
                      'synt_funct_word_via', 'synt_funct_word_all',
                      'synt_funct_word_can', 'synt_funct_word_is',
                      'synt_funct_word_on', 'synt_funct_word_such',
                      'synt_funct_word_we',
                      'synt_funct_word_although', 'synt_funct_word_cos',
                      'synt_funct_word_it', 'synt_funct_word_once',
                      'synt_funct_word_than',
                      'synt_funct_word_what',
                      'synt_funct_word_am', 'synt_funct_word_do',
                      'synt_funct_word_its', 'synt_funct_word_one',
                      'synt_funct_word_that',
                      'synt_funct_word_whatever',
                      'synt_funct_word_among', 'synt_funct_word_down',
                      'synt_funct_word_latter', 'synt_funct_word_onto',
                      'synt_funct_word_the',
                      'synt_funct_word_when',
                      'synt_funct_word_an', 'synt_funct_word_each',
                      'synt_funct_word_less', 'synt_funct_word_opposite',
                      'synt_funct_word_their',
                      'synt_funct_word_where',
                      'synt_funct_word_and', 'synt_funct_word_either',
                      'synt_funct_word_like', 'synt_funct_word_or',
                      'synt_funct_word_them',
                      'synt_funct_word_whether',
                      'synt_funct_word_another', 'synt_funct_word_enough',
                      'synt_funct_word_little', 'synt_funct_word_our',
                      'synt_funct_word_these',
                      'synt_funct_word_which', 'synt_funct_word_any',
                      'synt_funct_word_every', 'synt_funct_word_lots',
                      'synt_funct_word_outside',
                      'synt_funct_word_they',
                      'synt_funct_word_while', 'synt_funct_word_anybody',
                      'synt_funct_word_everybody', 'synt_funct_word_many',
                      'synt_funct_word_over',
                      'synt_funct_word_this', 'synt_funct_word_who',
                      'synt_funct_word_anyone', 'synt_funct_word_everyone',
                      'synt_funct_word_me',
                      'synt_funct_word_own',
                      'synt_funct_word_those', 'synt_funct_word_whoever',
                      'synt_funct_word_anything',
                      'synt_funct_word_everything',
                      'synt_funct_word_more', 'synt_funct_word_past',
                      'synt_funct_word_though', 'synt_funct_word_whom',
                      'synt_funct_word_are',
                      'synt_funct_word_few',
                      'synt_funct_word_most', 'synt_funct_word_per',
                      'synt_funct_word_through', 'synt_funct_word_whose',
                      'synt_funct_word_around',
                      'synt_funct_word_following', 'synt_funct_word_much',
                      'synt_funct_word_plenty', 'synt_funct_word_till',
                      'synt_funct_word_will',
                      'synt_funct_word_as',
                      'synt_funct_word_for', 'synt_funct_word_must',
                      'synt_funct_word_plus', 'synt_funct_word_to',
                      'synt_funct_word_with', 'synt_funct_word_at',
                      'synt_funct_word_from',
                      'synt_funct_word_my', 'synt_funct_word_regarding',
                      'synt_funct_word_toward', 'synt_funct_word_within',
                      'synt_funct_word_be', 'synt_funct_word_have',
                      'synt_funct_word_near', 'synt_funct_word_same',
                      'synt_funct_word_towards', 'synt_funct_word_without',
                      'synt_funct_word_because',
                      'synt_funct_word_he', 'synt_funct_word_need',
                      'synt_funct_word_several', 'synt_funct_word_under',
                      'synt_funct_word_worth', 'synt_funct_word_before',
                      'synt_funct_word_her', 'synt_funct_word_neither',
                      'synt_funct_word_she', 'synt_funct_word_unless',
                      'synt_funct_word_would',
                      'synt_funct_word_behind', 'synt_funct_word_him',
                      'synt_funct_word_no', 'synt_funct_word_should',
                      'synt_funct_word_unlike', 'synt_funct_word_yes',
                      'synt_funct_word_below', 'synt_funct_word_i',
                      'synt_funct_word_nobody', 'synt_funct_word_since',
                      'synt_funct_word_until', 'synt_funct_word_you',
                      'synt_funct_word_beside', 'synt_funct_word_if',
                      'synt_funct_word_none', 'synt_funct_word_so',
                      'synt_funct_word_up', 'synt_funct_word_your']
structural_features = ["struct_num_sentences", "struct_num_paragraphs", "struct_num_sentences_per_paragraph",
                       "struct_num_chars_per_paragraph", "struct_num_words_per_paragraph", "struct_has_greeting",
                       "struct_has_quote", "struct_paragraph_has_indentation", "struct_has_url_signature",
                       "struct_has_email_signature", "struct_has_name_signature"]
# The dataset path.
main_path = '/Users/rakshitha/AuthorIdentification/relevant_email_data'
''' i) We first perform a binary classification by taking the top 2 authors who have sent out the maximum number of emails.
    We compare the classification accuracy obtained by Naive Bayes and SVM classifiers. The size of the training data is
    set to 100 emails and the all the features are used for classification.
    ii) We next vary the feature parameter i.e compare the accuracies obtained by just using lexical features,
    lexical + syntactic features, lexical + syntactic + structural features. All the other parameters remain the
    same as mentioned above.
    iii) Next we change the size of the training set i.e increase the size from 25 emails, 50 emails, 75 emails to
    100 emails and compare the accuracies obtained. Other parameters remain the same as mentioned in (i)
    iv) Lastly to compare the effect on accuracies on increase in the number of authors, we increase the size of
    authors from 2 authors, 5 authors, 10 authors thereby performing a 2-way, 5-way, and 10-way
    classification respectively. The top n( where n being 2,4,8,10) who sent maximum mails are taken into
    consideration.
    '''

authorsList = [['mann-k', 'kaminski-v'], ['mann-k', 'kaminski-v', 'symes-k', 'germany-c', 'bass-e'], ['mann-k', 'kaminski-v', 'symes-k', 'germany-c', 'bass-e', 'scott-s', 'rogers-b', 'beck-s', 'arnold-j',
                'rodrique-r']]
feature_selection = [['char', 'word'], ['char', 'word', 'syntactic'], ['char', 'word', 'structural', 'syntactic']]
features = ['F1', 'F1+F2', 'F1+F2+F3']
training_sizes = [25, 50, 75, 100]

# Extracting features for training the data
training_emails = []
naive_bayes_mean = []
svm_mean = []

''' Execution of the (i)'''
print "Accuracy comparison of models"
author_names = authorsList[0]
count = 0
training_vectors = []
training_class_values = []
# Keeps a count of the files with no content or the mails which throw an exception.
bad_files = 0
feature_names = []
for name in author_names:
    count = 0
    mail_path = main_path + '/' + name
    for email in os.listdir(mail_path):
        # filter out only emails which have content. not the ones with metadata.
        if not email.startswith("content"):
            continue
        if count >= training_sizes[-1]:
            break
        combined_features = []
        fp = open(mail_path + '/' + email)
        content = fp.read()
        try:
            l_char_features = feature_extractor.extract_char_lex_feature(
                content)
            l_word_features = feature_extractor.extract_word_lex_features(
                content)
            syn_features = feature_extractor.extract_syntactic_features(
                content)
            struc_features = feature_extractor.extract_structural_features(
                content)
        except:
            bad_files += 1
            continue
        # We consider emails which have at least 50 to 100 words in them
        if l_word_features["lex_no_of_words"] not in range(50, 100):
            continue
        for char_feature in lexical_char_features:
            combined_features.append(l_char_features[char_feature])
            feature_names.append(char_feature)
        for word_feature in lexical_word_features:
            combined_features.append(l_word_features[word_feature])
            feature_names.append(word_feature)
        for syn_feature in syntactic_features:
            combined_features.append(syn_features[syn_feature])
            feature_names.append(syn_feature)
        for struct_feature in structural_features:
            combined_features.append(struc_features[struct_feature])
            feature_names.append(struct_feature)
        training_vectors.append(combined_features)
        training_class_values.append(name)
        count += 1
        # training_emails.append(email)
# Classification using sci-kit. 10-fold cross validation
X = np.array(training_vectors)
Y = np.array(training_class_values)
naive_bayes_model = MultinomialNB()
print "Naive bayes accuracy:"
score_1 = cross_val_score(naive_bayes_model, X, Y, cv=10, scoring="accuracy")
print score_1.mean()
svm_linear_kernel_model = svm.SVC(kernel='linear')
print "SVM accuracy: "
score_2 = cross_val_score(svm_linear_kernel_model, X, Y, cv=10, scoring="accuracy")
print score_2.mean()


'''Execution of part (ii)'''
print
print "*"*20
print "Accuracy comparison across features"
for feat in feature_selection:
    author_names = authorsList[0]
    training_vectors = []
    training_class_values = []
    bad_files = 0
    feature_names = []
    for name in author_names:
        count = 0
        mail_path = main_path + '/' + name
        for email in os.listdir(mail_path):
            if not email.startswith("content"):
                continue
            if count >= training_sizes[-1]:
                break
            combined_features = []
            fp = open(mail_path + '/' + email)
            content = fp.read()
            try:
                l_char_features = feature_extractor.extract_char_lex_feature(
                    content)
                l_word_features = feature_extractor.extract_word_lex_features(
                    content)
                syn_features = feature_extractor.extract_syntactic_features(
                    content)
                struc_features = feature_extractor.extract_structural_features(
                    content)
            except:
                bad_files += 1
                continue
            if l_word_features["lex_no_of_words"] not in range(50, 100):
                continue
            if "char" in feat:
                for char_feature in lexical_char_features:
                    combined_features.append(l_char_features[char_feature])
                    feature_names.append(char_feature)
            if "word" in feat:
                for word_feature in lexical_word_features:
                    combined_features.append(l_word_features[word_feature])
                    feature_names.append(word_feature)
            if "syntactic" in feat:
                for syn_feature in syntactic_features:
                    combined_features.append(syn_features[syn_feature])
                    feature_names.append(syn_feature)
            if "structural" in feat:
                for struct_feature in structural_features:
                    combined_features.append(struc_features[struct_feature])
                    feature_names.append(struct_feature)
            training_vectors.append(combined_features)
            training_class_values.append(name)
            count += 1

    X = np.array(training_vectors)
    Y = np.array(training_class_values)
    naive_bayes_model = MultinomialNB()
    print "naive bayes accuracy"
    score_1 = cross_val_score(naive_bayes_model, X, Y, cv=10, scoring="accuracy")
    print score_1.mean()
    naive_bayes_mean.append((score_1.mean() * 100))
    svm_linear_kernel_model = svm.SVC(kernel='linear')
    print "SVM accuracy"
    score_2 = cross_val_score(svm_linear_kernel_model, X, Y, cv=10, scoring="accuracy")
    print score_2.mean()
    svm_mean.append((score_2.mean() * 100))
# Plotting the graph for the variation
values = range(len(features))
# plt.plot(values, naive_bayes_mean, 'r-', linewidth=5, label='Naive Bayes')
# plt.xlabel("Features")
# plt.ylabel("Accuracy")
# plt.plot(values, svm_mean, 'y-', linewidth=5, label='SVM')
# plt.grid(True)
# plt.xticks(values, features)
# plt.legend(loc='upper left')
# plt.show()


''' Execution of part (iii) '''
print
print "*"*20
print "Accuracy comparison across training sizes"
for trainSize in training_sizes:
    author_names = authorsList[0]
    training_vectors = []
    training_class_values = []
    bad_files = 0
    feature_names = []
    for name in author_names:
        count = 0
        mail_path = main_path + '/' + name
        for email in os.listdir(mail_path):
            if not email.startswith("content"):
                continue
            if count >= trainSize:
                break
            combined_features = []
            fp = open(mail_path + '/' + email)
            content = fp.read()
            try:
                l_char_features = feature_extractor.extract_char_lex_feature(
                    content)
                l_word_features = feature_extractor.extract_word_lex_features(
                    content)
                syn_features = feature_extractor.extract_syntactic_features(
                    content)
                struc_features = feature_extractor.extract_structural_features(
                    content)
            except:
                bad_files += 1
                continue
            if l_word_features["lex_no_of_words"] not in range(50, 100):
                continue
            for char_feature in lexical_char_features:
                combined_features.append(l_char_features[char_feature])
                feature_names.append(char_feature)
            for word_feature in lexical_word_features:
                combined_features.append(l_word_features[word_feature])
                feature_names.append(word_feature)
            for syn_feature in syntactic_features:
                combined_features.append(syn_features[syn_feature])
                feature_names.append(syn_feature)
            for struct_feature in structural_features:
                combined_features.append(struc_features[struct_feature])
                feature_names.append(struct_feature)
            training_vectors.append(combined_features)
            training_class_values.append(name)
            count += 1

    X = np.array(training_vectors)
    Y = np.array(training_class_values)
    naive_bayes_model = MultinomialNB()
    print "naive bayes accuracy"
    score_1 = cross_val_score(naive_bayes_model, X, Y, cv=10, scoring="accuracy")
    print score_1.mean()
    naive_bayes_mean.append((score_1.mean() * 100))
    svm_linear_kernel_model = svm.SVC(kernel='linear')
    print "SVM accuracy"
    score_2 = cross_val_score(svm_linear_kernel_model, X, Y, cv=10, scoring="accuracy")
    print score_2.mean()
    svm_mean.append((score_2.mean() * 100))
# Plotting the graph for the variation
# plt.plot(training_sizes, naive_bayes_mean, 'r-', linewidth=5, label='Naive Bayes')
# plt.xlabel("training sizes")
# plt.ylabel("Accuracy")
# plt.plot(training_sizes, svm_mean, 'y-', linewidth=5, label='SVM')
# plt.grid(True)
# plt.legend(loc='upper left')
# plt.show()

''' Execution of part (iv) '''
print
print "*"*20
print "Accuracy comparison across different number of authors"
for authSet in authorsList:
    author_names = authSet
    training_vectors = []
    training_class_values = []
    bad_files = 0
    feature_names = []
    for name in author_names:
        count = 0
        mail_path = main_path + '/' + name
        for email in os.listdir(mail_path):
            if not email.startswith("content"):
                continue
            if count >= training_sizes[-1]:
                break
            combined_features = []
            fp = open(mail_path + '/' + email)
            content = fp.read()
            try:
                l_char_features = feature_extractor.extract_char_lex_feature(
                    content)
                l_word_features = feature_extractor.extract_word_lex_features(
                    content)
                syn_features = feature_extractor.extract_syntactic_features(
                    content)
                struc_features = feature_extractor.extract_structural_features(
                    content)
            except:
                bad_files += 1
                continue
            if l_word_features["lex_no_of_words"] not in range(50, 100):
                continue
            for char_feature in lexical_char_features:
                combined_features.append(l_char_features[char_feature])
                feature_names.append(char_feature)
            for word_feature in lexical_word_features:
                combined_features.append(l_word_features[word_feature])
                feature_names.append(word_feature)
            for syn_feature in syntactic_features:
                combined_features.append(syn_features[syn_feature])
                feature_names.append(syn_feature)
            for struct_feature in structural_features:
                combined_features.append(struc_features[struct_feature])
                feature_names.append(struct_feature)
            training_vectors.append(combined_features)
            training_class_values.append(name)
            count += 1

    X = np.array(training_vectors)
    Y = np.array(training_class_values)
    naive_bayes_model = MultinomialNB()
    print "naive bayes accuracy"
    score_1 = cross_val_score(naive_bayes_model, X, Y, cv=10, scoring="accuracy")
    print score_1.mean()
    naive_bayes_mean.append((score_1.mean() * 100))
    svm_linear_kernel_model = svm.SVC(kernel='linear')
    print "SVM accuracy"
    score_2 = cross_val_score(svm_linear_kernel_model, X, Y, cv=10, scoring="accuracy")
    print score_2.mean()
    svm_mean.append((score_2.mean() * 100))
# Plotting the graph for the variation
# values = [len(i) for i in authorsList]
# plt.plot(values, naive_bayes_mean, 'r-', linewidth=5, label='Naive Bayes')
# plt.xlabel("Number of authors")
# plt.ylabel("Accuracy")
# plt.plot(values, svm_mean, 'y-', linewidth=5, label='SVM')
# plt.grid(True)
# plt.legend(loc='upper left')
# plt.show()

