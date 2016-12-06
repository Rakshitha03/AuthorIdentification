from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import svm
from sklearn.cross_validation import cross_val_score
import numpy as np
import os
import feature_extractor
import json
from pprint import pprint

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
main_path = '/Users/rakshitha/AuthorIdentification/reddit_Data.json'
author_names = ['nittanylionstorm07', 'HardKnockRiffe', 'OnthefarWind','AJinxyCat','740Buckeye','GoldandBlue','Mufro']#, 'symes-k','germany-c', 'bass-e', 'scott-s','rogers-b','beck-s','arnold-j','rodrique-r']
with open(main_path) as data_file:
    data = json.load(data_file)
# pprint(data)
training_vectors = []
test_vectors = []
test_class_values = []
training_class_values = []
bad_files = 0
# Extracting features for training the data
training_emails = []
counts = [25,50,75,100]
for c_max in counts:
    for name in author_names:
        print name
        count = 0
    #     mail_path = main_path + '/' + name
        for comment in data[name]["comments"]:
            # print comment
            content = comment["body"]
    #         if not email.startswith("content"):
    #             continue
            if count >= c_max:
                break
            combined_features = []
    #         fp = open(mail_path + '/' + email)
    #         content = fp.read()
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
            # if l_word_features["lex_no_of_words"] not in range(50, 100):
            #     continue
    #         # print "word count", l_word_features["lex_no_of_words"]
    #         # print email
            for char_feature in lexical_char_features:
                combined_features.append(l_char_features[char_feature])
            for word_feature in lexical_word_features:
                combined_features.append(l_word_features[word_feature])
            for syn_feature in syntactic_features:
                combined_features.append(syn_features[syn_feature])
            for struct_feature in structural_features:
                combined_features.append(struc_features[struct_feature])
            training_vectors.append(combined_features)
            training_class_values.append(name)
            count += 1
#         # training_emails.append(email)
print bad_files
#
# # Extracting the features for test_set
# # for name in author_names:
# #     count = 0
# #     mail_path = main_path + '/' + name
# #     emails =  os.listdir(mail_path)
# #     random.shuffle(emails)
# #     for email in emails:
# #         if not email.startswith("content"):# and email in training_emails:
# #             continue
# #         if count >= 10:
# #             break
# #         combined_features = []
# #         fp = open(mail_path + '/' + email)
# #         content = fp.read()
# #         try:
# #             l_char_features = feature_extractor.extract_char_lex_feature(
# #                 content)
# #             l_word_features = feature_extractor.extract_word_lex_features(
# #                 content)
# #             # syn_features = feature_extractor.extract_syntactic_features(
# #             #     content)
# #             struc_features = feature_extractor.extract_structural_features(
# #                 content)
# #         except:
# #             bad_files += 1
# #             continue
# #         for word_feature in lexical_word_features:
# #             combined_features.append(l_word_features[word_feature])
# #         for char_feature in lexical_char_features:
# #             combined_features.append(l_char_features[char_feature])
# #         # for syn_feature in syntactic_features:
# #         #     combined_features.append(syn_features[syn_feature])
# #         for struct_feature in structural_features:
# #             combined_features.append(struc_features[struct_feature])
# #         test_vectors.append(combined_features)
# #         test_class_values.append(name)
# #         count += 1
# #         training_emails.append(email)
# Implementing the naive_bayes model here
# print training_vectors, training_class_values
X = np.array(training_vectors)
Y = np.array(training_class_values)
naive_bayes_model = MultinomialNB()
# naive_bayes_model.fit(X, Y)
print "naive bayes", naive_bayes_model
# print X,Y
# testing the accuracy
# test_X = np.array(test_vectors)
# test_Y = np.array(test_class_values)
scores = cross_val_score(naive_bayes_model, X, Y, cv=10, scoring="accuracy")
print scores
print scores.mean()
# # expected = test_Y
# # expected = Y
# # # predicted = naive_bayes_model.predict(test_X)
# # predicted = naive_bayes_model.predict(X)
# # # summarize the fit of the model
# # print(metrics.classification_report(expected, predicted))
# # print(metrics.confusion_matrix(expected, predicted))
# # print "*" * 15
# # implementing SVC here
# # X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state =0)
svm_linear_kernel_model = svm.SVC(kernel='linear')
# svm_rbf_kernel_model = svm.SVC(kernel='rbf')
# svm_polynomial_kernel_model = svm.SVC(kernel='poly')
# # cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)
# # gammas = np.logspace(-6, -1, 10)
# # classifier = GridSearchCV(estimator=svm_linear_kernel_model, cv=cv, param_grid=dict(gamma=gammas))
# # classifier.fit(X_train, y_train)
scores = cross_val_score(svm_linear_kernel_model, X, Y, cv=10, scoring="accuracy")
print scores
print scores.mean()
# scores = cross_val_score(svm_rbf_kernel_model, X, Y, cv=10, scoring="accuracy")
# print scores
# print scores.mean()
# scores = cross_val_score(svm_rbf_kernel_model, X, Y, cv=10, scoring="accuracy")
# print scores
# print scores.mean()
# title = 'Learning Curves (SVM, linear kernel, $\gamma=%.6f$)' %classifier.best_estimator_.gamma
# estimator = svm.SVC(kernel='linear', gamma=classifier.best_estimator_.gamma)
# plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
# plt.show()
# print classifier.score(X_test,y_test)

# svm_linear_kernel_model.fit(X, Y)
# print svm_linear_kernel_model
# # testing the accuracy
# expected = test_Y
# predicted = svm_linear_kernel_model.predict(test_X)
# # summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))
# print "*" * 15
#
# svm_rbf_kernel_model = svm.SVC(kernel='rbf')
# svm_rbf_kernel_model.fit(X, Y)
# print svm_rbf_kernel_model
# # testing the accuracy
# expected = test_Y
# predicted = svm_rbf_kernel_model.predict(test_X)
# # summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))
# print "*" * 15
#
# svm_polynomial_kernel_model = svm.SVC(kernel='poly')
# svm_polynomial_kernel_model.fit(X, Y)
# print svm_polynomial_kernel_model
# # testing the accuracy
# expected = test_Y
# predicted = svm_polynomial_kernel_model.predict(test_X)
# # summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))
# print "*" * 15
#
# # # implementing linear svms
# # linear_svm_linear_kernel_model = svm.LinearSVC()
# # linear_svm_linear_kernel_model.fit(X, Y)
# # print linear_svm_linear_kernel_model
# # # testing the accuracy
# # expected = Y
# # predicted = linear_svm_linear_kernel_model.predict(X)
# # # summarize the fit of the model
# # print(metrics.classification_report(expected, predicted))
# # print(metrics.confusion_matrix(expected, predicted))
# # print "*" * 15