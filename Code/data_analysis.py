import os

source_path = '/Users/rakshitha/AuthorIdentification/relevant_email_data'
mail_count = {}
author_content_info = {}
for person_dir in os.listdir(source_path):
    person_path = source_path + '/' + person_dir
    if not os.path.isdir(person_path):
        continue
    num_mails = 0
    for email in os.listdir(person_path):
        if not email.startswith("content"):
            continue
        num_mails += 1
        fp = open(person_path + '/' + email)
        content = fp.read()
        if person_dir not in author_content_info:
            author_content_info[person_dir] = {"content": []}
        content_info = {
            "mail_content": content,
            "word_length": len(content.split()),
            "sentence_length": len(filter(None, content.split('.')))
        }
        # if content == '':
        #     # print " Content nil ", email, person_dir
        #     continue
        #
        # lexical_features = feature_extractor.extract_char_lex_feature(
        #     content)
        # syntactic_features = feature_extractor.extract_syntactic_features(
        #     content)
        # structural_features = feature_extractor.extract_structural_features(
        #     content)

        author_content_info[person_dir]["content"].append(content_info)
        total_word_count = 0
        total_sentence_count = 0
    mail_count[person_dir] = num_mails
    for each_mail_info in author_content_info[person_dir]["content"]:
        total_word_count += each_mail_info["word_length"]
        total_sentence_count += each_mail_info["sentence_length"]
    author_content_info[person_dir]["total_word_count"] = total_word_count
    author_content_info[person_dir][
        "total_sentence_count"] = total_sentence_count
print "Statistics"
print "Top 20 email senders "
top_senders = sorted(mail_count, key=mail_count.get, reverse=True)
# import pdb
# pdb.set_trace()
word_count_list = []
sentence_count_list = []
authors = []
x = []
for i in range(20):
    author_name = top_senders[i]
    print top_senders[i], mail_count[top_senders[i]]
    authors.append(author_name)
    word_count_list.append(author_content_info[author_name]["total_word_count"])
    sentence_count_list.append(
        author_content_info[author_name]["total_sentence_count"])
    x.append(i+1)
    print "Total num of words\t Total num of sentences"
    print author_content_info[top_senders[i]]["total_word_count"], author_content_info[top_senders[i]]["total_sentence_count"]
    print
    print
print "Author \t Avg word len \t Avg sent len"
for i in range(20):
    average_word_length = word_count_list[i] / mail_count[top_senders[i]]
    average_sentence_length = sentence_count_list[i] / mail_count[
        top_senders[i]]
    print top_senders[i] + "\t\t" + str(average_word_length) + "\t\t" + str(
        average_sentence_length)
import matplotlib.pyplot as plt
plt.xlabel("Authors")
plt.ylabel("Word count")
plt.bar(x, word_count_list, align='center')
plt.xticks(x, authors)
plt.show()
plt.xlabel("Authors")
plt.ylabel("sentence count")
plt.bar(x, sentence_count_list, align='center')
plt.xticks(x, authors)
for i,j in zip(x,word_count_list):
    plt.annotate(str(j), xy=(i,j))
plt.show()
