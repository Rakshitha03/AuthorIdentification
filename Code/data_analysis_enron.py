''' Script which performs some data analysis on the enron dataset'''
import os
import matplotlib.pyplot as plt

source_path = '/Users/rakshitha/AuthorIdentification/relevant_email_data'
# dictionary used to store count of the number of mails sent by each author.
mail_count = {}
# dictionary used to store the content information for each other.
author_content_info = {}
for authorDir in os.listdir(source_path):
    authorPath = source_path + '/' + authorDir
    if not os.path.isdir(authorPath):
        continue
    num_mails = 0
    for email in os.listdir(authorPath):
        # Ignoring the files which contain metadata and selecting files which have only content.
        if not email.startswith("content"):
            continue
        num_mails += 1
        fp = open(authorPath + '/' + email)
        content = fp.read()
        if authorDir not in author_content_info:
            author_content_info[authorDir] = {"content": []}
        content_info = {
            "mail_content": content,
            "word_length": len(content.split()),
            "sentence_length": len(filter(None, content.split('.')))
        }
        author_content_info[authorDir]["content"].append(content_info)
        total_word_count = 0
        total_sentence_count = 0
    mail_count[authorDir] = num_mails
    # Analysing the total word length, sentence length information per author.
    for each_mail_info in author_content_info[authorDir]["content"]:
        total_word_count += each_mail_info["word_length"]
        total_sentence_count += each_mail_info["sentence_length"]
    author_content_info[authorDir]["total_word_count"] = total_word_count
    author_content_info[authorDir][
        "total_sentence_count"] = total_sentence_count
# printing statistics for the top 10 authors.
print "Statistics"
print "Top 10 email senders "
top_senders = sorted(mail_count, key=mail_count.get, reverse=True)
word_count_list = []
sentence_count_list = []
authors = []
x = []
for i in range(10):
    author_name = top_senders[i]
    print top_senders[i], mail_count[top_senders[i]]
    authors.append(author_name)
    word_count_list.append(author_content_info[author_name]["total_word_count"])
    sentence_count_list.append(
        author_content_info[author_name]["total_sentence_count"])
    x.append(i + 1)
    print "Total num of words\t Total num of sentences"
    print author_content_info[top_senders[i]]["total_word_count"], author_content_info[top_senders[i]][
        "total_sentence_count"]
    print
    print
print "Author \t Avg word len \t Avg sent len"
for i in range(10):
    average_word_length = word_count_list[i] / mail_count[top_senders[i]]
    average_sentence_length = sentence_count_list[i] / mail_count[
        top_senders[i]]
    print top_senders[i] + "\t\t" + str(average_word_length) + "\t\t" + str(
        average_sentence_length)
# plotting the sentence count and word count graph for top 10 authors

plt.xlabel("Authors")
plt.ylabel("Word count")
plt.bar(x, word_count_list, align='center')
plt.xticks(x, authors)
for i, j in zip(x, word_count_list):
    plt.annotate(str(j), xy=(i-0.15, j))
plt.show()
plt.xlabel("Authors")
plt.ylabel("sentence count")
plt.bar(x, sentence_count_list, align='center')
plt.xticks(x, authors)
for i, j in zip(x, sentence_count_list):
    plt.annotate(str(j), xy=(i-0.10, j))
plt.show()
