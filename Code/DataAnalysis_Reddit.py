import matplotlib.pyplot as pl
import json
import operator

subreddit_ids_above_threshold = json.load(open("subreddit_ids_above_threshold.json", "r"))
id_lengths = {}
for subreddit_id in subreddit_ids_above_threshold:
    id_lengths.update({subreddit_id: len(subreddit_ids_above_threshold[subreddit_id])})
count = 1
sorted_subreddit_ids = sorted(id_lengths.items(), key=operator.itemgetter(1))
sorted_subreddit_ids.reverse()
count = 1
# print subreddit ids and the number of authors in each of those subreddits
for index in range(len(sorted_subreddit_ids)):
    print count, ")", sorted_subreddit_ids[index]
    count += 1
xticks = []
x = []
y = []
for index in range(1, 21):
    xticks.append((sorted_subreddit_ids[index - 1][0].split('*$*'))[1])
    x.append(index)
    y.append(sorted_subreddit_ids[index - 1][1])
pl.xlabel("Subreddits")
pl.ylabel("Number of Authors")
pl.plot(x, y)
pl.xticks(x, xticks)
pl.show()
# get the largest subreddit
largest_subreddit = sorted_subreddit_ids[0][0]
print "The largest subreddit is : ", largest_subreddit
# extract all authors in the heaviest subreddit
authors_for_largest_subreddit = {}
for author in subreddit_ids_above_threshold[largest_subreddit]:
    authors_for_largest_subreddit.update(
        {author: subreddit_ids_above_threshold[largest_subreddit][author]["num_comments"]})
sorted_authors_for_largest_subreddit = sorted(authors_for_largest_subreddit.items(), key=operator.itemgetter(1))
sorted_authors_for_largest_subreddit.reverse()
x = []
y = []
xticks = []
# print the top 20 authors
print "The top 20 authors and their counts are"
for index in range(0, 21):
    print sorted_authors_for_largest_subreddit[index][0], " : ", sorted_authors_for_largest_subreddit[index][1]
    y.append(sorted_authors_for_largest_subreddit[index][1])
    x.append(index + 1)
    xticks.append(sorted_authors_for_largest_subreddit[index][0])
pl.xlabel("Authors")
pl.ylabel("Number of Comments")
pl.plot(x, y)
pl.xticks(x, xticks)
pl.show()
author_counts = {}
author_comment_range = {}
# finds out the number of words in the comments with the least number of words and finds out the number of words
#  in the comment with the largest number of comments
for index in (range(21)):
    author = sorted_authors_for_largest_subreddit[index][0]
    author_counts.update({author: {}})
    max = 1
    min = 1
    for index2 in range(len(subreddit_ids_above_threshold[largest_subreddit][author]["comments"])):
        wordlength = subreddit_ids_above_threshold[largest_subreddit][author]["comments"][index2]["wordlength"]
        if min > wordlength:
            min = wordlength
        if max < wordlength:
            max = wordlength
        if not wordlength in author_counts[author]:
            author_counts[author].update({wordlength: 0})
        author_counts[author][wordlength] += 1
    author_comment_range.update({author: (min, max)})
# find the range of the number of words between which a particular author's comments vary
for author in author_counts:
    print author, ":"
    print "range : ", author_comment_range[author]
author_average_word_percomment = {}
author_average_sentences_percomment = {}
# finds the average number of words per comment for a particular author in teh largest subreddit
for index in (range(21)):
    author = sorted_authors_for_largest_subreddit[index][0]
    comments = subreddit_ids_above_threshold[largest_subreddit][author]["comments"]
    num_comments = len(comments)
    word_count = 0
    sentences_count = 0
    for index2 in range(num_comments):
        word_count += subreddit_ids_above_threshold[largest_subreddit][author]["comments"][index2]["wordlength"]
        sentences_count += len(filter(None,
                                      subreddit_ids_above_threshold[largest_subreddit][author]["comments"][index2][
                                          "body"].split(".")))
    author_average_word_percomment.update({author: word_count / num_comments})
    print "The total number of sentences and words among all comments for ", author, "is :"
    print "sentences:", sentences_count, "num_comms:", num_comments
    author_average_sentences_percomment.update({author: sentences_count / num_comments})
print "The average number of words per comment for top 20 authors is:"
print author_average_word_percomment
print "The average number of sentences per comment for top 20 authors is:"
print author_average_sentences_percomment
