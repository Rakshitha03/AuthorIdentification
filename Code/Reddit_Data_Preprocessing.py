from collections import defaultdict
import json
import pickle

authorcount = defaultdict(float)
authorcomments = defaultdict(list)
subredditcollection = {}
subredditcollectioncount = defaultdict(float)
list = []
count = 0
N = 3000000
# RC_2015-01.json is a 35GB file containing 1 month of reddit comments from Jan 2015
with open("RC_2015-01.json") as myfile:
    # The first 3 million comments are parsed for the purpose of this project and stored in a list
    list = [next(myfile) for x in xrange(N)]
for x in range(N):
    # each comment is appended with a new line and as part of preprocesing it has to be removed before storing
    list[x] = list[x].replace("\n", '')
jsonlist = []
for x in range(N):
    # each comment is stored as a json block in list jsonlist
    jsonlist.append(json.loads(list[x]))
for dict in jsonlist:
    # for ease of identification subreddit_id and subreddit(name of the subredd with the given id) are appended and
    # stored with *$* in between as <subreddit_id>*$*<subreddit>
    subreddit_id = dict["subreddit_id"] + "*$*" + dict["subreddit"]
    author_name = dict["author"]
    # the following loops exclude comments with deleted comment bodies, authors and subreddits and are used for
    # indexing comments by subreddit_id and authorname
    if not subreddit_id == "[deleted]" and not author_name == "[deleted]" and not (
            dict["body"] == "[deleted]" or dict["body"] == "**[deleted]**"):
        if subreddit_id in subredditcollection:
            if author_name not in subredditcollection[subreddit_id]:
                subredditcollection[subreddit_id].update({author_name: {"comments": [], "num_comments": 0}})
            subredditcollection[subreddit_id][author_name]["comments"].append(
                {"body": dict["body"], "charlength": len(dict["body"]), "wordlength": len(dict["body"].split())})
            subredditcollection[subreddit_id][author_name]["num_comments"] += 1
        else:
            subredditcollection[subreddit_id] = {author_name: {"comments": [
                {"body": dict["body"], "charlength": len(dict["body"]), "wordlength": len(dict["body"].split())}],
                                                               "num_comments": 1}}
subreddit_ids_above_threshold = {}
# The following loops identify subreddits with atleast 4 authors who have more than 50 comments and extract
# those authors and their corresponding comments
for subreddit_id in subredditcollection:
    author_comment_count = 0
    authors_above_threshold = {}
    for author_name in subredditcollection[subreddit_id]:
        if subredditcollection[subreddit_id][author_name]["num_comments"] >= 50:
            author_comment_count += 1
            authors_above_threshold.update({author_name: subredditcollection[subreddit_id][author_name]})
    if author_comment_count > 4:
        subreddit_ids_above_threshold.update({subreddit_id: authors_above_threshold})
# the following loops print the subreddits that have been extracted above that meet threshold requirements specified above
print "the number of subreddits with requisite data:", len(subreddit_ids_above_threshold)
for subreddit_id in subreddit_ids_above_threshold:
    print "subreddit:", subreddit_id
    print "The authors, comments and comment stats:"
    for author_name in subreddit_ids_above_threshold[subreddit_id]:
        print "author:", author_name
        for comment in range(len(subreddit_ids_above_threshold[subreddit_id][author_name]["comments"])):
            print  subreddit_ids_above_threshold[subreddit_id][author_name]["comments"][comment]
# store indexed comments as a file on the system
json.dump(subredditcollection, open("subreddit_collection.json", "w"))
# store indexed comments that meet the threshold requirements on the file system
json.dump(subreddit_ids_above_threshold, open("subreddit_ids_above_threshold.json",
                                              "w"))
