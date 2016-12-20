import os
import shutil

'''
After this script runs, a new directory named relevant_email_data is created and for each author a new directory is
created which contains the sent emails of the author. Another file with only the content of the email removing all the
metadata is then created because we are only concerned with the email content and not its header information for
the purpose of this project
'''
# The path where the main dataset is present
enronPath = '/Users/rakshitha/AuthorIdentification/maildir'
# The path where the relevant information is stored after preprocessing
destPath = '../relevant_email_data'
for authorDir in os.listdir(enronPath):
    authorPath = enronPath + '/' + authorDir
    numForwards = 0
    # If the path is not a directory, but just a file, then skip it
    if not os.path.isdir(authorPath):
        continue
    destAuthorPath = destPath + '/' + authorDir
    if '_sent_mail' in os.listdir(authorPath):
        # if the destination path does not exist, create it.
        if not os.path.exists(destAuthorPath):
            os.makedirs(destAuthorPath)
        mailPath = authorPath + '/_sent_mail'
        for email in os.listdir(mailPath):
            filename = mailPath + '/' + email
            # This file consists of only the content of the email.
            dest_file_name = destAuthorPath + '/content_' + email
            if os.path.isfile(filename):
                fp1 = open(filename, 'r')
                # Metadata is separated from the content by consecutive carriage return and new line tabs.
                # Hence we split based on this regex and obtain the last
                content = fp1.read().split('\r\n\r\n')[-1]
                # If there is no content, then we do not create the new file.
                if content == '':
                    continue
                # Some of the emails contain only forwarded information. Since we are concerned about the writing
                # style of the author and not the person who has forwarded the mail to the author, we ignore such mails.
                if '---------------------- Forwarded' in content:
                    continue
                fp2 = open(dest_file_name, 'w')
                # we first copy the entire email with metadata. This is just for book keeping purposes.
                shutil.copy(filename, destAuthorPath)
                # We create a file with only the content to
                fp2.write(content)
                fp1.close()
                fp2.close()
