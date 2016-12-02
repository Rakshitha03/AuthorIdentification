import os
import shutil

'''
After this script runs, a new directory named relevant data is created and for each author a new directory is
created which contains the sent emails of the author. Another file with only the content of the email removing all the
metadata is then created because we are only concerned with the email content and not its header information for
the purpose of this project
'''
# The path where the main dataset is present
enron_path = '/Users/rakshitha/AuthorIdentification/maildir'
# The path where the relevant information is stored after preprocessing
dest_path = '../relevant_email_data'
for person_dir in os.listdir(enron_path):
    person_path = enron_path + '/' + person_dir
    noForwards = 0
    if not os.path.isdir(person_path):
        continue
    dest_person_path = dest_path + '/' + person_dir
    if '_sent_mail' in os.listdir(person_path):
        if not os.path.exists(dest_person_path):
            os.makedirs(dest_person_path)
        mail_path = person_path + '/_sent_mail'
        for email in os.listdir(mail_path):
            filename = mail_path + '/' + email
            dest_file_name = dest_person_path + '/content_' + email
            if (os.path.isfile(filename)):
                fp1 = open(filename, 'r')
                content = fp1.read().split('\r\n\r\n')[-1]
                # If there is no content, then we do not create the new file.
                if content == '':
                    continue
                # Some of the emails contain forwarded information. Since we are concerned about the writing
                # style of the author and not the person who has forwarded the mail to the author, we ignore such mails.
                if '---------------------- Forwarded' in content:
                    continue
                fp2 = open(dest_file_name, 'w')
                shutil.copy(filename, dest_person_path)
                fp2.write(content)
                fp1.close()
                fp2.close()
