import imaplib
import smtplib
import time
import imaplib
import email
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText

class emailNotificationClass(object):
    def __init__(self):
        self.nothing = 0


    def send_email(self,user, pwd, recipient, subject, body):
        import smtplib

        gmail_user = user
        gmail_pwd = pwd
        FROM = user
        TO = recipient if type(recipient) is list else [recipient]
        SUBJECT = subject
        TEXT = body

        # Prepare actual message
        message = """From: %s\nTo: %s\nSubject: %s\n\n%s
        """ % (FROM, ", ".join(TO), SUBJECT, TEXT)

        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.ehlo()
            server.starttls()
            server.login(gmail_user, gmail_pwd)
            server.sendmail(FROM, TO, message)
            server.close()
            print 'successfully sent the mail'
        except:
            print "failed to send mail"

    ORG_EMAIL = "@gmail.com"
    FROM_EMAIL = "yourEmailAddress" + ORG_EMAIL
    FROM_PWD = "yourPassword"
    SMTP_SERVER = "imap.gmail.com"
    SMTP_PORT = 993


    # -------------------------------------------------
    #
    # Utility to read email from Gmail Using Python
    #
    # ------------------------------------------------

    def read_email_from_gmail(self):
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login("hadi.beikmohammadi@gmail.com", "RickandMorthy")
            mail.select('inbox')

            type, data = mail.search(None, 'ALL')
            mail_ids = data[0]

            id_list = mail_ids.split()
            first_email_id = int(id_list[0])
            latest_email_id = int(id_list[-1])

            for i in range(latest_email_id, first_email_id, -1):
                typ, data = mail.fetch(i, '(RFC822)')

                for response_part in data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_string(response_part[1])
                        email_subject = msg['subject']
                        email_from = msg['from']
                        #print 'From : ' + email_from + '\n'
                        print email_subject + '\n'

        except Exception, e:
            print str(e)


if __name__ == "__main__":
    email_sender=emailNotificationClass()
    email_sender.read_email_from_gmail()
    #email_sender.send_email("whatsuploop@gmail.com","RickandMorthy","hadi.bmohammadi@gmail.com","Happy Message" , "first point is finished")
    #  vis = visdom.Visdom()
    #  vis.text('Hello, world!')
    #  vis.image(np.ones((3, 10, 10)))
