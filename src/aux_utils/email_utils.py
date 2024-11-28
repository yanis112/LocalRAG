import email
import functools
import imaplib
import json
import os
import re
from datetime import datetime
from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime
from html import unescape

from bs4 import BeautifulSoup
from dotenv import load_dotenv


def cache_daily(func):
    """Decorator to cache email fetching operations for one day"""
    cache_file = "aux_data/email_fetch_cache.json"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Créer le dossier aux_data s'il n'existe pas
        os.makedirs("aux_data", exist_ok=True)

        today = datetime.now().date().isoformat()

        # Check if cache file exists and is from today
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cache = json.load(f)
                if cache.get("last_fetch_date") == today:
                    print("Using cached email fetch from today")
                    return True  # Indique que le cache a été utilisé

        # Si pas de cache, exécuter la fonction
        result = func(*args, **kwargs)

        # Mettre à jour le cache
        with open(cache_file, "w") as f:
            json.dump({"last_fetch_date": today}, f)

        return result

    return wrapper


class EmailAgent:
    def __init__(self):
        """Initialize the EmailUtils class and load environment variables."""
        load_dotenv()
        self.imap_server = os.getenv("IMAP_SERVER")
        self.email_address = os.getenv("EMAIL_ADDRESS")
        self.password = os.getenv("EMAIL_PASSWORD")
        self.imap = None

    def get_emails(self, last_k=100):
        """
        Retrieve the latest `last_k` emails from the inbox.

        Args:
            last_k (int): Number of latest emails to retrieve.

        Returns:
            list: List of dictionaries, each containing keys 'date', 'content',
            'sender', 'attachment', and 'subject'.
        """
        _, msgnums = self.imap.search(None, "ALL")
        latest_msgnums = msgnums[0].split()[-last_k:]

        emails = []

        for msgnum in latest_msgnums:
            _, data = self.imap.fetch(msgnum, "(RFC822)")
            message = email.message_from_bytes(data[0][1])

            sender = self.decode_header(message["From"])
            subject = self.decode_header(message["Subject"])
            date = self.parse_date(message["Date"])
            content = ""
            attachments = []

            for part in message.walk():
                content_type = part.get_content_type()
                disposition = str(part.get("Content-Disposition"))

                if (
                    content_type == "text/plain"
                    and "attachment" not in disposition
                ):
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or "utf-8"
                    decoded_content = payload.decode(charset, errors="ignore")
                    cleaned_content = clean_text(decoded_content)
                    content += cleaned_content + "\n"
                elif (
                    content_type == "text/html"
                    and "attachment" not in disposition
                    and not content.strip()
                ):
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or "utf-8"
                    decoded_content = payload.decode(charset, errors="ignore")
                    text = html_to_text(decoded_content)
                    cleaned_content = clean_text(text)
                    content += cleaned_content + "\n"
                elif "attachment" in disposition:
                    filename = part.get_filename()
                    if filename:
                        filename = self.decode_header(filename)
                        attachments.append(filename)

            email_dict = {
                "date": date,
                "content": content.strip(),
                "sender": sender,
                "attachment": attachments,
                "subject": subject,
            }
            emails.append(email_dict)

        return emails

    @cache_daily
    def fetch_new_emails(self, last_k=100):
        """
        Fetch the latest `last_k` emails and save them as markdown files.

        Args:
            last_k (int): Number of latest emails to retrieve.
        """

        # launch the connect function
        self.connect()

        emails = self.get_emails(last_k=last_k)
        if not os.path.exists("data/emails"):
            os.makedirs("data/emails")

        for email_data in emails:
            subject = clean_filename(email_data["subject"][:50])
            # just keep the day and month, and year, no need for the time (H-M-S)
            short_date_str = email_data["date"].strftime("%Y-%m-%d")
            filename = f"data/emails/{short_date_str}_{subject}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# {email_data['subject']}\n\n")
                f.write(f"**Date:** {email_data['date']}\n\n")
                f.write(f"**From:** {email_data['sender']}\n\n")
                if email_data["attachment"]:
                    attachments = ", ".join(email_data["attachment"])
                    f.write(f"**Attachments:** {attachments}\n\n")
                f.write("## Content\n\n")
                f.write(email_data["content"])

        # disconnect from the email server
        self.disconnect()

    def disconnect(self):
        """Close the connection to the IMAP server."""
        self.imap.close()
        self.imap.logout()

    def connect(self):
        """Connect to the IMAP server and login."""
        self.imap = imaplib.IMAP4_SSL(self.imap_server)
        self.imap.login(self.email_address, self.password)
        self.imap.select("Inbox")

    @staticmethod
    def decode_header(header_value):
        """
        Decode email header value.

        Args:
            header_value (str): The header value to decode.

        Returns:
            str: The decoded header value.
        """
        if header_value:
            decoded_header = make_header(decode_header(header_value))
            return str(decoded_header)
        return ""

    @staticmethod
    def parse_date(date_str):
        """
        Parse email date string to datetime object.

        Args:
            date_str (str): The date string from the email header.

        Returns:
            datetime: The parsed datetime object.
        """
        return parsedate_to_datetime(date_str)


def clean_text(text):
    """
    Clean the text by removing unwanted characters and formatting.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    text = unescape(text)
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = text.strip()
    return text


def html_to_text(html_content):
    """
    Convert HTML content to plain text.

    Args:
        html_content (str): The HTML content to convert.

    Returns:
        str: The plain text extracted from HTML.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n")
    return text


def clean_filename(filename):
    """
    Clean the filename by removing or replacing invalid characters.

    Args:
        filename (str): The filename to clean.

    Returns:
        str: The cleaned filename.
    """
    filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
    filename = filename.strip().replace(" ", "_")
    return filename


if __name__ == "__main__":
    email_utils = EmailAgent()
    email_utils.connect()
    email_utils.fetch_new_emails(last_k=100)
    email_utils.disconnect()
