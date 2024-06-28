import os
import base64
import re
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import html
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from email.utils import parseaddr
from email.mime.text import MIMEText
import time

# Gmail API setup
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.send', 'https://www.googleapis.com/auth/gmail.modify']

# Load environment variables
load_dotenv()

# RAG setup
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

chat = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
)

qa = RetrievalQA.from_chain_type(
    llm=chat, chain_type="stuff", retriever=vectorstore.as_retriever()
)

def get_gmail_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'client_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('gmail', 'v1', credentials=creds)

def decode_body(data):
    try:
        return base64.urlsafe_b64decode(data + '=' * (4 - len(data) % 4)).decode('utf-8')
    except Exception as e:
        print(f"Error decoding body: {e}")
        return ""

def get_email_details(service, msg_id):
    try:
        message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
        payload = message['payload']
        headers = payload['headers']

        subject = next((header['value'] for header in headers if header['name'].lower() == 'subject'), 'No Subject')
        sender = next((header['value'] for header in headers if header['name'].lower() == 'from'), 'Unknown Sender')
        sender_email = parseaddr(sender)[1]  # Extract email address from sender

        def parse_parts(parts):
            body = ""
            for part in parts:
                if part.get('parts'):
                    body += parse_parts(part['parts'])
                if part.get('mimeType') == 'text/plain':
                    data = part['body'].get('data')
                    if data:
                        body += decode_body(data)
                elif part.get('mimeType') == 'text/html':
                    data = part['body'].get('data')
                    if data and not body:
                        body += decode_body(data)
            return body

        if 'parts' in payload:
            body = parse_parts(payload['parts'])
        elif 'body' in payload:
            data = payload['body'].get('data')
            if data:
                body = decode_body(data)
        else:
            body = "Unable to find email body"

        body = html.unescape(body)
        body = re.sub(r'<[^>]+>', '', body)
        body = re.sub(r'\s+', ' ', body).strip()
        body = re.sub(r'\[image:.*?\]', '', body)

        return subject, sender_email, body, message['threadId']
    except Exception as e:
        print(f"An error occurred while processing message {msg_id}: {e}")
        return "No Subject", "Unknown Sender", "", ""

def create_message(sender, to, subject, message_text, thread_id):
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    return {
        'raw': raw_message,
        'threadId': thread_id
    }

def send_message(service, user_id, message):
    try:
        message = service.users().messages().send(userId=user_id, body=message).execute()
        print(f"Message Id: {message['id']}")
        return message
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def process_new_emails(service):
    while True:
        try:
            results = service.users().messages().list(userId='me', labelIds=['INBOX'], q='is:unread').execute()
            messages = results.get('messages', [])

            if not messages:
                print("No new messages found. Waiting...")
            else:
                print(f"Processing {len(messages)} new email(s):")
                for i, message in enumerate(messages, 1):
                    print(f"Email {i}:")
                    subject, sender_email, body, thread_id = get_email_details(service, message['id'])
                    print(f"Subject: {subject}")
                    print(f"From: {sender_email}")
                    if body:
                        print("Email Content Preview:")
                        print(body[:500] + "..." if len(body) > 500 else body)
                        print("\nGenerating RAG Response...")
                        rag_input = f"Subject: {subject}\nFrom: {sender_email}\nBody: {body}"
                        rag_response = qa.invoke(rag_input)
                        print("RAG Response:")
                        print(rag_response["result"])
                        
                        # Prepare and send reply
                        reply_subject = f"Re: {subject}"
                        reply_body = f"In response to your email:\n\n{rag_response['result']}\n\nThis is an automated response generated by an AI assistant."
                        reply_message = create_message("me", sender_email, reply_subject, reply_body, thread_id)
                        print("\nSending reply...")
                        send_message(service, "me", reply_message)
                        print("Reply sent successfully.")

                        # Mark the email as read
                        service.users().messages().modify(userId='me', id=message['id'], body={'removeLabelIds': ['UNREAD']}).execute()
                    else:
                        print("No readable content found in this email.")
                    print("-" * 50)

            # Wait for a short period before checking for new emails again
            time.sleep(5)  # Check every 5 seconds

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)

if __name__ == '__main__':
    service = get_gmail_service()
    print("Starting email monitoring...")
    process_new_emails(service)