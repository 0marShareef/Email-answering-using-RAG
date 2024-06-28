import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import uvicorn

# Import necessary functions from your existing script
from inbound_parse import get_gmail_service, get_email_details, qa, create_message, send_message

app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Gmail service
service = get_gmail_service()

class Email(BaseModel):
    id: str
    subject: str
    sender: str
    snippet: str

class ResponseModel(BaseModel):
    content: str

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/emails", response_model=List[Email])
async def get_emails():
    results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=10).execute()
    messages = results.get('messages', [])
    emails = []
    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id'], format='metadata').execute()
        subject = next((header['value'] for header in msg['payload']['headers'] if header['name'].lower() == 'subject'), 'No Subject')
        sender = next((header['value'] for header in msg['payload']['headers'] if header['name'].lower() == 'from'), 'Unknown Sender')
        emails.append(Email(id=message['id'], subject=subject, sender=sender, snippet=msg['snippet']))
    return emails

@app.get("/email/{email_id}")
async def get_email(email_id: str):
    subject, sender, body, thread_id = get_email_details(service, email_id)
    return {"subject": subject, "sender": sender, "body": body, "thread_id": thread_id}

@app.post("/generate_response", response_model=ResponseModel)
async def generate_response(email: dict):
    rag_input = f"Subject: {email['subject']}\nFrom: {email['sender']}\nBody: {email['body']}"
    rag_response = qa.invoke(rag_input)
    return ResponseModel(content=rag_response["result"])

@app.post("/send_reply")
async def send_reply(email: dict):
    reply_subject = f"Re: {email['subject']}"
    reply_message = create_message("me", email['sender'], reply_subject, email['response'], email['thread_id'])
    send_message(service, "me", reply_message)
    return {"status": "success"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)