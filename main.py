import requests
import logging
from dotenv import load_dotenv
import os
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from telethon import TelegramClient, events
from telethon.tl.types import User
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import datetime

load_dotenv()

# Telegram Bot API credentials
TELEGRAM_API_ID = os.getenv("TG_API_ID")
TELEGRAM_API_HASH = os.getenv("TG_API_HASH")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Google Gemini API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///telegram_messages.db")  # Default to SQLite
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = sessionmaker(bind=engine)
session = Session()


# --- Database Model ---
class Message(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True)
    message_id = Column(Integer)  # Original Telegram message ID
    chat_id = Column(Integer)  # ID of the chat the message belongs to
    user_id = Column(Integer)
    username = Column(String)
    user_tag = Column(String)
    message_text = Column(Text)
    reaction = Column(String)
    reply_to_message_id = Column(Integer)
    time_sent = Column(DateTime)


Base.metadata.create_all(engine)

# --- Google Gemini Configuration ---
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('models/gemini-pro')

# Define safety settings for Gemini
safety_settings = {
    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
    'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
}

# Define generation config for Gemini
generation_config = GenerationConfig(
    temperature=0.9,
    top_k=40,
    top_p=0.95,
    max_output_tokens=1024,
)

# --- Telegram Bot Client ---
bot = TelegramClient('telegram_bot', TELEGRAM_API_ID, TELEGRAM_API_HASH).start(bot_token=TELEGRAM_BOT_TOKEN)


# --- Helper Functions ---
def query_gemini(prompt):
    try:
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config=generation_config
        )
        return response.text.strip()
    except Exception as e:
        logging.error(f"An error occurred while querying Gemini: {e}")
        return "An error occurred. Please try again later."


def format_messages_for_gemini(messages):
    messages_str = ""
    for msg in messages:
        messages_str += f"[{msg.time_sent}] {msg.username or msg.user_tag or 'Unknown'}: {msg.message_text}\n"
    return messages_str


# --- Telegram Bot Commands ---
@bot.on(events.NewMessage(pattern='/start'))
async def start_command(event):
    await event.respond("Hello! I'm a Telegram bot that can summarize and analyze conversations.")


@bot.on(events.NewMessage(pattern='/summarize'))
async def summarize_command(event):
    # Implement logic to retrieve messages from the database based on timeframe
    # For now, let's just get the last 10 messages:
    messages = session.query(Message).order_by(Message.time_sent.desc()).limit(10).all()

    if messages:
        # Format the retrieved messages for Gemini input
        formatted_messages = format_messages_for_gemini(messages)
        prompt = f"Please provide a concise summary of the following conversation:\n\n{formatted_messages}"
        # Query Gemini for summarization
        summary = query_gemini(prompt)
        await event.respond(f"Summary:\n\n{summary}")
    else:
        await event.respond("No messages found.")


@bot.on(events.NewMessage(pattern='/analyze'))
async def analyze_command(event):
    try:
        user_tag = event.message.text.split(" ")[1]
        user_messages = session.query(Message).filter_by(user_tag=user_tag).order_by(Message.time_sent.desc()).limit(
            5000).all()

        if user_messages:
            formatted_messages = format_messages_for_gemini(user_messages)
            prompt = f"Please analyze the following messages and provide insights about the user's behavior, communication patterns, or anything noteworthy:\n\n{formatted_messages}"
            analysis = query_gemini(prompt)
            await event.respond(f"Analysis of {user_tag}:\n\n{analysis}")
        else:
            await event.respond(f"No messages found for user {user_tag}.")

    except IndexError:
        await event.respond("Please provide a user tag after the /analyze command. Example: /analyze @example_user")


# --- Message Handling ---
@bot.on(events.NewMessage)
async def handle_new_message(event):
    # Retrieve the message sender
    sender = await event.get_sender()

    # Check if the sender is a User (not a Channel, etc.)
    if isinstance(sender, User):
        new_message = Message(
            message_id=event.message.id,
            chat_id=event.chat_id,
            user_id=sender.id,
            username=sender.username,
            user_tag=f'@{sender.username}' if sender.username else None,
            message_text=event.message.text,
            # Extract other details as needed (reaction, reply, etc.)
            time_sent=event.message.date,
        )
        session.add(new_message)
        session.commit()


@bot.on(events.NewMessage(pattern='/ask'))
async def ask_command(event):
    try:
        user_question = event.message.text[5:]  # Get text after '/ask '

        if not user_question:
            await event.respond(
                "Please provide a question after the /ask command. Example: /ask What's the weather like today?")
            return

        # Retrieve the last 5000 messages from the database for the current chat
        messages = session.query(Message).filter_by(chat_id=event.chat_id).order_by(Message.time_sent.desc()).limit(
            5000).all()

        if messages:
            # Format the messages for Gemini input
            formatted_messages = format_messages_for_gemini(messages)

            # Construct the prompt for Gemini
            prompt = f"Context:\n\n{formatted_messages}\n\nQuestion: {user_question}\n\nAnswer:"

            # Query Google Gemini
            gemini_response = query_gemini(prompt)

            await event.respond(gemini_response)
        else:
            await event.respond("No previous messages found in this chat.")

    except Exception as e:
        logging.error(f"An error occurred while processing /ask command: {e}")
        await event.respond("An error occurred. Please try again later.")


# --- Start the Bot ---
print("Bot is running...")
bot.run_until_disconnected()
