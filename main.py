from telethon import TelegramClient, events
from telethon.tl.types import InputPeerUser
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import asyncio
import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import logging
import re

load_dotenv()

api_id = os.getenv("TG_API_ID")
api_hash = os.getenv("TG_API_HASH")
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the Telethon client
bot = TelegramClient('bot', api_id, api_hash).start(bot_token=bot_token)

# Connect to the SQLite database
conn = sqlite3.connect('data.sqlite')
cursor = conn.cursor()

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


async def get_chat_messages(date):
    query = f"""
    SELECT m.id, m.date, u.username, m.content, COUNT(r.id) as reaction_count
    FROM messages m
    JOIN users u ON m.user_id = u.id
    LEFT JOIN reactions r ON m.id = r.message_id
    WHERE DATE(m.date) = '{date}'
    GROUP BY m.id
    ORDER BY m.date
    LIMIT 5000
    """

    df = pd.read_sql_query(query, conn)
    return df


async def summarize_chat(date):
    df = await get_chat_messages(date)

    if df.empty:
        return "No messages found in the specified time range."

    # Prepare the prompt for Gemini
    prompt = f"Summarize the following chat messages from this date: {date}\n\n"
    for _, row in df.iterrows():
        prompt += f"{row['date']} - {row['username']}: {row['content']}\n"

    # Get summary from Gemini
    summary = query_gemini(prompt)
    return summary


@bot.on(events.NewMessage(pattern='/start'))
async def start(event):
    await event.reply("Welcome! I can summarize your chat. Use /summarize <days> to get a summary.")


@bot.on(events.NewMessage(pattern='/summarize'))
async def summarize(event):
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    match = re.search(date_pattern, event.message.text)
    if match:
        date = str(match.group())
        print("Дата:", date)
    else:
        current_date = datetime.now()
        date = str(current_date.year) + '-' + (
            str(current_date.month) if current_date.month > 9 else '0' + str(current_date.month)) + '-' + (
                   str(current_date.day) if current_date.day > 9 else '0' + str(current_date.day))
    # try:
    #     date = int(event.message.text.split()[1])
    #     print(date)
    # except (IndexError, ValueError):
    #     current_date = datetime.now()
    #     date = str(current_date.year) + '-' + (
    #         str(current_date.month) if current_date.month > 9 else '0' + str(current_date.month)) + '-' + (
    #                str(current_date.day) if current_date.day > 9 else '0' + str(current_date.day))

    # print(date)
    await event.reply("Generating summary, please wait...")
    summary = await summarize_chat(date)
    await event.reply(summary)


@bot.on(events.NewMessage(pattern='/stats'))
async def stats(event):
    cursor.execute("SELECT COUNT(*) FROM messages")
    total_messages = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM reactions")
    total_reactions = cursor.fetchone()[0]

    stats_message = f"Chat Statistics:\n"
    stats_message += f"Total Messages: {total_messages}\n"
    stats_message += f"Total Users: {total_users}\n"
    stats_message += f"Total Reactions: {total_reactions}"

    await event.reply(stats_message)


async def main():
    await bot.run_until_disconnected()


if __name__ == '__main__':
    with bot:
        bot.loop.run_until_complete(main())
