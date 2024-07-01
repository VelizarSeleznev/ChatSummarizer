import asyncio
import logging
import os
import sqlite3
import re

from telethon import TelegramClient, events
from telethon.tl.types import Message, InputPeerUser
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from telethon.tl.functions.messages import GetMessageReactionsListRequest

from db import DB, User, Message as DBMessage, Reaction, Media

# Настройте логирование
logging.basicConfig(level=logging.INFO)

# Загрузка переменных окружения из файла .env
load_dotenv()

# Получите API ID, API Hash и токен бота из переменных окружения
API_ID = os.getenv('TG_API_ID')
API_HASH = os.getenv('TG_API_HASH')
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Путь к базе данных SQLite
DB_PATH = 'data.sqlite'

# Инициализация базы данных
db = DB(DB_PATH)

# Создание экземпляра клиента Telegram
client = TelegramClient('bot_session', API_ID, API_HASH).start(
    bot_token=BOT_TOKEN)

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
conn = sqlite3.connect('data.sqlite')
cursor = conn.cursor()


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

    print(prompt)

    # Get summary from Gemini
    summary = query_gemini(prompt)
    return summary


@client.on(events.NewMessage)
async def handler(event: events.NewMessage.Event):
    """Обработчик новых сообщений."""
    message: Message = event.message

    # Получите информацию о пользователе
    user = await _get_user(message.sender_id)

    # Обработайте медиа-файлы (если есть)
    media = await _process_media(message)

    # Создайте объект сообщения для базы данных
    db_message = DBMessage(
        id=message.id,
        type="message",
        date=message.date,
        edit_date=message.edit_date,
        content=message.raw_text,
        reply_to=message.reply_to_msg_id,
        user=user,
        media=media
    )

    # Вставьте сообщение в базу данных
    db.insert_message(db_message)
    db.commit()

    # Обработайте реакции на сообщение
    await _process_reactions(message)


@client.on(events.NewMessage(pattern='/start'))
async def start(event):
    await event.reply("Welcome! I can summarize your chat. Use /summarize <YYYY-MM-DD> to get a summary.")


@client.on(events.NewMessage(pattern='/summarize'))
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
                   str(current_date.day) if current_date.day >
                                            9 else '0' + str(current_date.day))
    await event.reply("Generating summary, please wait...")
    summary = await summarize_chat(date)
    await event.reply(summary)


@client.on(events.NewMessage(pattern='/stats'))
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


async def _get_user(user_id: int) -> User:
    """Получите информацию о пользователе из Telegram и сохраните ее в базе данных."""
    try:
        user = await client.get_entity(user_id)
        db_user = User(
            id=user.id,
            username=user.username,
            first_name=user.first_name,
            last_name=user.last_name,
            tags="bot" if user.bot else "",
            avatar=None  # Загрузка аватара здесь опциональна
        )
        db.insert_user(db_user)
        db.commit()
        return db_user
    except Exception as e:
        logging.error(f"Ошибка получения пользователя: {e}")
        return User(id=user_id, username=str(user_id), first_name=None, last_name=None, tags="", avatar=None)


async def _process_media(message: Message) -> Media:
    """Обработайте медиа-файлы, прикрепленные к сообщению."""
    if message.media:
        # Здесь вы можете добавить логику для загрузки и обработки
        # медиа-файлов, таких как фотографии, видео, документы и т. д.
        # и вернуть объект Media с информацией о медиа-файле.
        pass
    return None


async def _process_reactions(message: Message) -> None:
    """Обработайте реакции на сообщение."""
    if message.reactions is not None and message.reactions.results:
        try:
            # Используем стандартную функцию GetMessageReactionsRequest
            reactions = await client(GetMessageReactionsListRequest(
                peer=message.peer_id,
                id=message.id,
                limit=1000  # Измените лимит при необходимости
            ))

            for reaction in reactions.reactions:
                try:
                    user = await _get_user(reaction.peer_id.user_id)
                    reaction_type = ""
                    reaction_content = ""
                    if reaction.reaction.emoticon:
                        reaction_type = 'emoticon'
                        reaction_content = reaction.reaction.emoticon
                    else:
                        reaction_type = 'custom'
                        reaction_content = str(reaction.reaction)
                    db_reaction = Reaction(
                        id='',
                        date=message.date,
                        message_id=message.id,
                        user_id=user.id,
                        type=reaction_type,
                        content=reaction_content
                    )
                    db.insert_reaction(db_reaction)
                except AttributeError:
                    logging.warning(
                        "Skipping reaction: could not get user or reaction info.")
            db.commit()
        except Exception as e:
            logging.error(
                f"Error fetching or processing reactions: {e}")


async def main():
    """Запустите бота и ждите новых сообщений."""
    logging.info("Бот запущен!")
    await client.run_until_disconnected()


if __name__ == '__main__':
    with client:
        client.loop.run_until_complete(main())
