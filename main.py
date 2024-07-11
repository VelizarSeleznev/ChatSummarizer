import asyncio
import logging
import os
import sqlite3
import re

import io
import matplotlib.pyplot as plt
from telethon.tl.types import InputPeerUser

from telethon import TelegramClient, events
from telethon.tl.types import Message, InputPeerUser
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from telethon.tl.functions.messages import GetMessageReactionsListRequest
import ast
import json

import requests  # сделано больше для того, чтобы использовать другие функции запроса к ллм

from db import DBManager, User, Message as DBMessage, Reaction, Media
from plotting_scripts import messages_by_day, activity_by_hour, message_length_distribution, user_activity_comparison, \
    word_trend

# Настройте логирование
logging.basicConfig(level=logging.INFO)

# Загрузка переменных окружения из файла .env
load_dotenv()

# Получите API ID, API Hash и токен бота из переменных окружения
API_ID = os.getenv('TG_API_ID')
API_HASH = os.getenv('TG_API_HASH')
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Directory to store chat-specific LLM functions
CHAT_FUNCTIONS_DIR = 'chat_functions'

# Ensure the directory exists
os.makedirs(CHAT_FUNCTIONS_DIR, exist_ok=True)

# Dictionary to store users who are in the process of updating query_gemini
updating_users = {}

# Dictionary to store chat-specific query_llm functions
chat_query_llm = {}

# Путь к базе данных SQLite
DB_PATH = 'data.sqlite'

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
    temperature=0.1,
    top_k=40,
    # top_p=0.95,
    # max_output_tokens=1024,
)
conn = sqlite3.connect('data.sqlite')
cursor = conn.cursor()


def load_text_data(filepath):
    prompts = {}
    current_key = None
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                current_key = line[1:-1]
                prompts[current_key] = ""
            elif current_key:
                prompts[current_key] += line + "\n"
    return prompts


prompts = load_text_data('prompts.txt')
help_texts = load_text_data('help_texts.txt')


# Default query_llm function
def default_query_llm(prompt):
    try:
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config=generation_config
        )
        return response.text.strip()
    except Exception as e:
        logging.error(f"An error occurred while querying the LLM: {e}")
        return "An error occurred. Please try again later."


def get_chat_functions_file(chat_id):
    return os.path.join(CHAT_FUNCTIONS_DIR, f'{chat_id}_functions.json')


def load_chat_functions(chat_id):
    file_path = get_chat_functions_file(chat_id)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}


def save_chat_functions(chat_id, functions):
    file_path = get_chat_functions_file(chat_id)
    with open(file_path, 'w') as f:
        json.dump(functions, f)


@client.on(events.NewMessage(pattern='/update_query_llm'))
async def handle_update_query_llm(event):
    sender = await event.get_sender()
    chat_id = event.chat_id
    await event.reply("Пожалуйста, введите имя для новой функции (не может быть 'default'):")
    updating_users[sender.id] = {'chat_id': chat_id, 'state': 'naming'}


@client.on(events.NewMessage(
    func=lambda e: e.sender_id in updating_users and updating_users[e.sender_id]['state'] == 'naming'))
async def handle_function_name(event):
    sender = await event.get_sender()
    chat_id = updating_users[sender.id]['chat_id']
    name = event.message.text.strip()

    if name == '/update_query_llm':
        return

    if name.lower() == 'default':
        await event.reply("Нельзя использовать 'default' в качестве имени функции. Пожалуйста, выберите другое имя:")
        return

    chat_functions = load_chat_functions(chat_id)
    if name in chat_functions:
        await event.reply("Это имя уже используется. Пожалуйста, выберите другое имя:")
        return

    updating_users[sender.id]['name'] = name
    updating_users[sender.id]['state'] = 'waiting_for_file'
    await event.reply("Пожалуйста, отправьте .txt файл с кодом новой функции query_llm.")


@client.on(events.NewMessage(
    func=lambda e: e.document and e.sender_id in updating_users and updating_users[e.sender_id][
        'state'] == 'waiting_for_file'))
async def handle_document(event):
    sender = await event.get_sender()
    chat_id = updating_users[sender.id]['chat_id']
    name = updating_users[sender.id]['name']

    if not event.document.attributes[-1].file_name.endswith('.txt'):
        await event.reply("Пожалуйста, отправьте .txt файл.")
        return

    try:
        content = await client.download_media(event.document, file=bytes)
        new_code = content.decode('utf-8')
        await update_query_llm(event, new_code, chat_id, name)
    except Exception as e:
        await event.reply(f"Ошибка при обработке файла: {e}")
    finally:
        del updating_users[sender.id]


async def update_query_llm(event, new_code, chat_id, name):
    try:
        ast.parse(new_code)
    except SyntaxError as e:
        await event.reply(f"Синтаксическая ошибка в предоставленном коде: {e}")
        return

    try:
        exec(new_code)
        temp_query_llm = locals()['query_llm']
    except Exception as e:
        await event.reply(f"Ошибка при создании функции: {e}")
        return

    try:
        result = temp_query_llm("Тестовый запрос")
        if not isinstance(result, str):
            raise ValueError("Функция должна возвращать строку")
    except Exception as e:
        await event.reply(f"Ошибка при тестировании новой функции: {e}")
        return

    chat_functions = load_chat_functions(chat_id)
    chat_functions[name] = new_code
    save_chat_functions(chat_id, chat_functions)
    await event.reply(f"Функция query_llm '{name}' успешно сохранена для этого чата!")
    await change_active_function(event.chat_id, name)


@client.on(events.NewMessage(pattern='/list_functions'))
async def list_functions(event):
    chat_id = event.chat_id
    chat_functions = load_chat_functions(chat_id)
    function_list = "Доступные функции:\n- default (встроенная)\n" + "\n".join(
        f"- {name}" for name in chat_functions if name != 'current_function')
    await event.reply(function_list)


@client.on(events.NewMessage(pattern='/set_function'))
async def set_function(event):
    chat_id = event.chat_id
    chat_functions = load_chat_functions(chat_id)
    function_list = "Выберите функцию для использования:\n0. default (встроенная)\n" + "\n".join(
        f"{i + 1}. {name}" for i, name in enumerate(chat_functions) if name != 'current_function')
    await event.reply(function_list + "\n\nОтветьте номером функции, которую вы хотите использовать.")
    updating_users[event.sender_id] = {'chat_id': chat_id, 'state': 'selecting_function'}


async def change_active_function(chat_id, function_name):
    chat_functions = load_chat_functions(chat_id)
    if function_name == 'default' or function_name in chat_functions:
        with open(get_chat_functions_file(chat_id), 'r+') as f:
            data = json.load(f)
            data['current_function'] = function_name
            f.seek(0)
            json.dump(data, f)
            f.truncate()
        return True
    return False


@client.on(events.NewMessage(
    func=lambda e: e.sender_id in updating_users and updating_users[e.sender_id]['state'] == 'selecting_function'))
async def handle_function_selection(event):
    sender = await event.get_sender()
    chat_id = updating_users[sender.id]['chat_id']
    chat_functions = load_chat_functions(chat_id)
    if event.message.text == '/set_function':
        return
    try:
        selection = int(event.message.text.strip()) - 1
        if selection == -1:
            function_name = 'default'
        else:
            function_name = list(chat_functions.keys())[selection]

        if await change_active_function(chat_id, function_name):
            await event.reply(f"Успешно установлена текущая функция '{function_name}'.")
        else:
            await event.reply("Не удалось установить выбранную функцию. Пожалуйста, попробуйте еще раз.")
    except (ValueError, IndexError):
        await event.reply("Неверный выбор. Пожалуйста, попробуйте еще раз.")
    finally:
        del updating_users[sender.id]


async def get_query_llm(chat_id):
    chat_functions = load_chat_functions(chat_id)
    current_function = chat_functions.get('current_function', 'default')

    if current_function == 'default':
        return default_query_llm
    else:
        function_code = chat_functions.get(current_function)
        if function_code:
            exec(function_code)
            return locals()['query_llm']
        else:
            logging.warning(
                f"Функция '{current_function}' не найдена для чата {chat_id}. Используется функция по умолчанию.")
            return default_query_llm


async def get_chat_messages(chat_id: int, date: str):
    db = DBManager.get_db(chat_id)
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

    df = pd.read_sql_query(query, db.conn)
    return df


async def get_last_messages(chat_id: int, limit: int = 5000):
    db = DBManager.get_db(chat_id)
    query = f"""
    SELECT m.id, m.date, u.username, m.content
    FROM messages m
    JOIN users u ON m.user_id = u.id
    ORDER BY m.date DESC
    LIMIT {limit}
    """

    df = pd.read_sql_query(query, db.conn)
    return df.sort_values('date')  # Sort by date in ascending order


async def ask_question(chat_id: int, question: str):
    df = await get_last_messages(chat_id)

    if df.empty:
        return "В истории чата не найдено сообщений."

    prompt = prompts['ASK_PROMPT'].format(question=question)

    for _, row in df.iterrows():
        prompt += f"{row['date']} - {row['username']}: {row['content']}\n"

    query_llm = await get_query_llm(chat_id)
    answer = query_llm(prompt)
    return answer


@client.on(events.NewMessage(pattern='/messages_by_day'))
async def _messages_by_day(event):
    await messages_by_day(event, client)


@client.on(events.NewMessage(pattern='/activity_by_hour'))
async def _activity_by_hour(event):
    await activity_by_hour(event, client)


@client.on(events.NewMessage(pattern='/message_length'))
async def _message_length_distribution(event):
    await message_length_distribution(event, client)


@client.on(events.NewMessage(pattern='/user_activity'))
async def _user_activity_comparison(event):
    await user_activity_comparison(event, client)


@client.on(events.NewMessage(pattern='/word_trend'))
async def _word_trend(event):
    await word_trend(event, client)


@client.on(events.NewMessage(pattern=r'/ask (.+)'))
async def handle_ask(event):
    chat_id = event.chat_id
    question = event.pattern_match.group(1)

    answer = await ask_question(chat_id, question)
    await event.reply(answer)


@client.on(events.NewMessage)
async def handler(event: events.NewMessage.Event):
    """Обработчик новых сообщений."""
    message: Message = event.message
    chat_id = event.chat_id

    db = DBManager.get_db(chat_id)

    # Получите информацию о пользователе
    user = await _get_user(message.sender_id, chat_id)

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
    await _process_reactions(message, chat_id)


@client.on(events.NewMessage(pattern='/start'))
async def start(event):
    welcome_message = help_texts['START']
    await event.reply(welcome_message)


@client.on(events.NewMessage(pattern='/help'))
async def help(event):
    help_message = help_texts['HELP']
    await event.reply(help_message)


# Добавим дополнительные команды для получения справки по конкретным функциям

@client.on(events.NewMessage(pattern='/summarize --help'))
async def summarize_help(event):
    help_message = help_texts['SUMMARIZE_HELP']
    await event.reply(help_message)


@client.on(events.NewMessage(pattern='/ask --help'))
async def ask_help(event):
    help_message = help_texts['ASK_HELP']
    await event.reply(help_message)


@client.on(events.NewMessage(pattern='/update_query_llm --help'))
async def update_query_llm_help(event):
    help_message = help_texts['UPDATE_QUERY_LLM_HELP']
    await event.reply(help_message)


@client.on(events.NewMessage(pattern='/summarize'))
async def summarize(event):
    chat_id = event.chat_id
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    dates = re.findall(date_pattern, event.message.text)

    if len(dates) == 2:
        start_date = dates[0]
        end_date = dates[1]
    elif len(dates) == 1:
        start_date = dates[0]
        end_date = dates[0]
    else:
        current_date = datetime.now().date()
        start_date = current_date
        end_date = current_date

    # Get chat messages within the specified date range
    df = await get_chat_messages_between_dates(chat_id, start_date, end_date)

    if df.empty:
        await event.reply("No messages found in the specified time range.")
        return

    # Prepare the prompt for Gemini
    prompt = prompts['SUMMARIZE_PROMPT']

    for _, row in df.iterrows():
        prompt += f"{row['date']} - {row['username']}: {row['content']}\n"

    # Get summary from Gemini
    # summary = query_gemini(prompt)
    query_llm = await get_query_llm(chat_id)
    summary = query_llm(prompt)
    await event.reply(summary)


async def get_chat_messages_between_dates(chat_id: int, start_date: str, end_date: str):
    db = DBManager.get_db(chat_id)
    query = f"""
    SELECT m.id, m.date, u.username, m.content, COUNT(r.id) as reaction_count
    FROM messages m
    JOIN users u ON m.user_id = u.id
    LEFT JOIN reactions r ON m.id = r.message_id
    WHERE DATE(m.date) BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY m.id
    ORDER BY m.date
    LIMIT 5000
    """

    df = pd.read_sql_query(query, db.conn)
    return df


@client.on(events.NewMessage(pattern='/stats'))
async def stats(event):
    chat_id = event.chat_id
    db = DBManager.get_db(chat_id)

    cursor = db.conn.cursor()
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


async def _get_user(user_id: int, chat_id: int) -> User:
    """Получите информацию о пользователе из Telegram и сохраните ее в базе данных."""
    db = DBManager.get_db(chat_id)
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


async def _process_reactions(message: Message, chat_id: int) -> None:
    """Обработайте реакции на сообщение."""
    db = DBManager.get_db(chat_id)
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
                    user = await _get_user(reaction.peer_id.user_id, chat_id)
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
