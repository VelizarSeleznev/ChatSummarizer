import logging
import os
import re

from telethon.tl.types import PeerChannel, PeerChat
from telethon.sync import TelegramClient
from telethon import events
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
from enum import Enum, auto
from typing import Dict, Callable, Any
import ast
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
from telethon.tl.types import MessageMediaPhoto, Channel, DocumentAttributeFilename, User
from telethon import events, types
from telethon.tl.functions.bots import SetBotCommandsRequest
from telethon.tl.types import BotCommand
from PIL import Image

import requests

from db import DBManager, Message as DBMessage, Reaction, Media, User as db_User, Message

# Настройте логирование
logging.basicConfig(level=logging.INFO)

# Загрузка переменных окружения из файла .env
load_dotenv()

# Получите API ID, API Hash и токен бота из переменных окружения
API_ID = os.getenv('TG_API_ID')
API_HASH = os.getenv('TG_API_HASH')
BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = ''  # os.getenv("ANTHROPIC_API_KEY")

# Directory to store chat-specific LLM functions
CHAT_FUNCTIONS_DIR = 'chat_functions'

# Directory to store chat-specific prompts
CHAT_PROMPTS_DIR = 'chat_prompts'
os.makedirs(CHAT_PROMPTS_DIR, exist_ok=True)

# Directory to store chat-specific limits
CHAT_LIMITS_DIR = 'chat_limits'
os.makedirs(CHAT_LIMITS_DIR, exist_ok=True)

# Default limits
DEFAULT_LIMITS = {
    'summarize_limit': 500,
    'ask_limit': 500,
    'last_messages_limit': 500
}

# Add this global variable to store users in ongoing dialogues
ongoing_dialogues = set()

# Ensure the directory exists
os.makedirs(CHAT_FUNCTIONS_DIR, exist_ok=True)

# Dictionary to store users who are in the process of updating query_gemini
updating_users = {}

# Путь к базе данных SQLite
DB_PATH = 'data.sqlite'

# Создание экземпляра клиента Telegram
client = TelegramClient('test_bot_session', API_ID, API_HASH).start(
    bot_token=BOT_TOKEN)

BOT_NAME = client.get_me()

# --- Google Gemini Configuration ---
genai.configure(api_key=GOOGLE_API_KEY, transport='rest')
model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
}

# Define generation config for Gemini
generation_config = GenerationConfig(
    temperature=0.5,
    top_k=10,
    # top_p=0.95,
    # max_output_tokens=1024,
)


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


class CommandState(Enum):
    IDLE = auto()
    WAITING_FOR_INPUT = auto()


async def get_sender_id(event):
    try:
        sender = await event.get_sender()
        sender_id = sender.id
    except Exception as e:
        chat = await event.get_chat()
        sender_id = chat.id
    return int(sender_id)


class CommandHandler:
    def __init__(self):
        self.user_states: Dict[int, CommandState] = {}
        self.user_contexts: Dict[int, Any] = {}
        self.command_callbacks: Dict[str, Callable] = {}
        self.bot_username = BOT_NAME.username

    def register_command(self, command: str, callback: Callable):
        self.command_callbacks[command] = callback

    async def handle_message(self, event):
        sender_id = await get_sender_id(event)
        message = event.message

        command_match = re.match(r'/(\w+)(@\w+)?', message.text)
        if command_match:
            command = command_match.group(1)
            mentioned_username = command_match.group(2)

            # Проверяем, упоминается ли имя нашего бота или нет
            if mentioned_username:
                if mentioned_username == f'@{self.bot_username}':
                    if command == 'cancel':
                        await self.cancel_command(event)
                    elif command in self.command_callbacks:
                        self.user_states[sender_id] = CommandState.WAITING_FOR_INPUT
                        self.user_contexts[sender_id] = {'command': command, 'chat_id': event.chat_id}
                        await self.command_callbacks[command](event, self.user_contexts[sender_id])
            else:
                if command == 'cancel':
                    await self.cancel_command(event)
                elif command in self.command_callbacks:
                    self.user_states[sender_id] = CommandState.WAITING_FOR_INPUT
                    self.user_contexts[sender_id] = {'command': command, 'chat_id': event.chat_id}
                    await self.command_callbacks[command](event, self.user_contexts[sender_id])
        elif sender_id in self.user_states and self.user_states[sender_id] == CommandState.WAITING_FOR_INPUT:
            command = self.user_contexts[sender_id]['command']
            await self.command_callbacks[command](event, self.user_contexts[sender_id])

    async def cancel_command(self, event):
        sender_id = await get_sender_id(event)
        if sender_id in self.user_states:
            del self.user_states[sender_id]
        if sender_id in self.user_contexts:
            del self.user_contexts[sender_id]
        await event.reply("Current operation has been cancelled.")

    def reset_user_state(self, sender_id: int):
        if sender_id in self.user_states:
            del self.user_states[sender_id]
        if sender_id in self.user_contexts:
            del self.user_contexts[sender_id]


# Initialize the command handler
command_handler = CommandHandler()


# Default query_llm function
def default_query_llm(prompt):
    return query_gemini(prompt)


def query_gemini(prompt, images=None):
    try:
        content = [prompt]
        if images:
            content.extend(images)

        response = model.generate_content(
            content,
            safety_settings=safety_settings,
            generation_config=generation_config
        )
        return response.text.strip()
    except Exception as e:
        logging.error(f"An error occurred while querying the LLM: {e}")
        return f"произошла ошибка {e}"


def query_claude(prompt):
    headers = {
        "Content-Type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }

    data = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post("https://api.anthropic.com/v1/messages", json=data, headers=headers)

    if response.status_code == 200:
        return response.json()['content'][0]['text']
    else:
        return f"Error: {response.status_code}, {response.text}"


def get_chat_functions_file(chat_id):
    return os.path.join(CHAT_FUNCTIONS_DIR, f'{chat_id}_functions.json')


def get_chat_prompts_file(chat_id):
    return os.path.join(CHAT_PROMPTS_DIR, f'{chat_id}_prompts.json')


def get_chat_limits_file(chat_id):
    return os.path.join(CHAT_LIMITS_DIR, f'{chat_id}_limits.json')


def load_chat_prompts(chat_id):
    file_path = get_chat_prompts_file(chat_id)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return prompts  # Return default prompts if no custom prompts exist


def save_chat_prompts(chat_id, new_prompts):
    file_path = get_chat_prompts_file(chat_id)
    with open(file_path, 'w') as f:
        json.dump(new_prompts, f)


def load_chat_limits(chat_id):
    file_path = get_chat_limits_file(chat_id)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return DEFAULT_LIMITS


def save_chat_limits(chat_id, new_limits):
    file_path = get_chat_limits_file(chat_id)
    with open(file_path, 'w') as f:
        json.dump(new_limits, f)


def load_chat_functions(chat_id):
    file_path = get_chat_functions_file(chat_id)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        data = {
            'current_function': 'gemini',
            'claude': {'code': 'query_claude', 'can_process_images': False},
            'gemini': {'code': 'query_gemini', 'can_process_images': True}
        }
        with open(file_path, 'w') as f:
            json.dump(data, f)
    return data


def save_chat_functions(chat_id, functions):
    file_path = get_chat_functions_file(chat_id)
    with open(file_path, 'w') as f:
        json.dump(functions, f)


async def handle_update_prompt(event, context):
    sender_id = await get_sender_id(event)

    if event.message.text == '/cancel':
        command_handler.reset_user_state(sender_id)
        await event.reply("Operation cancelled.")
        return

    if 'step' not in context:
        context['step'] = 'prompt_naming'
        prompts = load_chat_prompts(context['chat_id'])
        prompts = prompts.keys()
        string = "\n".join(f"{key}" for key in prompts)
        await event.reply(f"Enter the name of the prompt you want to update (or /cancel to abort):\n" + string)
    elif context['step'] == 'prompt_naming':
        prompt_name = event.message.text.strip().upper()
        prompts = load_chat_prompts(context['chat_id'])
        prompts = prompts.keys()
        if prompt_name in prompts:
            context['prompt_name'] = prompt_name
            context['step'] = 'waiting_for_prompt'
            await event.reply(f"Please send the new content for the prompt '{prompt_name}' (or /cancel to abort).")
        else:
            await event.reply("Выберите существующий промпт")
    elif context['step'] == 'waiting_for_prompt':
        new_prompt = event.message.text
        chat_id = context['chat_id']
        chat_prompts = load_chat_prompts(chat_id)
        chat_prompts[context['prompt_name']] = new_prompt
        save_chat_prompts(chat_id, chat_prompts)
        await event.reply(f"Prompt '{context['prompt_name']}' has been updated.")
        command_handler.reset_user_state(sender_id)


async def handle_update_limit(event, context):
    sender_id = await get_sender_id(event)

    if event.message.text == '/cancel':
        command_handler.reset_user_state(sender_id)
        await event.reply("Operation cancelled.")
        return

    if 'step' not in context:
        context['step'] = 'limit_naming'
        chat_limits = load_chat_limits(context["chat_id"])
        limits_list = "\n".join(f"- {name}: {value}" for name, value in chat_limits.items())
        await event.reply("Enter the name of the limit you want to update (or /cancel to abort):\n" + limits_list)
    elif context['step'] == 'limit_naming':
        limit_name = event.message.text.strip().lower()
        context['limit_name'] = limit_name
        context['step'] = 'waiting_for_limit'
        await event.reply(
            f"Please send the new value for the limit '{limit_name}' (must be a positive integer, or /cancel to abort):")
    elif context['step'] == 'waiting_for_limit':
        try:
            new_limit = int(event.message.text)
            if new_limit <= 0:
                raise ValueError
            chat_id = context['chat_id']
            chat_limits = load_chat_limits(chat_id)
            chat_limits[context['limit_name']] = new_limit
            save_chat_limits(chat_id, chat_limits)
            await event.reply(f"Limit '{context['limit_name']}' has been updated to {new_limit}.")
        except ValueError:
            await event.reply("Invalid input. Please enter a positive integer.")
        finally:
            command_handler.reset_user_state(sender_id)


async def handle_update_query_llm(event, context):
    sender_id = await get_sender_id(event)

    if event.message.text == '/cancel':
        command_handler.reset_user_state(sender_id)
        await event.reply("Operation cancelled.")
        return

    if 'step' not in context:
        context['step'] = 'naming'
        await event.reply("Please enter a name for the new function (cannot be 'default', or /cancel to abort):")
    elif context['step'] == 'naming':
        name = event.message.text.strip()
        if name.lower() in ['default', 'claude', 'gemini']:
            await event.reply(
                "This name cannot be used as a function name. Please choose another name (or /cancel to abort):")
            return
        context['name'] = name
        context['step'] = 'waiting_for_file'
        await event.reply("Please send a .txt file with the code for the new query_llm function (or /cancel to abort).")
    elif context['step'] == 'waiting_for_file':
        if not event.message.document or not event.message.document.attributes[-1].file_name.endswith('.txt'):
            await event.reply("Please send a .txt file (or /cancel to abort).")
            return
        try:
            content = await client.download_media(event.message.document, file=bytes)
            new_code = content.decode('utf-8')
            chat_id = context['chat_id']
            await update_query_llm(event, new_code, chat_id, context['name'])
        except Exception as e:
            await event.reply(f"Error processing file: {e}")
        finally:
            command_handler.reset_user_state(sender_id)


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

    # Check if the function can process images
    can_process_images = False
    try:
        test_image = Image.new('RGB', (100, 100), color='red')
        result = temp_query_llm("Describe this image", images=[test_image])
        if isinstance(result, str) and len(result) > 0:
            can_process_images = True
    except Exception as e:
        pass

    chat_functions = load_chat_functions(chat_id)
    chat_functions[name] = {'code': new_code, 'can_process_images': can_process_images}
    save_chat_functions(chat_id, chat_functions)
    await event.reply(
        f"Функция query_llm '{name}' успешно сохранена для этого чата! Может обрабатывать изображения: {'Да' if can_process_images else 'Нет'}")
    await change_active_function(event.chat_id, name)


async def list_functions(event, context):
    chat_id = context['chat_id']
    chat_functions = load_chat_functions(chat_id)
    function_list = "Доступные функции:\n- default (встроенная)\n" + "\n".join(
        f"- {name}" for name in chat_functions if name != 'current_function')
    await event.reply(function_list)
    sender_id = await get_sender_id(event)
    command_handler.reset_user_state(sender_id)


async def set_function(event, context):
    sender_id = await get_sender_id(event)
    if event.message.text == '/cancel':
        command_handler.reset_user_state(sender_id)
        await event.reply("Операция отменена.")
        return
    if 'step' not in context:
        context['step'] = 'selecting_function'
        chat_id = context['chat_id']
        chat_functions = load_chat_functions(chat_id)
        function_list = "Выберите функцию для использования:\n1. default (встроенная)\n" + "\n".join(
            f"{i + 1}. {name}" for i, name in enumerate(chat_functions) if name != 'current_function')
        await event.reply(function_list + "\n\nОтветьте именем функции, которую вы хотите использовать.")
    elif context['step'] == 'selecting_function':
        chat_id = context['chat_id']
        function_name = event.message.text.strip()
        if await change_active_function(chat_id, function_name):
            await event.reply(f"Успешно установлена текущая функция '{function_name}'.")
            command_handler.reset_user_state(sender_id)
        else:
            await event.reply('Выберите верное имя функции или напишите /cancel')


async def change_active_function(chat_id, function_name):
    chat_functions = load_chat_functions(chat_id)
    if function_name == 'default' or function_name in chat_functions and function_name != 'current_function':
        with open(get_chat_functions_file(chat_id), 'r+') as f:
            data = json.load(f)
            data['current_function'] = function_name
            f.seek(0)
            json.dump(data, f)
            f.truncate()
        return True
    return False


async def get_query_llm(chat_id):
    chat_functions = load_chat_functions(chat_id)
    current_function = chat_functions.get('current_function', 'default')

    if current_function == 'default':
        return default_query_llm, True
    elif current_function == 'gemini':
        return query_gemini, True
    elif current_function == 'claude':
        return query_claude, False
    else:
        function_data = chat_functions.get(current_function)
        if function_data:
            exec(function_data['code'])
            return locals()['query_llm'], function_data['can_process_images']
        else:
            logging.warning(
                f"Function '{current_function}' not found for chat {chat_id}. Using Gemini as default.")
            return default_query_llm, True


async def get_last_messages(chat_id: int, limit: int):
    db = DBManager.get_db(chat_id)
    query = f"""
SELECT 
    messages.date, 
    messages.content, 
    users.first_name
FROM 
    messages
JOIN 
    users
ON 
    messages.user_id = users.id
LIMIT {limit};"""

    df = pd.read_sql_query(query, db.conn)
    return df


async def ask_question(chat_id: int, question: str):
    chat_prompts = load_chat_prompts(chat_id)
    chat_limits = load_chat_limits(chat_id)
    df = await get_last_messages(chat_id, chat_limits['ask_limit'])

    if df.empty:
        return "В истории чата не найдено ни одного сообщения."

    prompt = f"Вопрос пользователя: {question}" + chat_prompts['ASK_PROMPT']
    data_as_string = df.to_string(index=False)
    prompt += f"\n {data_as_string}"
    query_llm, _ = await get_query_llm(chat_id)
    answer = query_llm(prompt)
    return answer


async def messages_by_day(event, context):
    chat_id = event.chat_id
    db = DBManager.get_db(chat_id)
    query = """
    SELECT DATE(date) as date, COUNT(*) as count
    FROM messages
    GROUP BY DATE(date)
    ORDER BY date
    """
    df = pd.read_sql_query(query, db.conn)

    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['count'])
    plt.title('Количество сообщений по дням')
    plt.xlabel('Дата')
    plt.ylabel('Количество сообщений')
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    await event.reply(file=buf, attributes=[DocumentAttributeFilename('messages_by_day.png')])
    sender_id = await get_sender_id(event)
    command_handler.reset_user_state(sender_id)


async def activity_by_hour(event, context):
    chat_id = event.chat_id
    db = DBManager.get_db(chat_id)
    query = """
    SELECT strftime('%H', date) as hour, COUNT(*) as count
    FROM messages
    GROUP BY hour
    ORDER BY hour
    """
    df = pd.read_sql_query(query, db.conn)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='hour', y='count', data=df)
    plt.title('Активность по времени суток')
    plt.xlabel('Час')
    plt.ylabel('Количество сообщений')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    await event.reply(file=buf, attributes=[DocumentAttributeFilename('activity_by_hour.png')])
    sender_id = await get_sender_id(event)
    command_handler.reset_user_state(sender_id)


async def message_length_distribution(event, context):
    chat_id = event.chat_id
    db = DBManager.get_db(chat_id)
    query = "SELECT length(content) as length FROM messages"
    df = pd.read_sql_query(query, db.conn)

    plt.figure(figsize=(12, 6))
    sns.histplot(df['length'], kde=True)
    plt.title('Распределение длины сообщений')
    plt.xlabel('Длина сообщения')
    plt.ylabel('Частота')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    await event.reply(file=buf, attributes=[DocumentAttributeFilename('message_length_distribution.png')])
    sender_id = await get_sender_id(event)
    command_handler.reset_user_state(sender_id)


async def user_activity_comparison(event, context):
    chat_id = event.chat_id
    db = DBManager.get_db(chat_id)
    query = """
    SELECT u.username, COUNT(*) as message_count
    FROM messages m
    JOIN users u ON m.user_id = u.id
    GROUP BY u.id
    ORDER BY message_count DESC
    LIMIT 10
    """
    df = pd.read_sql_query(query, db.conn)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='username', y='message_count', data=df)
    plt.title('Сравнение активности пользователей')
    plt.xlabel('Пользователь')
    plt.ylabel('Количество сообщений')
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    await event.reply(file=buf, attributes=[DocumentAttributeFilename('user_activity_comparison.png')])
    sender_id = await get_sender_id(event)
    command_handler.reset_user_state(sender_id)


async def word_trend(event, context):
    chat_id = event.chat_id
    word = event.raw_text.split(maxsplit=1)[1] if len(event.raw_text.split()) > 1 else None

    if not word:
        await event.reply("Пожалуйста, укажите слово после команды /word_trend")
        return

    db = DBManager.get_db(chat_id)
    query = f"""
    SELECT DATE(date) as date, 
        SUM(CASE WHEN content LIKE '%{word}%' THEN 1 ELSE 0 END) as count
    FROM messages
    GROUP BY DATE(date)
    ORDER BY date
    """
    df = pd.read_sql_query(query, db.conn)

    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['count'])
    plt.title(f'Тренд использования слова "{word}"')
    plt.xlabel('Дата')
    plt.ylabel('Количество упоминаний')
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    await event.reply(file=buf, attributes=[DocumentAttributeFilename(f'word_trend_{word}.png')])
    sender_id = await get_sender_id(event)
    command_handler.reset_user_state(sender_id)


async def handle_ask(event, context):
    chat_id = event.chat_id
    message = event.message
    db = DBManager.get_db(chat_id)

    # Check if there's a reply
    replied_to = await message.get_reply_message()

    # Extract the command and potential bot name
    command_match = re.match(r'^/(\w+)(@\w+)?', message.text)

    images = []
    if message.media and isinstance(message.media, MessageMediaPhoto):
        image = await message.download_media(file=bytes)
        pil_image = Image.open(io.BytesIO(image))
        images.append(pil_image)

    if replied_to and replied_to.media and isinstance(replied_to.media, MessageMediaPhoto):
        replied_image = await replied_to.download_media(file=bytes)
        pil_replied_image = Image.open(io.BytesIO(replied_image))
        images.append(pil_replied_image)

    if command_match:
        command = command_match.group(1)
        bot_name = command_match.group(2)

        if bot_name:
            # If bot name is present, don't look for username
            username_match = None
        else:
            # Look for username in the rest of the message
            username_match = re.search(r'@(\w+)', message.text[len(command_match.group(0)):])
    else:
        # If it's not a command, don't look for username
        username_match = None

    query_llm, can_process_images = await get_query_llm(chat_id)

    if replied_to:
        # Check if the replied message contains a link
        urls = re.findall(r'(https?://\S+)', replied_to.text)
        if urls:
            url = urls[0]
            try:
                prompts = load_chat_prompts(context["chat_id"])
                prompt = prompts["SUMMARIZE_URL_PROMPT"]
                question = re.sub(r'^/\w+(@\w+)?(\s+)?', '', message.text).strip()
                if question:
                    prompt = f"{url} \n" + prompt + f"\nТак же ответь на вопрос {question}"
                else:
                    prompt = f"{url} \n" + prompt
                summary = query_llm(prompt)
                response = f"{summary}"
            except requests.RequestException as e:
                response = f"Failed to fetch the content from the link: {e}"
        elif images and can_process_images:
            question = re.sub(r'^/\w+(@\w+)?(\s+)?', '', message.text).strip()
            prompts = load_chat_prompts(context["chat_id"])
            prompt = prompts["SUMMARIZE_IMAGE_PROMPT"]
            if question:
                prompt = prompt + "\nТак же ответь на вопрос" + question
            response = query_llm(prompt, images=images)
        elif images and not can_process_images:
            response = "The current query_llm function cannot process images. Please switch to a function that supports image processing."
        else:
            # Use user_info functionality if no link is found
            response = await analyze_user(event, db)
    elif username_match:
        # Use user_info functionality if username is mentioned
        response = await analyze_user(event, db)
    else:
        # Use original ask functionality
        question = message.text.split()
        string = ''
        for word in question[1:]:
            string += word
            string += ' '
        if not string and not images:
            response = "Please provide a question after the /ask command or attach an image."
        elif images and not can_process_images:
            response = "The current query_llm function cannot process images. Please switch to a function that supports image processing."
        else:
            response = query_llm(string, images=images if can_process_images else None)

    sender_id = await get_sender_id(event)
    command_handler.reset_user_state(sender_id)

    await event.reply(response)


@client.on(events.NewMessage)
async def handler(event: events.NewMessage.Event):
    """Обработчик новых сообщений."""
    await command_handler.handle_message(event)
    message = event.message
    chat_id = event.chat_id

    db = DBManager.get_db(chat_id)

    # Получите информацию о пользователе
    user = await get_user(event)

    # Создайте объект сообщения для базы данных
    db_message = DBMessage(
        id=message.id,
        type="message",
        date=message.date,
        edit_date=message.edit_date,
        content=message.raw_text,
        reply_to=message.reply_to_msg_id,
        user=user,
        media=None
    )

    # Вставьте сообщение в базу данных
    db.insert_message(db_message)
    db.commit()


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


@client.on(events.NewMessage(pattern='/user_info --help'))
async def user_info_help(event):
    help_message = help_texts['USER_INFO_HELP']
    await event.reply(help_message)


@client.on(events.NewMessage(pattern='/update_query_llm --help'))
async def update_query_llm_help(event):
    help_message = help_texts['UPDATE_QUERY_LLM_HELP']
    await event.reply(help_message)


@client.on(events.NewMessage(pattern='/update_prompt --help'))
async def update_prompt_help(event):
    help_message = help_texts['UPDATE_PROMPT_HELP']
    await event.reply(help_message)


@client.on(events.NewMessage(pattern='/update_limit --help'))
async def update_limit_help(event):
    help_message = help_texts['UPDATE_LIMIT_HELP']
    await event.reply(help_message)


async def list_prompts(event, context):
    chat_id = event.chat_id
    chat_prompts = load_chat_prompts(chat_id)
    prompts_list = "\n".join(f"- {name}" for name in chat_prompts.keys())
    message = f"""Available prompts for this chat:

{prompts_list}

Use /update_prompt to modify a prompt."""
    await event.reply(message)
    sender_id = await get_sender_id(event)
    command_handler.reset_user_state(sender_id)


async def list_limits(event, context):
    chat_id = event.chat_id
    chat_limits = load_chat_limits(chat_id)
    limits_list = "\n".join(f"- {name}: {value}" for name, value in chat_limits.items())
    message = f"""Current message load limits for this chat:

{limits_list}

Use /update_limit to modify a limit."""
    await event.reply(message)
    sender_id = await get_sender_id(event)
    command_handler.reset_user_state(sender_id)


async def summarize(event, context):
    chat_id = event.chat_id
    chat_prompts = load_chat_prompts(chat_id)
    chat_limits = load_chat_limits(chat_id)
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    dates = re.findall(date_pattern, event.message.text)

    # Check if the message is a reply
    replied_to = await event.message.get_reply_message()

    query_llm, can_process_images = await get_query_llm(chat_id)

    if replied_to:
        if replied_to.media and isinstance(replied_to.media, MessageMediaPhoto):
            if can_process_images:
                image = await replied_to.download_media(file=bytes)
                pil_image = Image.open(io.BytesIO(image))
                prompt = chat_prompts['SUMMARIZE_IMAGE_PROMPT']
                summary = query_llm(prompt, images=[pil_image])
                await event.reply(summary)
            else:
                await event.reply(
                    "The current query_llm function cannot process images. Please switch to a function that supports image processing.")
        else:
            prompt = chat_prompts['SUMMARIZE_MESSAGE_PROMPT']
            prompt += f"\n{replied_to.text}"
            summary = query_llm(prompt)
            await event.reply(summary)
    else:
        if len(dates) == 2:
            start_date = dates[0]
            end_date = dates[1]
        elif len(dates) == 1:
            start_date = end_date = dates[0]
        else:
            current_date = datetime.now().date()
            start_date = end_date = current_date.strftime('%Y-%m-%d')

        df = await get_chat_messages_between_dates(chat_id, start_date, end_date, chat_limits['summarize_limit'])

        if df.empty:
            await event.reply("Не найдено ни одного сообщения в указанном диапазоне времени.")
            return

        prompt = chat_prompts['SUMMARIZE_PROMPT']
        prompt += df.to_string(index=False)

        summary = query_llm(prompt)
        await event.reply(summary)

    sender_id = await get_sender_id(event)
    command_handler.reset_user_state(sender_id)


async def get_chat_messages_between_dates(chat_id: int, start_date: str, end_date: str, limit: int):
    db = DBManager.get_db(chat_id)
    query = f"""
    SELECT m.date, u.first_name, m.content, COUNT(r.id) as reaction_count
    FROM messages m
    JOIN users u ON m.user_id = u.id
    LEFT JOIN reactions r ON m.id = r.message_id
    WHERE DATE(m.date) BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY m.id
    ORDER BY m.date
    LIMIT {limit}
    """

    df = pd.read_sql_query(query, db.conn)
    return df


async def stats(event, context):
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
    sender_id = await get_sender_id(event)
    command_handler.reset_user_state(sender_id)


async def analyze_user(event, db):
    chat_id = event.chat_id
    message = event.message
    replied_to = await message.get_reply_message()

    # Extract username if provided
    username_match = re.search(r'@(\w+)', message.text)
    username = username_match.group(1) if username_match else None

    # Get user from reply or username
    if replied_to:
        try:
            user = await client.get_entity(replied_to.from_id)
            user_id = user.id
        except Exception as e:
            chat = await event.get_chat()
            user_id = chat.id
    elif username:
        user = await client.get_entity(username)
        user_id = user.id
    else:
        return "Please provide a username or reply to a user's message."

    # Get user data from the database
    cursor = db.conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user_data = cursor.fetchone()

    if not user_data:
        return "User not found in the database."

    # Get user message count
    cursor.execute("SELECT COUNT(*) FROM messages WHERE user_id = ?", (user_id,))
    message_count = cursor.fetchone()[0]

    # Check if there's a question after the command
    question = re.sub(r'^/ask(@\w+)?(\s+)?(@\w+)?(\s+)?', '', message.text).strip()

    # Get user messages
    chat_limits = load_chat_limits(chat_id)
    cursor.execute(f"""
        SELECT content FROM messages 
        WHERE user_id = ? 
        ORDER BY date DESC 
        LIMIT {chat_limits['ask_limit']}
    """, (user_id,))
    user_messages = cursor.fetchall()

    messages_text = "\n".join([msg[0] for msg in user_messages])

    # Prepare improved prompt for LLM
    prompt = f"""
    Задача: Создать точный и подробный портрет пользователя на основе его сообщений в чате.

    Данные о пользователе:
    - Имя: {user_data[2]}
    - Общее количество сообщений: {message_count}

    создай психологический портрет человека, про которого задали вопрос, приводя цитаты из его сообщений и ответь на вопрос

    Сообщения пользователя для анализа:
    {messages_text}

    Вопрос (если есть): {question}

    Формат ответа:
    1. Краткое общее описание пользователя (2-3 предложения с цитатами)
    2. Подробный анализ по каждому пункту инструкций с цитатами
    3. Ответ на конкретный вопрос (если задан)
    """

    # Query LLM
    query_llm, _ = await get_query_llm(chat_id)
    answer = query_llm(prompt)

    return answer


async def handle_user_info(event, context):
    chat_id = event.chat_id
    db = DBManager.get_db(chat_id)

    # Generate and send user activity graphs
    await send_user_activity_graphs(event, db)

    sender_id = await get_sender_id(event)
    command_handler.reset_user_state(sender_id)


async def send_user_activity_graphs(event, db):
    message = event.message
    replied_to = await message.get_reply_message()

    # Extract username if provided
    username_match = re.search(r'@(\w+)', message.text)
    username = username_match.group(1) if username_match else None

    # Get user from reply or username
    if replied_to:
        try:
            user = await client.get_entity(replied_to.from_id)
            user_id = user.id
        except Exception as e:
            chat = await event.get_chat()
            user_id = chat.id
    elif username:
        user = await client.get_entity(username)
        user_id = user.id
    else:
        await event.reply("Please provide a username or reply to a user's message to generate activity graphs.")
        return

    # Generate graphs
    await generate_user_activity_by_day(event, db, user_id)
    await generate_user_activity_by_hour(event, db, user_id)
    await generate_user_message_length_distribution(event, db, user_id)


async def generate_user_activity_by_day(event, db, user_id):
    query = f"""
    SELECT DATE(date) as date, COUNT(*) as count
    FROM messages
    WHERE user_id = {user_id}
    GROUP BY DATE(date)
    ORDER BY date
    """
    df = pd.read_sql_query(query, db.conn)

    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['count'])
    plt.title('User Activity by Day')
    plt.xlabel('Date')
    plt.ylabel('Number of Messages')
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    await event.reply(file=buf, attributes=[DocumentAttributeFilename('user_activity_by_day.png')])


async def generate_user_activity_by_hour(event, db, user_id):
    query = f"""
    SELECT strftime('%H', date) as hour, COUNT(*) as count
    FROM messages
    WHERE user_id = {user_id}
    GROUP BY hour
    ORDER BY hour
    """
    df = pd.read_sql_query(query, db.conn)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='hour', y='count', data=df)
    plt.title('User Activity by Hour')
    plt.xlabel('Hour')
    plt.ylabel('Number of Messages')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    await event.reply(file=buf, attributes=[DocumentAttributeFilename('user_activity_by_hour.png')])


async def generate_user_message_length_distribution(event, db, user_id):
    query = f"SELECT length(content) as length FROM messages WHERE user_id = {user_id}"
    df = pd.read_sql_query(query, db.conn)

    plt.figure(figsize=(12, 6))
    sns.histplot(df['length'], kde=True)
    plt.title('User Message Length Distribution')
    plt.xlabel('Message Length')
    plt.ylabel('Frequency')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    await event.reply(file=buf, attributes=[DocumentAttributeFilename('user_message_length_distribution.png')])


async def get_user(event) -> db_User:
    """Получите информацию о пользователе или канале из Telegram и сохраните ее в базе данных."""
    db = DBManager.get_db(event.chat_id)
    try:
        sender = await event.get_sender()
        if isinstance(sender, User):
            user_id = sender.id
            username = sender.username if sender.username else sender.first_name
            first_name = sender.first_name
            last_name = sender.last_name if sender.last_name else None

            db_user = db_User(
                id=user_id,
                username=username,
                first_name=first_name,
                last_name=last_name,
                tags="user",
                avatar=None  # Загрузка аватара здесь опциональна
            )
        elif isinstance(sender, Channel):
            channel_username = sender.username
            channel_id = sender.id
            channel_name = sender.title
            db_user = db_User(
                id=channel_id,
                username=channel_username,
                first_name=channel_name,
                last_name=None,
                tags="channel",
                avatar=None  # Загрузка аватара здесь опциональна
            )
        elif event.is_group:
            chat = await event.get_chat()
            group_id = chat.id
            group_name = chat.title
            group_username = chat.username
            db_user = db_User(
                id=group_id,
                username=group_username,
                first_name=group_name,
                last_name=None,
                tags="group",
                avatar=None  # Загрузка аватара здесь опциональна
            )
        else:
            entity = await event.get_sender()
            db_user = db_User(
                id=entity.id,
                username=entity.username,
                first_name=getattr(entity, 'first_name', 'Канал'),
                last_name=getattr(entity, 'last_name', None),
                tags="bot" if getattr(entity, 'bot', False) else "",
                avatar=None  # Загрузка аватара здесь опциональна
            )
        db.insert_user(db_user)
        db.commit()
        return db_user
    except Exception as e:
        logging.error(f"Ошибка получения пользователя: {e}")
        return db_User(id=event.sender_id, username=str(event.sender_id), first_name=str(event.sender_id),
                       last_name=None, tags="", avatar=None)


def initialize_functions():
    command_handler.register_command('update_prompt', handle_update_prompt)
    command_handler.register_command('update_limit', handle_update_limit)
    command_handler.register_command('update_query_llm', handle_update_query_llm)
    command_handler.register_command('list_functions', list_functions)
    command_handler.register_command('set_function', set_function)
    command_handler.register_command('messages_by_day', messages_by_day)
    command_handler.register_command('activity_by_hour', activity_by_hour)
    command_handler.register_command('message_length', message_length_distribution)
    command_handler.register_command('user_activity', user_activity_comparison)
    command_handler.register_command('word_trend', word_trend)
    command_handler.register_command('ask', handle_ask)
    command_handler.register_command('list_prompts', list_prompts)
    command_handler.register_command('list_limits', list_limits)
    command_handler.register_command('summarize', summarize)
    command_handler.register_command('stats', stats)
    command_handler.register_command('user_info', handle_user_info)


async def set_bot_commands(client):
    commands = [
        BotCommand(command="start", description="Запустить бота"),
        BotCommand(command="help", description="Получить помощь"),
        BotCommand(command="summarize", description="Суммировать сообщения чата"),
        BotCommand(command="ask", description="Задать вопрос о чате"),
        BotCommand(command="user_info", description="Получить информацию о пользователе"),
        BotCommand(command="stats", description="Получить статистику чата"),
        BotCommand(command="list_prompts", description="Список доступных подсказок"),
        BotCommand(command="list_limits", description="Список текущих ограничений"),
        BotCommand(command="update_prompt", description="Обновить подсказку"),
        BotCommand(command="update_limit", description="Обновить ограничение"),
        BotCommand(command="update_query_llm", description="Обновить функцию запроса LLM"),
        BotCommand(command="list_functions", description="Список доступных функций"),
        BotCommand(command="set_function", description="Установить активную функцию"),
        BotCommand(command="messages_by_day", description="Показать график сообщений по дням"),
        BotCommand(command="activity_by_hour", description="Показать активность по часам"),
        BotCommand(command="message_length", description="Показать распределение длины сообщений"),
        BotCommand(command="user_activity", description="Сравнить активность пользователей"),
        BotCommand(command="word_trend", description="Показать тренд использования слов"),
    ]

    await client(SetBotCommandsRequest(
        scope=types.BotCommandScopeDefault(),
        lang_code='ru',
        commands=commands
    ))
    await client(SetBotCommandsRequest(
        scope=types.BotCommandScopeUsers(),
        lang_code='ru',
        commands=commands
    ))
    await client(SetBotCommandsRequest(
        scope=types.BotCommandScopeChats(),
        lang_code='ru',
        commands=commands
    ))
    await client(SetBotCommandsRequest(
        scope=types.BotCommandScopeChatAdmins(),
        lang_code='ru',
        commands=commands
    ))

    await client(SetBotCommandsRequest(
        scope=types.BotCommandScopeDefault(),
        lang_code='ua',
        commands=commands
    ))
    await client(SetBotCommandsRequest(
        scope=types.BotCommandScopeUsers(),
        lang_code='ua',
        commands=commands
    ))
    await client(SetBotCommandsRequest(
        scope=types.BotCommandScopeChats(),
        lang_code='ua',
        commands=commands
    ))
    await client(SetBotCommandsRequest(
        scope=types.BotCommandScopeChatAdmins(),
        lang_code='ua',
        commands=commands
    ))


async def main():
    """Запустите бота и ждите новых сообщений."""
    initialize_functions()
    await set_bot_commands(client)
    logging.info("Бот запущен!")
    await client.run_until_disconnected()


if __name__ == '__main__':
    with client:
        client.loop.run_until_complete(main())
