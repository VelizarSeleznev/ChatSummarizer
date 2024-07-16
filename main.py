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
from telethon import events, Button
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from telethon.tl.functions.messages import GetMessageReactionsListRequest
from telethon.tl.types import User as TelegramUser
from telethon.tl.types import Channel
from telethon.tl.types import User as TelegramUser, Channel, PeerUser, PeerChannel
from telethon.tl.types import InputPeerUser, InputPeerChannel
from telethon.errors import UserIdInvalidError
from telethon.tl.types import PeerChannel, PeerChat
from telethon.tl.types import User, Chat, Channel
from telethon.sync import TelegramClient
from enum import Enum, auto
from typing import Dict, Callable, Any
import ast
import json

import requests  # сделано больше для того, чтобы использовать другие функции запроса к ллм

from db import DBManager, Message as DBMessage, Reaction, Media
from db import DBManager, User as db_User, Message
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
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

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


class CommandState(Enum):
    IDLE = auto()
    WAITING_FOR_INPUT = auto()


class CommandHandler:
    def __init__(self):
        self.user_states: Dict[int, CommandState] = {}
        self.user_contexts: Dict[int, Any] = {}
        self.command_callbacks: Dict[str, Callable] = {}

    def register_command(self, command: str, callback: Callable):
        self.command_callbacks[command] = callback

    async def handle_message(self, event):
        sender_id = event.sender_id
        message = event.message

        if message.text.startswith('/'):
            command = message.text.split()[0][1:]
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
        sender_id = event.sender_id
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


def query_gemini(prompt):
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


def check_format(string):
    # Регулярное выражение для основного формата
    main_pattern = r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - [a-zA-Z0-9_]+: .+$'

    # Регулярное выражение для поиска смайликов (эмодзи)
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        u"\U0001F700-\U0001F77F"  # Alchemical Symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols & Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)

    # Регулярное выражение для поиска ссылок
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    if not re.match(main_pattern, string):
        return False
    if emoji_pattern.search(string):
        return False
    if url_pattern.search(string):
        return False

    return True


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
            'current_function': 'claude',
            'claude': 'default_query_llm',
            'gemini': 'query_gemini'
        }
        with open(file_path, 'w') as f:
            json.dump(data, f)
    return data


def save_chat_functions(chat_id, functions):
    file_path = get_chat_functions_file(chat_id)
    with open(file_path, 'w') as f:
        json.dump(functions, f)


async def handle_update_prompt(event, context):
    sender_id = event.sender_id

    if event.message.text == '/cancel':
        command_handler.reset_user_state(sender_id)
        await event.reply("Operation cancelled.")
        return

    if 'step' not in context:
        context['step'] = 'prompt_naming'
        await event.reply("Enter the name of the prompt you want to update (or /cancel to abort):")
    elif context['step'] == 'prompt_naming':
        prompt_name = event.message.text.strip().upper()
        context['prompt_name'] = prompt_name
        context['step'] = 'waiting_for_prompt'
        await event.reply(f"Please send the new content for the prompt '{prompt_name}' (or /cancel to abort).")
    elif context['step'] == 'waiting_for_prompt':
        new_prompt = event.message.text
        chat_id = context['chat_id']
        chat_prompts = load_chat_prompts(chat_id)
        chat_prompts[context['prompt_name']] = new_prompt
        save_chat_prompts(chat_id, chat_prompts)
        await event.reply(f"Prompt '{context['prompt_name']}' has been updated.")
        command_handler.reset_user_state(sender_id)


command_handler.register_command('update_prompt', handle_update_prompt)


@client.on(events.NewMessage(
    func=lambda e: e.sender_id in updating_users and updating_users[e.sender_id]['state'] == 'prompt_naming'))
async def handle_prompt_name(event):
    sender = await event.get_sender()
    if sender.id not in ongoing_dialogues:
        return
    chat_id = updating_users[sender.id]['chat_id']
    prompt_name = event.message.text.strip().upper()

    if prompt_name == '/cancel':
        ongoing_dialogues.remove(sender.id)
        del updating_users[sender.id]
        await event.reply("Действие отменено.")
        return

    if prompt_name == '/UPDATE_PROMPT':
        return

    chat_prompts = load_chat_prompts(chat_id)
    if prompt_name not in chat_prompts:
        await event.reply(f"Промпт '{prompt_name}' не найден. Пожалуйста, выберите правильное имя промпта.")
        return

    updating_users[sender.id]['prompt_name'] = prompt_name
    updating_users[sender.id]['state'] = 'waiting_for_prompt'
    await event.reply(f"Пожалуйста, пришлите новое содержимое для промпта '{prompt_name}'.")


@client.on(events.NewMessage(
    func=lambda e: e.sender_id in updating_users and updating_users[e.sender_id]['state'] == 'waiting_for_prompt'))
async def handle_new_prompt(event):
    sender = await event.get_sender()
    if sender.id not in ongoing_dialogues:
        return
    if event.message.text in ['ASK_PROMPT', 'SUMMARIZE_PROMPT']:
        return
    if event.message.text == '/cancel':
        ongoing_dialogues.remove(sender.id)
        del updating_users[sender.id]
        await event.reply("Действие отменено.")
        return
    chat_id = updating_users[sender.id]['chat_id']
    prompt_name = updating_users[sender.id]['prompt_name']
    new_prompt = event.message.text

    chat_prompts = load_chat_prompts(chat_id)
    chat_prompts[prompt_name] = new_prompt
    save_chat_prompts(chat_id, chat_prompts)

    await event.reply(f"Промпт '{prompt_name}' был успешно обновлен для этого чата.")
    ongoing_dialogues.remove(sender.id)
    del updating_users[sender.id]


async def handle_update_limit(event, context):
    sender_id = event.sender_id

    if event.message.text == '/cancel':
        command_handler.reset_user_state(sender_id)
        await event.reply("Operation cancelled.")
        return

    if 'step' not in context:
        context['step'] = 'limit_naming'
        await event.reply("Enter the name of the limit you want to update (or /cancel to abort):")
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


command_handler.register_command('update_limit', handle_update_limit)


@client.on(events.NewMessage(
    func=lambda e: e.sender_id in updating_users and updating_users[e.sender_id]['state'] == 'limit_naming'))
async def handle_limit_name(event):
    sender = await event.get_sender()
    chat_id = updating_users[sender.id]['chat_id']
    limit_name = event.message.text.strip().lower()

    if limit_name == '/cancel':
        ongoing_dialogues.remove(sender.id)
        del updating_users[sender.id]
        await event.reply("Действие отменено.")
        return

    if limit_name == '/update_limit':
        return

    chat_limits = load_chat_limits(chat_id)
    if limit_name not in chat_limits:
        await event.reply(f"Лимит '{limit_name}' не найден. Пожалуйста, выберите правильное имя лимита.")
        return

    updating_users[sender.id]['limit_name'] = limit_name
    updating_users[sender.id]['state'] = 'waiting_for_limit'
    await event.reply(
        f"Пожалуйста, пришлите новое значение для лимита '{limit_name}' (должно быть целое положительное число).")


@client.on(events.NewMessage(
    func=lambda e: e.sender_id in updating_users and updating_users[e.sender_id]['state'] == 'waiting_for_limit'))
async def handle_new_limit(event):
    sender = await event.get_sender()
    if sender.id not in ongoing_dialogues:
        return
    if event.message.text == '/cancel':
        ongoing_dialogues.remove(sender.id)
        del updating_users[sender.id]
        await event.reply("Действие отменено.")
        return
    chat_id = updating_users[sender.id]['chat_id']
    limit_name = updating_users[sender.id]['limit_name']

    try:
        new_limit = int(event.message.text)
        if new_limit <= 0:
            raise ValueError
    except ValueError:
        await event.reply("Неверный ввод. Пожалуйста, введите целое положительное число.")
        return

    chat_limits = load_chat_limits(chat_id)
    chat_limits[limit_name] = new_limit
    save_chat_limits(chat_id, chat_limits)

    await event.reply(f"Лимит '{limit_name}' был успешно обновлен до {new_limit} для этого чата.")
    ongoing_dialogues.remove(sender.id)
    del updating_users[sender.id]


@client.on(events.NewMessage(pattern='/cancel'))
async def cancel_dialogue(event):
    sender_id = event.sender_id
    if sender_id in ongoing_dialogues:
        ongoing_dialogues.remove(sender_id)
        if sender_id in updating_users:
            del updating_users[sender_id]
        # await event.reply("Current operation has been cancelled.")
    # else:
    # await event.reply("There's no ongoing operation to cancel.")


async def handle_update_query_llm(event, context):
    sender_id = event.sender_id

    if event.message.text == '/cancel':
        command_handler.reset_user_state(sender_id)
        await event.reply("Operation cancelled.")
        return

    if 'step' not in context:
        context['step'] = 'naming'
        await event.reply("Please enter a name for the new function (cannot be 'default', or /cancel to abort):")
    elif context['step'] == 'naming':
        name = event.message.text.strip()
        if name.lower() == 'default':
            await event.reply(
                "'default' cannot be used as a function name. Please choose another name (or /cancel to abort):")
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


command_handler.register_command('update_query_llm', handle_update_query_llm)


@client.on(events.NewMessage(
    func=lambda e: e.sender_id in updating_users and updating_users[e.sender_id]['state'] == 'naming'))
async def handle_function_name(event):
    sender = await event.get_sender()
    chat_id = updating_users[sender.id]['chat_id']
    name = event.message.text.strip()

    if sender.id not in ongoing_dialogues:
        return

    if name == '/update_query_llm':
        return
    if name == '/cancel':
        ongoing_dialogues.remove(sender.id)
        del updating_users[sender.id]
        await event.reply("Действие отменено.")
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
    if sender.id not in ongoing_dialogues:
        return
    chat_id = updating_users[sender.id]['chat_id']
    name = updating_users[sender.id]['name']

    if event.message.text == '/cancel':
        ongoing_dialogues.remove(sender.id)
        del updating_users[sender.id]
        await event.reply("Действие отменено.")
        return

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
        ongoing_dialogues.remove(sender.id)


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
    sender = await event.get_sender()
    ongoing_dialogues.add(sender.id)
    chat_id = event.chat_id
    chat_functions = load_chat_functions(chat_id)
    function_list = "Выберите функцию для использования:\n0. default (встроенная)\n" + "\n".join(
        f"{i + 1}. {name}" for i, name in enumerate(chat_functions) if name != 'current_function')
    await event.reply(function_list + "\n\nОтветьте номером функции, которую вы хотите использовать.")
    updating_users[event.sender_id] = {'chat_id': chat_id, 'state': 'selecting_function'}


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


@client.on(events.NewMessage(
    func=lambda e: e.sender_id in updating_users and updating_users[e.sender_id]['state'] == 'selecting_function'))
async def handle_function_selection(event):
    sender = await event.get_sender()
    chat_id = updating_users[sender.id]['chat_id']
    chat_functions = load_chat_functions(chat_id)
    if sender.id not in ongoing_dialogues:
        return
    if event.message.text == '/cancel':
        ongoing_dialogues.remove(sender.id)
        del updating_users[sender.id]
        await event.reply("Действие отменено.")
        return
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
        ongoing_dialogues.remove(sender.id)


async def get_query_llm(chat_id):
    chat_functions = load_chat_functions(chat_id)
    current_function = chat_functions.get('current_function', 'claude')

    if current_function == 'claude':
        return default_query_llm
    elif current_function == 'gemini':
        return query_gemini
    else:
        function_code = chat_functions.get(current_function)
        if function_code:
            exec(function_code)
            return locals()['query_llm']
        else:
            logging.warning(
                f"Function '{current_function}' not found for chat {chat_id}. Using Claude as default.")
            return default_query_llm


# Add a new command to switch between Claude and Gemini
@client.on(events.NewMessage(pattern='/switch_llm'))
async def switch_llm(event):
    chat_id = event.chat_id
    chat_functions = load_chat_functions(chat_id)
    current_function = chat_functions.get('current_function', 'claude')

    new_function = 'gemini' if current_function == 'claude' else 'claude'
    chat_functions['current_function'] = new_function

    with open(get_chat_functions_file(chat_id), 'w') as f:
        json.dump(chat_functions, f)

    await event.reply(f"Switched to {new_function.capitalize()} for LLM queries.")


async def get_chat_messages(chat_id: int, date: str):
    db = DBManager.get_db(chat_id)
    query = f"""
    SELECT m.id, m.date, u.first_name, m.content, COUNT(r.id) as reaction_count
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


async def get_last_messages(chat_id: int, limit: int):
    db = DBManager.get_db(chat_id)
    # query = f"""
    # SELECT m.id, m.date, u.first_name, m.content
    # FROM messages m
    # JOIN users u ON m.user_id = u.id
    # ORDER BY m.date DESC
    # LIMIT {limit}
    # """
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

    prompt = chat_prompts['ASK_PROMPT'].format(question=question)
    for _, row in df.iterrows():
        string = f"{row['date']} - {row['first_name']}: {row['content']}\n"
        if check_format(string):
            prompt += string
    data_as_string = df.to_string(index=False)
    prompt += f"\n {data_as_string}"
    query_llm = await get_query_llm(chat_id)
    answer = query_llm(prompt)
    print(prompt)
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


@client.on(events.NewMessage(pattern=r'/ask'))
async def handle_ask(event):
    chat_id = event.chat_id
    message = event.message
    db = DBManager.get_db(chat_id)

    # Check if there's a reply or a user mention
    replied_to = await message.get_reply_message()
    username_match = re.search(r'@(\w+)', message.text)

    if replied_to or username_match:
        # Use user_info functionality
        response = await get_user_info(event, db)
    else:
        # Use original ask functionality
        question = re.sub(r'^/ask(@\w+)?(\s+)?', '', message.text).strip()
        if not question:
            response = "Please provide a question after the /ask command."
        else:
            response = await ask_question(chat_id, question)

    await event.reply(response)


@client.on(events.NewMessage)
async def handler(event: events.NewMessage.Event):
    """Обработчик новых сообщений."""
    message: Message = event.message
    chat_id = event.chat_id

    db = DBManager.get_db(chat_id)

    # Получите информацию о пользователе
    user = await _get_user(event)

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


@client.on(events.NewMessage(pattern='/update_prompt --help'))
async def update_prompt_help(event):
    help_message = help_texts['UPDATE_PROMPT_HELP']
    await event.reply(help_message)


@client.on(events.NewMessage(pattern='/update_limit --help'))
async def update_limit_help(event):
    help_message = help_texts['UPDATE_LIMIT_HELP']
    await event.reply(help_message)


@client.on(events.NewMessage(pattern='/list_prompts'))
async def list_prompts(event):
    chat_id = event.chat_id
    chat_prompts = load_chat_prompts(chat_id)
    prompts_list = "\n".join(f"- {name}" for name in chat_prompts.keys())
    message = f"""Available prompts for this chat:

{prompts_list}

Use /update_prompt to modify a prompt."""
    await event.reply(message)


@client.on(events.NewMessage(pattern='/list_limits'))
async def list_limits(event):
    chat_id = event.chat_id
    chat_limits = load_chat_limits(chat_id)
    limits_list = "\n".join(f"- {name}: {value}" for name, value in chat_limits.items())
    message = f"""Current message load limits for this chat:

{limits_list}

Use /update_limit to modify a limit."""
    await event.reply(message)


@client.on(events.NewMessage(pattern='/summarize'))
async def summarize(event):
    chat_id = event.chat_id
    chat_prompts = load_chat_prompts(chat_id)
    chat_limits = load_chat_limits(chat_id)
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    dates = re.findall(date_pattern, event.message.text)

    if len(dates) == 2:
        start_date = dates[0]
        end_date = dates[1]
    elif len(dates) == 1:
        start_date = end_date = dates[0]
    else:
        current_date = datetime.now().date()
        start_date = end_date = current_date

    df = await get_chat_messages_between_dates(chat_id, start_date, end_date, chat_limits['summarize_limit'])

    if df.empty:
        await event.reply("Не найдено ни одного сообщения в указанном диапазоне времени.")
        return

    prompt = chat_prompts['SUMMARIZE_PROMPT']

    prompt += df.to_string(index=False)
    # for _, row in df.iterrows():
    #     string = f"{row['date']} - {row['username']}: {row['content']}\n"
    #     if check_format(string):
    #         prompt += string

    query_llm = await get_query_llm(chat_id)
    summary = query_llm(prompt)
    print(prompt)
    await event.reply(summary)


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


async def get_user_info(event, db):
    chat_id = event.chat_id
    message = event.message
    replied_to = await message.get_reply_message()

    # Extract username if provided
    username_match = re.search(r'@(\w+)', message.text)
    username = username_match.group(1) if username_match else None

    # Get user from reply or username
    if replied_to:
        user = await client.get_entity(replied_to.from_id)
        user_id = user.id
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

    Инструкции:
    1. Внимательно проанализируйте предоставленные сообщения пользователя.
    2. Определите ключевые характеристики пользователя, включая:
       - Основные темы и интересы
       - Стиль общения и язык
       - Эмоциональный тон сообщений
       - Уровень активности и вовлеченности в обсуждения
       - Отношение к другим участникам чата
       - Уровень экспертизы в обсуждаемых темах
    3. Для каждой выявленной характеристики приведите конкретный пример из сообщений пользователя.
    4. Если возможно, определите роль пользователя в чате (например, лидер мнений, эксперт, новичок, троллЬ и т.д.).
    5. Укажите любые заметные изменения в поведении или интересах пользователя со временем.
    6. Если задан конкретный вопрос, ответьте на него, основываясь на проведенном анализе.

    Сообщения пользователя для анализа:
    {messages_text}

    Вопрос (если есть): {question}

    Формат ответа:
    1. Краткое общее описание пользователя (2-3 предложения)
    2. Подробный анализ по каждому пункту инструкций с примерами
    3. Ответ на конкретный вопрос (если задан)

    Важно: Все утверждения должны быть подкреплены примерами из сообщений пользователя. Избегайте необоснованных предположений.
    """

    # Query LLM
    query_llm = await get_query_llm(chat_id)
    answer = query_llm(prompt)
    print(prompt)

    return answer


@client.on(events.NewMessage(pattern='/user_info'))
async def handle_user_info(event):
    chat_id = event.chat_id
    db = DBManager.get_db(chat_id)

    response = await get_user_info(event, db)
    await event.reply(response)


async def _get_user(event) -> db_User:
    """Получите информацию о пользователе или канале из Telegram и сохраните ее в базе данных."""
    db = DBManager.get_db(event.chat_id)
    try:
        # Check if the sender is a channel or a chat
        if isinstance(event.message.to_id, PeerChannel) or isinstance(event.message.to_id, PeerChat):
            # It's a group or channel message
            entity = await event.get_chat()
        else:
            # It's a user message
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
        return db_User(id=event.sender_id, username=str(event.sender_id), first_name=str(event.sender_id), last_name=None, tags="", avatar=None)


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


@client.on(events.NewMessage)
async def handler(event):
    await command_handler.handle_message(event)


async def main():
    """Запустите бота и ждите новых сообщений."""
    logging.info("Бот запущен!")
    command_handler.register_command('update_query_llm', handle_update_query_llm)
    command_handler.register_command('update_prompt', handle_update_prompt)
    command_handler.register_command('update_limit', handle_update_limit)
    await client.run_until_disconnected()


if __name__ == '__main__':
    with client:
        client.loop.run_until_complete(main())
