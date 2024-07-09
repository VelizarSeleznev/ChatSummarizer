from telethon import TelegramClient, events
from telethon.tl.types import InputPeerUser
from telethon.tl.functions.users import GetFullUserRequest
import asyncio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import sqlite3
from dotenv import load_dotenv
import os
from db import DBManager


async def messages_by_day(event, bot):
    query = """
    SELECT DATE(date) as date, COUNT(*) as count
    FROM messages
    GROUP BY DATE(date)
    ORDER BY date
    """
    db = DBManager.get_db(event.chat_id)
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

    await bot.send_file(event.chat_id, buf, caption='Messages by Day')
    plt.close()


async def activity_by_hour(event, bot):
    query = """
    SELECT strftime('%H', date) as hour, COUNT(*) as count
    FROM messages
    GROUP BY hour
    ORDER BY hour
    """
    db = DBManager.get_db(event.chat_id)
    df = pd.read_sql_query(query, db.conn)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='hour', y='count', data=df)
    plt.title('Активность по времени суток')
    plt.xlabel('Час')
    plt.ylabel('Количество сообщений')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    await bot.send_file(event.chat_id, buf, caption='Activity by Hour')
    plt.close()


async def message_length_distribution(event, bot):
    query = "SELECT length(content) as length FROM messages"
    db = DBManager.get_db(event.chat_id)
    df = pd.read_sql_query(query, db.conn)

    plt.figure(figsize=(12, 6))
    sns.histplot(df['length'], kde=True)
    plt.title('Распределение длины сообщений')
    plt.xlabel('Длина сообщения')
    plt.ylabel('Частота')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    await bot.send_file(event.chat_id, buf, caption='Message Length Distribution')
    plt.close()


async def user_activity_comparison(event, bot):
    query = """
    SELECT u.username, COUNT(*) as message_count
    FROM messages m
    JOIN users u ON m.user_id = u.id
    GROUP BY u.id
    ORDER BY message_count DESC
    LIMIT 10
    """
    db = DBManager.get_db(event.chat_id)
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

    await bot.send_file(event.chat_id, buf, caption='User Activity Comparison')
    plt.close()


async def word_trend(event, bot):
    word = event.raw_text.split(maxsplit=1)[1] if len(event.raw_text.split()) > 1 else None

    if not word:
        await event.reply('Please provide a word to analyze. Usage: /word_trend [word]')
        return

    query = f"""
    SELECT DATE(date) as date, 
        SUM(CASE WHEN content LIKE '%{word}%' THEN 1 ELSE 0 END) as count
    FROM messages
    GROUP BY DATE(date)
    ORDER BY date
    """
    db = DBManager.get_db(event.chat_id)
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

    await bot.send_file(event.chat_id, buf, caption=f'Word Trend: "{word}"')
    plt.close()
