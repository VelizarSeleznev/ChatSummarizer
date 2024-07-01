# Telegram ChatSummarizer

ChatSummarizer is a Telegram bot powered by Google's Gemini AI, designed to summarize and analyze conversations, as well as answer questions based on chat history.

## Features

- Summarize recent conversations
- Analyze user behavior and communication patterns
- Answer questions based on chat context
- Store message history in a database

## Prerequisites

- Python 3.7+
- Telegram Bot API credentials
- Google Gemini API key
- SQLAlchemy compatible database (default: SQLite)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/VelizarSeleznev/ChatSummarizer.git
   cd ChatSummarizer
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables in a `.env` file:
   ```
   TG_API_ID=<your_telegram_api_id>
   TG_API_HASH=<your_telegram_api_hash>
   TELEGRAM_BOT_TOKEN=<your_telegram_bot_token>
   GOOGLE_API_KEY=<your_google_gemini_api_key>
   DATABASE_URL=<your_database_url>  # Optional, defaults to SQLite
   ```

## Usage

Run the bot:
```
python main.py
```

### Available Commands

- `/start`: Introduces the bot
- `/summarize`: Summarizes recent conversations
- `/analyze @username`: Analyzes a user's communication patterns
- `/ask [question]`: Answers questions based on chat context

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.