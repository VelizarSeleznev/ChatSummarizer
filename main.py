import time
import random
from collections import defaultdict, Counter
from typing import List, Dict


# Имитация Telegram API
class TelegramAPI:
    def get_new_messages(self) -> List[Dict]:
        # В реальности здесь был бы код для получения сообщений из Telegram
        return [
            {"author": f"user{random.randint(1, 5)}", "text": f"Message {random.randint(1, 100)}"}
            for _ in range(random.randint(1, 10))
        ]


# Имитация Gemini AI
class GeminiAI:
    def generate_commentary(self, context: str) -> str:
        # В реальности здесь был бы запрос к Gemini AI
        return f"Exciting commentary based on: {context}"


class TelegramSummarizer:
    def __init__(self):
        self.telegram_api = TelegramAPI()
        self.gemini_ai = GeminiAI()
        self.active_participants = defaultdict(lambda: {"messages": 0, "last_active": 0})
        self.topics = defaultdict(int)
        self.significant_events = []
        self.last_summary_time = time.time()
        self.messages_since_last_summary = []
        self.sentiment_scores = []

    def run(self):
        while True:
            # Получаем новые сообщения из Telegram
            new_messages = self.telegram_api.get_new_messages()
            # Анализируем полученные сообщения
            self.analyze_messages(new_messages)

            # Проверяем, нужно ли генерировать новый комментарий
            if self.should_generate_commentary():
                # Генерируем комментарий в спортивном стиле
                commentary = self.generate_sports_commentary()
                # Отправляем комментарий
                self.send_commentary(commentary)
                # Обновляем информацию о последней сводке
                self.update_last_summary()

            # Задержка между проверками
            time.sleep(5)

    def analyze_messages(self, messages: List[Dict]):
        for message in messages:
            # Обновляем информацию об активных участниках
            self.update_active_participants(message['author'])
            # Анализируем содержание сообщения
            self.analyze_message_content(message['text'])
            # Добавляем сообщение в список сообщений с момента последней сводки
            self.messages_since_last_summary.append(message)
            # Определяем значимые события
            self.detect_significant_events(message)

    def update_active_participants(self, author: str):
        # Увеличиваем счетчик сообщений участника
        self.active_participants[author]["messages"] += 1
        # Обновляем время последней активности участника
        self.active_participants[author]["last_active"] = time.time()

    def analyze_message_content(self, text: str):
        # Разбиваем текст на слова и обновляем счетчики тем
        words = text.lower().split()
        self.topics.update(words)
        # Анализируем настроение сообщения
        sentiment = self.analyze_sentiment(text)
        # Добавляем результат анализа настроения в список
        self.sentiment_scores.append(sentiment)

    def analyze_sentiment(self, text: str) -> float:
        # Упрощенный анализ настроения
        positive_words = set(['good', 'great', 'awesome', 'nice', 'cool'])
        negative_words = set(['bad', 'awful', 'terrible', 'sucks', 'shit'])
        words = set(text.lower().split())
        # Вычисляем оценку настроения
        score = (len(words & positive_words) - len(words & negative_words)) / len(words)
        return max(-1.0, min(1.0, score))  # Нормализуем от -1 до 1

    def detect_significant_events(self, message: Dict):
        text = message['text'].lower()
        # Определяем эмоциональные сообщения
        if '!' in text or '?' in text:
            self.significant_events.append(f"Эмоциональное сообщение от {message['author']}")
        # Определяем длинные сообщения
        if len(text.split()) > 20:
            self.significant_events.append(f"Длинное сообщение от {message['author']}")
        # Определяем важные объявления
        if any(word in text for word in ['важно', 'срочно', 'внимание']):
            self.significant_events.append(f"Важное объявление от {message['author']}")

    def should_generate_commentary(self) -> bool:
        # Пороговые значения для генерации комментария
        time_threshold = 300  # 5 минут
        message_threshold = 50
        event_threshold = 3
        # Проверяем, нужно ли генерировать комментарий
        return (
                time.time() - self.last_summary_time > time_threshold or
                len(self.messages_since_last_summary) > message_threshold or
                len(self.significant_events) >= event_threshold
        )

    def generate_sports_commentary(self) -> str:
        # Подготавливаем контекст для генерации комментария
        context = self.prepare_context()
        # Генерируем комментарий с помощью AI
        return self.gemini_ai.generate_commentary(context)

    def prepare_context(self) -> str:
        # Определяем топ участников
        top_participants = sorted(
            self.active_participants.items(),
            key=lambda x: (x[1]['messages'], -x[1]['last_active']),
            reverse=True
        )[:5]
        # Определяем топ темы
        top_topics = Counter(self.topics).most_common(5)
        # Вычисляем среднее настроение
        avg_sentiment = sum(self.sentiment_scores) / len(self.sentiment_scores) if self.sentiment_scores else 0

        # Формируем контекст для AI
        context = f"Топ игроки: {[p[0] for p in top_participants]}. "
        context += f"Горячие темы: {[t[0] for t in top_topics]}. "
        context += f"Настроение чата: {'позитивное' if avg_sentiment > 0 else 'негативное' if avg_sentiment < 0 else 'нейтральное'}. "
        context += f"Ключевые моменты: {self.significant_events[:3]}"
        return context

    def send_commentary(self, commentary: str):
        # Выводим комментарий (в реальности здесь был бы код для отправки комментария в Telegram)
        print(f"Новый комментарий от спортивного обозревателя:\n{commentary}")

    def update_last_summary(self):
        # Обновляем время последней сводки
        self.last_summary_time = time.time()
        # Очищаем списки сообщений и значимых событий
        self.messages_since_last_summary.clear()
        self.significant_events.clear()
        self.sentiment_scores.clear()


if __name__ == "__main__":
    summarizer = TelegramSummarizer()
    summarizer.run()