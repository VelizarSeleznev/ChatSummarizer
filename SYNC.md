
### Конфигурационный файл `config.yaml`

```yaml
api_id: "YOUR_API_ID"
api_hash: "YOUR_API_HASH"
group: "YOUR_GROUP_NAME_OR_ID"
download_avatars: true
avatar_size: [64, 64]
download_media: false
media_dir: "media"
media_mime_types: []
proxy:
  enable: false
fetch_batch_size: 2000
fetch_wait: 5
fetch_limit: 0
sync_reactions: true
download_custom_emojis: true
timezone: "UTC"
```

### Проверка функционала

1. **Создайте конфигурационный файл `config.yaml`** с вашими параметрами.
2. **Запустите скрипт для различных режимов синхронизации** и убедитесь, что данные корректно синхронизируются в базу данных SQLite.

### Пример использования

1. **Простая синхронизация:**
   ```sh
   python sync.py
   ```

2. **Синхронизация с указанного сообщения:**
   ```sh
   python sync.py --start-id 12345
   ```

3. **Синхронизация последних 1000 сообщений:**
   ```sh
   python sync.py --last-1000
   ```

Если у вас возникнут вопросы или потребуется дополнительная помощь, пожалуйста, дайте знать!