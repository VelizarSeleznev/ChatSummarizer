import os
import logging
import yaml
import argparse
import time
import asyncio
from telethon import TelegramClient, errors
from telethon.tl import types, functions
from db import DB, User, Message, Media, Reaction
import pytz
from io import BytesIO
from PIL import Image
import shutil
import tempfile
import json

# Загрузка конфигурации
def get_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

class Sync:
    def __init__(self, config, session_file, db):
        self.config = config
        self.db = db

    async def start_client(self, session, config):
        if "proxy" in config and config["proxy"].get("enable"):
            proxy = config["proxy"]
            self.client = TelegramClient(session, config["api_id"], config["api_hash"], proxy=(proxy["protocol"], proxy["addr"], proxy["port"]))
        else:
            self.client = TelegramClient(session, config["api_id"], config["api_hash"])
        
        await self.client.start()

    async def sync(self, ids=None, from_id=None):
        if not self.client.is_connected():
            await self.client.connect()

        if ids:
            last_id, last_date = (ids, None)
            logging.info("fetching message id={}".format(ids))
        elif from_id:
            last_id, last_date = (from_id, None)
            logging.info("fetching from last message id={}".format(last_id))
        else:
            last_id, last_date = self.db.get_last_message_id()
            logging.info("fetching from last message id={} ({})".format(last_id, last_date))

        group_id = await self._get_group_id(self.config["group"])

        n = 0
        while True:
            has = False
            async for m in self._get_messages(group_id, offset_id=last_id if last_id else 0, ids=ids):
                if not m:
                    continue

                has = True

                self.db.insert_user(m.user)

                if m.media:
                    self.db.insert_media(m.media)

                self.db.insert_message(m)

                last_date = m.date
                n += 1
                if n % 300 == 0:
                    logging.info("fetched {} messages".format(n))
                    self.db.commit()

                if 0 < self.config["fetch_limit"] <= n or ids:
                    has = False
                    break

            self.db.commit()
            if has:
                last_id = m.id
                logging.info("fetched {} messages. sleeping for {} seconds".format(n, self.config["fetch_wait"]))
                await asyncio.sleep(self.config["fetch_wait"])
            else:
                break

        self.db.commit()
        logging.info("finished. fetched {} messages. last message = {}".format(n, last_date))

    async def _get_messages(self, group, offset_id, ids=None):
        messages = await self._fetch_messages(group, offset_id, ids)
        for m in messages:
            if not m or not m.sender:
                continue

            logging.info("Message #{}".format(m.id))

            sticker = None
            med = None
            if m.media:
                if isinstance(m.media, types.MessageMediaDocument) and hasattr(m.media, "document") and m.media.document.mime_type == "application/x-tgsticker":
                    alt = [a.alt for a in m.media.document.attributes if isinstance(a, types.DocumentAttributeSticker)]
                    if len(alt) > 0:
                        sticker = alt[0]
                elif isinstance(m.media, types.MessageMediaPoll):
                    med = self._make_poll(m)
                else:
                    med = await self._get_media(m)

            typ = "message"
            if m.action:
                if isinstance(m.action, types.MessageActionChatAddUser):
                    typ = "user_joined"
                elif isinstance(m.action, types.MessageActionChatDeleteUser):
                    typ = "user_left"

            if m.reactions and self.config.get("sync_reactions", False):
                reactions = await self._fetch_reactions(m)
                for r in reactions:
                    self.db.insert_reaction(r)

            yield Message(
                type=typ,
                id=m.id,
                date=m.date,
                edit_date=m.edit_date,
                content=sticker if sticker else m.raw_text,
                reply_to=m.reply_to_msg_id if m.reply_to and m.reply_to.reply_to_msg_id else None,
                user=await self._get_user(m.sender),
                media=med
            )

    async def _fetch_messages(self, group, offset_id, ids=None):
        try:
            messages = await self.client.get_messages(group, offset_id=offset_id, limit=self.config["fetch_batch_size"], ids=ids, reverse=True)
            return messages
        except errors.FloodWaitError as e:
            logging.info("flood waited: have to wait {} seconds".format(e.seconds))
            await asyncio.sleep(e.seconds)
            return await self._fetch_messages(group, offset_id, ids)

    async def _fetch_reactions(self, message):
        reactions = []
        try:
            request = functions.messages.GetMessageReactionsListRequest(peer=message.peer_id, id=message.id, limit=1000)
            message_reactions = await self.client(request)
            for reaction in message_reactions.reactions:
                input_entity = await self.client.get_input_entity(reaction.peer_id)
                reaction_user = await self.client.get_entity(input_entity)
                self.db.insert_user(await self._get_user(reaction_user))
                reaction_user_id = reaction_user.id

                reaction_content = ''
                reaction_type = 'unknown'

                if isinstance(reaction.reaction, types.ReactionEmpty):
                    reaction_type = 'empty'
                elif isinstance(reaction.reaction, types.ReactionEmoji):
                    reaction_type = 'emoticon'
                    reaction_content = reaction.reaction.emoticon
                elif isinstance(reaction.reaction, types.ReactionCustomEmoji):
                    reaction_type = 'custom'
                    reaction_content = str(reaction.reaction.document_id)
                    if self.config.get("download_custom_emojis", False):
                        await self._get_custom_emoji(reaction.reaction.document_id)

                reactions.append(Reaction(
                    id='',
                    date=reaction.date,
                    message_id=message.id,
                    user_id=reaction_user_id,
                    type=reaction_type,
                    content=reaction_content
                ))
        except Exception as e:
            logging.error(f"Error fetching reactions for message {message.id}: {str(e)}")
        return reactions

    async def _get_custom_emoji(self, document_id):
        emoji_fname = "custom_emoji_{}".format(document_id)
        emoji_fpath = os.path.join(self.config["media_dir"], emoji_fname)

        if os.path.exists(emoji_fpath + ".webm"):
            logging.info("   skip downloading emoji {}".format(emoji_fname + ".webm"))
            return
        if os.path.exists(emoji_fpath + ".webp"):
            logging.info("   skip downloading emoji {}".format(emoji_fname + ".webp"))
            return
        if os.path.exists(emoji_fpath + ".tgs"):
            logging.info("   skip downloading emoji {}".format(emoji_fname + ".tgs"))
            return
    
        logging.info("   downloading emoji {}".format(document_id))
        request = functions.messages.GetCustomEmojiDocumentsRequest(
            document_id=[document_id]
        )
        custom_emoji_documents = await self.client(request)
        for document in custom_emoji_documents:
            for attribute in document.attributes:
                if isinstance(attribute, types.DocumentAttributeCustomEmoji):
                    self.db.insert_custom_emoji(document_id, attribute.alt)
                    custom_emoji_path = await self._download_custom_emoji(
                        attribute.stickerset,
                        attribute.alt
                    )
                    if custom_emoji_path:
                        basename = os.path.basename(custom_emoji_path)
                        newname = "custom_emoji_{}.{}".format(document_id, self._get_file_ext(basename))
                        shutil.move(custom_emoji_path, os.path.join(self.config["media_dir"], newname))
                    else:
                        logging.warning(f"Failed to download custom emoji {document_id}")
        return custom_emoji_documents
                    
    async def _download_custom_emoji(self, stickerset, emoji_alt):
        sticker_set = await self.client(functions.messages.GetStickerSetRequest(
            stickerset=stickerset,
            hash=0
        ))
        for document in sticker_set.documents:
            for attribute in document.attributes:
                if hasattr(attribute, 'alt') and attribute.alt == emoji_alt:
                    try:
                        fpath = await self.client.download_media(document, file=tempfile.gettempdir())
                        return fpath
                    except Exception as e:
                        logging.error(f"Error downloading custom emoji: {e}")
                        return None
        logging.warning(f"Custom emoji not found in sticker set: {emoji_alt}")
        return None

    async def _get_user(self, u):
        tags = []
        is_normal_user = isinstance(u, types.User)

        if isinstance(u, types.ChannelForbidden):
            return User(
                id=u.id,
                username=u.title,
                first_name=None,
                last_name=None,
                tags=tags,
                avatar=None
            )

        if is_normal_user:
            if u.bot:
                tags.append("bot")

        if u.scam:
            tags.append("scam")

        if u.fake:
            tags.append("fake")

        avatar = None
        if self.config["download_avatars"]:
            try:
                avatar = await self._download_avatar(u)
            except Exception as e:
                logging.error("error downloading avatar: #{}: {}".format(u.id, e))

        return User(
            id=u.id,
            username=u.username if u.username else str(u.id),
            first_name=u.first_name if is_normal_user else None,
            last_name=u.last_name if is_normal_user else None,
            tags=','.join(tags),
            avatar=avatar
        )

    def _make_poll(self, msg):
        if not msg.media.results or not msg.media.results.results:
            return None

        options = [{"label": a.text, "count": 0, "correct": False} for a in msg.media.poll.answers]

        total = msg.media.results.total_voters
        if msg.media.results.results:
            for i, r in enumerate(msg.media.results.results):
                options[i]["count"] = r.voters
                options[i]["percent"] = r.voters / total * 100 if total > 0 else 0
                options[i]["correct"] = r.correct

        return Media(
            id=msg.id,
            type="poll",
            url=None,
            title=msg.media.poll.question,
            description=json.dumps(options),
            thumb=None
        )

    async def _get_media(self, msg):
        if isinstance(msg.media, types.MessageMediaWebPage) and not isinstance(msg.media.webpage, types.WebPageEmpty):
            return Media(
                id=msg.id,
                type="webpage",
                url=msg.media.webpage.url,
                title=msg.media.webpage.title,
                description=msg.media.webpage.description if msg.media.webpage.description else None,
                thumb=None
            )
        elif isinstance(msg.media, types.MessageMediaPhoto) or isinstance(msg.media, types.MessageMediaDocument) or isinstance(msg.media, types.MessageMediaContact):
            if self.config["download_media"]:
                if len(self.config["media_mime_types"]) > 0:
                    if hasattr(msg, "file") and hasattr(msg.file, "mime_type") and msg.file.mime_type:
                        if msg.file.mime_type not in self.config["media_mime_types"]:
                            logging.info("skipping media #{} / {}".format(msg.file.name, msg.file.mime_type))
                            return

                logging.info("downloading media #{}".format(msg.id))
                try:
                    basename, fname, thumb = await self._download_media(msg)
                    return Media(
                        id=msg.id,
                        type="photo",
                        url=fname,
                        title=basename,
                        description=None,
                        thumb=thumb
                    )
                except Exception as e:
                    logging.error("error downloading media: #{}: {}".format(msg.id, e))

    async def _download_media(self, msg):
        fpath = await self.client.download_media(msg, file=tempfile.gettempdir())
        basename = os.path.basename(fpath)

        newname = "{}.{}".format(msg.id, self._get_file_ext(basename))
        shutil.move(fpath, os.path.join(self.config["media_dir"], newname))

        tname = None
        if isinstance(msg.media, types.MessageMediaPhoto):
            tpath = await self.client.download_media(msg, file=tempfile.gettempdir(), thumb=1)
            tname = "thumb_{}.{}".format(msg.id, self._get_file_ext(os.path.basename(tpath)))
            shutil.move(tpath, os.path.join(self.config["media_dir"], tname))

        return basename, newname, tname

    def _get_file_ext(self, f):
        if "." in f:
            e = f.split(".")[-1]
            if len(e) < 6:
                return e
        return ".file"

    async def _download_avatar(self, user):
        fname = "avatar_{}.jpg".format(user.id)
        fpath = os.path.join(self.config["media_dir"], fname)

        if os.path.exists(fpath):
            return fname

        logging.info("downloading avatar #{}".format(user.id))

        b = BytesIO()
        profile_photo = await self.client.download_profile_photo(user, file=b)
        if profile_photo is None:
            logging.info("user has no avatar #{}".format(user.id))
            im = Image.new('RGB', self.config["avatar_size"], (25, 25, 25))
            im.save(fpath, "JPEG")
            return fname

        im = Image.open(b)
        im.thumbnail(self.config["avatar_size"], Image.LANCZOS)
        im.save(fpath, "JPEG")

        return fname

    async def _get_group_id(self, group):
        await self.client.get_dialogs()

        try:
            group = int(group)
        except ValueError:
            pass

        try:
            entity = await self.client.get_entity(group)
        except ValueError:
            logging.critical("the group: {} does not exist, or the authorized user is not a participant!".format(group))
            exit(1)

        return entity.id

# Основная функция
async def main():
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)

    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Sync Telegram data to SQLite")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument("-d", "--data", type=str, default="data.sqlite", help="Path to the SQLite data file")
    parser.add_argument("-s", "--session", type=str, default="session.session", help="Path to the session file")
    parser.add_argument("--start-id", type=int, help="Start message ID for syncing")
    parser.add_argument("--last-1000", action="store_true", help="Sync the last 1000 messages to update reactions")
    args = parser.parse_args()

    # Загрузка конфигурации
    config = get_config(args.config)

    # Инициализация базы данных
    db = DB(args.data, config.get("timezone"))

    # Инициализация и запуск синхронизации
    sync = Sync(config, args.session, db)
    await sync.start_client(args.session, config)

    if args.last_1000:
        last_id, _ = db.get_last_message_id()
        await sync.sync(from_id=max(0, last_id - 1000))
    else:
        await sync.sync(from_id=args.start_id)

if __name__ == "__main__":
    asyncio.run(main())