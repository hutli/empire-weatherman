import atexit
import datetime
import io
import json
import multiprocessing
import os
import pathlib
import re
import time

import bs4
import dotenv
import httpx
import loguru
import multiprocessing.sharedctypes
import mutagen.mp3
import pydantic
import pydub
import tqdm_loggable.auto  # type: ignore
import websockets.sync.client


class Article(pydantic.BaseModel):
    title: str
    path: str
    season: str | None = None

    def __hash__(self) -> int:
        return hash((type(self),) + tuple(self.__dict__.values()))


dotenv.load_dotenv()

DISCORD_API_TOKEN: str | None = os.getenv("DISCORD_API_TOKEN") or None
DISCORD_API_ENDPOINT: str = "https://discordapp.com/api"
DISCORD_POSTING_CHANNEL_NAME: str = "winds-watch"
DISCORD_HEADERS = {
    "Authorization": f"Bot {DISCORD_API_TOKEN}",
    "Content-Type": "application/json",
}
DISCORD_BANNED_GUILDS: list[str] = []  # ["1015714452445864048"]
DISCORD_DMS: list[str] = json.loads(os.environ["DISCORD_DMS"])
DISCORD_MAX_UPLOAD_SIZE: int = 8_388_608  # 8 MiB

WIKI_URL = "https://www.profounddecisions.co.uk"
TTS_URL = "https://www.pprofounddecisions.co.uk"

TAGS_IGNORE_LINE1 = ["winds", "recent history", "text to speech"]
TAGS_IGNORE_LINE2 = ["imperial:.*", "special:.*", "file:.*"]
TAGS_BAN_LIST = ["senate motion"]
TMP_FORCE_ADD: list[Article] = []
TMP_FORCE_ADD_SEASON = ""

# SEASONS = json.loads(os.environ["SEASONS"])
SEASON_RE = re.compile(r"Category:\d{3}YE_(Spring|Summer|Autumn|Winter)")
RECENT_HISTORY = "Category:Recent_History"
ARTICLES_JSON = pathlib.Path(os.environ["ARTICLES_JSON"])
USERS_JSON = pathlib.Path(os.environ["USERS_JSON"])
TTS_SEASON_JSON_DIR = pathlib.Path(os.environ["TTS_SEASON_JSON_DIR"])
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MAX_TOKENS = 10_000
OPENAI_API_MODEL = os.environ["OPENAI_API_MODEL"]
IO_SUMMARY_CHANNELS = os.getenv("IO_SUMMARY_CHANNELS")
HEADER_REGEX = re.compile(r"^h[0-9]{1}$")
NEWLINE = "\n"
MENTIONS = [
    {"search": "Archmage of Day", "mention": "<@279953529991593986>"},
    {"search": "Ashborn Trosk", "mention": "<@250734912599097344>"},
    {"search": "Bloodcrow Knott", "mention": "<@133962599015383041>"},
    {"search": "Bloodcrow Rasp", "mention": "<@289762927718957056>"},
    {"search": "Bloodcrow Udoo", "mention": "<@234781274546372610>"},
    {"search": "Gralka", "mention": "<@279953529991593986>"},
    {"search": "Protectorate Kardak", "mention": "<@111124276140072960>"},
    {"search": "Rykana", "mention": "<@279700839533379585>"},
    {"search": "Weigher of Worth", "mention": "<@423209028529815562>"},
    {"search": "Winds of War", "mention": "<@214070708676984833>"},
    {"search": "Winds of War", "mention": "<@279700839533379585>"},
    {"search": "Winds of War", "mention": "<@728023074527772733>"},
]
BANNERS = [
    "Ashborn",
    "Bleakshield",
    "Bloodcrow",
    "Frostbear",
    "Gaterender",
    "Irontide",
    "Oathborn",
    "Palerictus",
    "Protectorate",
    "Redhand",
    "Ringforge",
    "Sandviper",
    "Skywise",
    "Stormcrow",
    "Thundercall",
]
CHATGPT_INTRO = f"""First some background information to help you understand the Empire and specifically the Imperial Orcs:

Territories consists of regions. If you control a majority of the regions in a territory you control the territory.

The Empire is surrounded by barbarian nations nearly all being slaving nations and nearly all at war with the Empire.
 - The Jotun to the west are slavers currently at war with the Empire
 - The Grendel to the south are slavers currently at war with the Empire
 - The Druj to the east are slavers currently at war with the Empire
 - The Thule to the north have recently given up slavery and is in a tenious alliance with the Empire
 - There are several individual territories around the empire infected with the Vallorn which is a malignant hostile ecosystem made of spring magic infecting all living things within it. You will often hear the Vallorn spoken about like another hostile Barbarian nation but the Vallorn has no will or goals apart from spreading.

The Imperial Orcs are the previous slaves of the Empire. They freed themselves through a bloody slave-rebellion over 60 years ago.
The nations of the Empire have 1 senator per territory, all making up the senate; the most powerful institution in the Empire.
All nations have multiple territories and therefore multiple senators with the Imperial Orcs having only 2 territories called Skarsind and Mareave, its senators currently being Bloodcrow Rasp and Pathfinder Gaddak respectively.
The Imperial Orcs are one of the 10 nations that make up the Empire.
The Imperial Orcs have three armies the first legion "Winter Sun" and the second legion "Summer Storm" and the newest army the "Autumn Hammers".
The Imperial Orcs consists of the following banners: {", ".join(BANNERS)}.
The Imperial Orcs are made up of 6 septs: Ethengraw, Illarawm, Sannite, Sunstorm, Tamazi and Yerende. Septs are unique orc cultures and many septs exists outside the Imperial Orcs.

The Imperial Orcs have recently also claimed the name "Unshackled" as a nation as a few orcs have started joining other nations within the Empire making "Imperial Orcs" a bit confusing. These are groups like the League Orcs which now call themselves the Unbroken or the Marcher orcs, previously the Mourn orcs.

Now, for your task. The following is an article describing a recent development in the Empire. We are ONLY interested in information DIRECTLY relevant to the Imperial Orc nation (no indirect relevancy). Please write a summary in less than 150 words containing ONLY information relevant to the Imperial Orc nation. If there is no information directly relevant for the Imperial Orc nation please only respond with "There is no new information relevant to the Imperial Orc nation in this article." and do not write a summary nor explain why.

"""
RETRIES = 6


def sizeof_fmt(num: float, suffix: str = "B") -> str:
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def summarize(url: str) -> str | None:
    soup = bs4.BeautifulSoup(
        httpx.get(url).text,
        "html.parser",
    )
    content = soup.find(attrs={"id": "mw-content-text"})
    text = ""
    if not isinstance(content, bs4.Tag):
        return loguru.logger.error(
            f'Could not summarize article "{url}: no "mw-content-text" in soup'
        )

    while isinstance(content, bs4.Tag) and len(list(content.children)) == 1:
        content = list(content.children)[0]  # type: ignore

    for tag in content:
        if isinstance(tag, bs4.Tag) and tag.name:
            if tag.name == "p":
                text += tag.text + "\n"
            elif tag.name == "ul":
                text += " - " + "\n - ".join(tag.text.split("\n")) + "\n"
            elif HEADER_REGEX.match(tag.name):
                text += f"# {tag.text}\n"
            else:
                tag.decompose()

    text = "\n".join(t.strip() for t in text.split("\n"))

    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    while True:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_API_MODEL,
                # "temperature": 0.2,
                # "max_tokens": 8192,
                "messages": [{"role": "user", "content": CHATGPT_INTRO + text}],
            },
            timeout=60,
        )

        if not resp.is_error:
            resp_json = resp.json()
            result = None
            if "choices" not in resp_json:
                result = f'Could not summarize article "{url}": No key "choices" in response: "{resp_json}"'
            if not resp_json["choices"]:
                result = f'Could not summarize article "{url}": Zero "choices" in response: "{resp_json}"'
            if "message" not in resp_json["choices"][0]:
                result = f'Could not summarize article "{url}": Choice 1 in response has no "message": "{resp_json}"'
            if "content" not in resp_json["choices"][0]["message"]:
                result = f'Could not summarize article "{url}": "message" in choice 1 in response has no "content": "{resp_json}"'

            if result:
                loguru.logger.error(result)
                return result

            result = resp_json["choices"][0]["message"]["content"]
            if not isinstance(result, str):
                result = f'Could not summarize article "{url}": "content" in "message" in choice 1 in response is not string: "{resp_json}"'
                loguru.logger.error(result)
                return result
            loguru.logger.info(f'Successfully summarized article "{url}"')
            return result
        else:
            try:
                if resp.json()["error"]["code"] == "rate_limit_exceeded":
                    loguru.logger.warning(
                        "Hit OpenAI rate limiting, retrying in 20 seconds."
                    )
                    time.sleep(20)
                    continue
                elif resp.json()["error"]["code"] == "context_length_exceeded":
                    return "Article too long for AI to summarise... Sorry"
                elif resp.json()["error"]["code"] == "insufficient_quota":
                    loguru.logger.warning(
                        "Insufficient OpenAI Quota for summarization - retrying in 1 hour."
                    )
                    time.sleep(60 * 60)  # 1 hour
                    continue
                else:
                    loguru.logger.error(
                        f'Could not summarize article "{url}": {resp.text}'
                    )

            except json.decoder.JSONDecodeError:
                loguru.logger.error(f'Could not summarize article "{url}": {resp}')
            return None


def create_link_comp(label: str, url: str, emoji: str) -> dict:
    return {
        "style": 5,
        "label": label,
        "url": url,
        "disabled": False,
        "emoji": {"id": None, "name": emoji},
        "type": 2,
    }


def split_audio(audio_bytes: bytes) -> list[bytes]:
    bit_rate = mutagen.mp3.MP3(io.BytesIO(audio_bytes)).info.bitrate  # type: ignore

    audio = pydub.AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

    bytes_list = []
    for i, chunk in enumerate(
        make_chunks(audio, int((DISCORD_MAX_UPLOAD_SIZE / bit_rate * 8) * 1000))
    ):
        # Export chunk into the buffer
        buf = io.BytesIO()
        chunk.set_channels(1)
        chunk.export(buf, format="mp3", bitrate=str(bit_rate // 1000) + "k")

        # Fetch the byte data & add it to the list
        byte_data = buf.getvalue()
        bytes_list.append(byte_data)

    return bytes_list


def make_chunks(
    audio: pydub.AudioSegment, chunk_duration_ms: int
) -> list[pydub.AudioSegment]:
    chunks = []
    while len(audio) > chunk_duration_ms:
        chunks.append(audio[:chunk_duration_ms])
        audio = audio[chunk_duration_ms:]
    chunks.append(audio)
    return chunks


def upload_audios(title: str, channel_id: str, audio_datas: list[bytes]) -> list[dict]:
    attachments: list[dict] = []

    files = [
        {"file_size": len(audio_data), "filename": f"{title}_{i}.mp3"}
        for i, audio_data in enumerate(audio_datas)
    ]
    r = httpx.post(
        f"{DISCORD_API_ENDPOINT}/channels/{channel_id}/attachments",
        headers=DISCORD_HEADERS,
        json={"files": files},
    )
    if r.is_error:
        loguru.logger.error(f"{r}: {r.text}")
        r.raise_for_status()

    for i, (upload, audio_data) in enumerate(zip(r.json()["attachments"], audio_datas)):
        r = httpx.put(upload["upload_url"], content=audio_data)
        r.raise_for_status()

        attachments.append(
            {
                "id": i,
                "filename": f"{title}_{i}.mp3",
                "uploaded_filename": upload["upload_filename"],
            }
        )

    return attachments


def post_to_discord(
    title: str,
    description: str,
    url: str,
    tags_line_0: list[tuple[str, str, str]],
    tags_line_1: list[tuple[str, str, str]],
    img: str | None = None,
    content: str = "",
    relevancy: str = "",
    path: str = "",
) -> bool:
    if DISCORD_API_TOKEN:
        # Start generating TTS audio immidately
        article_id = path.split("/")[-1].split("&")[0]
        httpx.get(f"{TTS_URL}/api/manuscript/{article_id}")

        # Construct embed
        summary = summarize(url) if IO_SUMMARY_CHANNELS else None
        embed: dict = {
            "type": "rich",
            "title": title,
            "description": description,
            "url": url,
            "fields": [],
            "timestamp": str(datetime.datetime.now(datetime.UTC)),
        }
        if img:
            embed["image"] = {"url": img}
        if tags_line_0:
            embed["author"] = {
                "name": "・".join(
                    [t for t, _, _ in tags_line_0 if t.lower() not in TAGS_IGNORE_LINE1]
                )
            }

        # Construct body
        json_body: dict = {
            "content": content,
            "tts": False,
            "embeds": [embed],
            "components": [
                {
                    "type": 1,
                    "components": [
                        create_link_comp(tag, url, emoji)
                        for tag, url, emoji in tags_line_0[
                            :5
                        ]  # Max 5 components per line
                    ],
                },
                {
                    "type": 1,
                    "components": [
                        create_link_comp(tag, url, emoji)
                        for tag, url, emoji in tags_line_1[
                            :5
                        ]  # Max 5 components per line
                    ],
                },
            ],
        }

        r = httpx.get(
            DISCORD_API_ENDPOINT + "/users/@me/guilds", headers=DISCORD_HEADERS
        )
        if not r.is_success:
            loguru.logger.error(f"Could not access Discord: {r.text}")
            return False

        my_guilds = r.json()
        if not my_guilds:
            loguru.logger.warning("Bot not added to any guilds!")
            return False

        channels = []
        for guild in my_guilds:
            if "id" in guild and guild["id"] not in DISCORD_BANNED_GUILDS:
                while True:
                    r = httpx.get(
                        f"{DISCORD_API_ENDPOINT}/guilds/{guild['id']}/channels",
                        headers=DISCORD_HEADERS,
                    )
                    if r.is_success:
                        for channel in r.json():
                            if channel["name"] == DISCORD_POSTING_CHANNEL_NAME:
                                channel["guild_name"] = guild["name"]
                                channels.append(channel)
                                break
                        break
                    else:
                        timeout = 10
                        if r.status_code == 429:
                            timeout = r.json()["retry_after"] / 1000
                            loguru.logger.warning(
                                f"Too many requests to Discord, waiting {timeout} seconds: {r} | {r.text}"
                            )
                        else:
                            loguru.logger.warning(
                                f"Could not get discord guilds, waiting 10 seconds: {r} | {r.text}"
                            )
                        time.sleep(timeout)

        for u in json.load(open(USERS_JSON)).values():
            channels += [
                httpx.get(
                    DISCORD_API_ENDPOINT + f"/channels/{c}", headers=DISCORD_HEADERS
                ).json()
                for c in u["channels"]
            ]

        channels_posted_to: list = []
        for channel in channels:
            channel_identifier = channel["id"]
            if "guild_name" in channel and channel["guild_name"]:
                channel_identifier = f"{channel["guild_name"]} (Guild)"
            elif "recipients" in channel and channel["recipients"]:
                channel_identifier = (
                    ", ".join(r["username"] for r in channel["recipients"])
                    + " (Direct message)"
                )

            # Wait for TTS audio
            for i in range(RETRIES):
                while True:
                    r = httpx.get(f"{TTS_URL}/api/manuscript/{article_id}")
                    if r.is_success:
                        break
                    else:
                        loguru.logger.warning(
                            f"Error getting manuscript, retrying in 1 min: {r} | {r.text}"
                        )
                        time.sleep(60)
                if r.is_error:
                    loguru.logger.warning(
                        f'"{article_id}" API error "{r}", retrying in 10 min ({i+1}/{RETRIES} retries)'
                    )
                manuscript = r.json()
                if (
                    manuscript["state"] == "disallowed"
                    or manuscript["state"] == "error"
                ):
                    loguru.logger.warning(
                        f'TTS for "{article_id}" {manuscript["state"]} - skipping audio download'
                    )
                    break
                if manuscript["state"] != "done":
                    loguru.logger.info(
                        f'"{article_id}" not done ({manuscript["state"]}), retrying in 10 min ({i+1}/{RETRIES} retries)'
                    )
                elif "complete_audio_url" not in manuscript:
                    loguru.logger.info(
                        f'No complete audio URL in "{article_id}", retrying in 10 min ({i+1}/{RETRIES} retries)'
                    )
                else:
                    # Download audio
                    complete_audio_url = f'{TTS_URL}{r.json()["complete_audio_url"]}'

                    audio_data = httpx.get(complete_audio_url).content
                    audio_datas = split_audio(audio_data)

                    if len(audio_datas) > 1:
                        loguru.logger.warning(
                            f"Audio file too large ({sizeof_fmt(len(audio_data))}) splitting into {len(audio_datas)} files"
                        )

                    json_body["attachments"] = upload_audios(
                        title, channel["id"], audio_datas
                    )
                    break

                time.sleep(10 * 60)

            footer = {
                "text": (f"IO SUMMARY: {summary}\n\n" if summary else "")
                + (f"RELEVANT FOR: {relevancy}\n\n" if relevancy else "")
            }
            if "created" in manuscript:
                footer["text"] += "Created"
                json_body["embeds"][0]["timestamp"] = manuscript["created"]
            elif "lastmod" in manuscript:
                footer["text"] += "Last modified"
                json_body["embeds"][0]["timestamp"] = manuscript["lastmod"]
            else:
                footer["text"] += "Created (Estimated)"
                loguru.logger.warning(
                    f'No "created" or "lastmod" in manuscript - using detection time'
                )

            json_body["embeds"][0]["footer"] = footer

            # Send message
            while True:
                try:
                    r = httpx.post(
                        f"{DISCORD_API_ENDPOINT}/channels/{channel['id']}/messages",
                        headers=DISCORD_HEADERS,
                        json=json_body,
                    )
                    break
                except httpx.ReadTimeout as e:
                    loguru.logger.warning(
                        f"Error while posting to Discord, retrying in 10s: {e}"
                    )
                    time.sleep(10)

            if r.is_success:
                channels_posted_to.append(channel_identifier)
                message = r.json()
                if "guild_id" in channel:
                    while True:
                        try:
                            r = httpx.post(
                                f"{DISCORD_API_ENDPOINT}/channels/{channel['id']}/messages/{message['id']}/threads",
                                headers=DISCORD_HEADERS,
                                json={
                                    "name": title,
                                    "auto_archive_duration": 1440,
                                },
                            )
                            break
                        except httpx.ReadTimeout:
                            loguru.logger.warning(
                                f"Could not create message thread, retrying in 10s"
                            )
                            time.sleep(10)

                    if r.is_success:
                        thread = r.json()
                        r = httpx.post(
                            f"{DISCORD_API_ENDPOINT}/channels/{thread['id']}/messages",
                            headers=DISCORD_HEADERS,
                            json={
                                "content": f"**Please try to keep discussions in these threads.**"
                            },
                        )
                        if not r.is_success:
                            loguru.logger.error(
                                f'"{title}" in guild "{channel_identifier}" - Could not post message to thread: {r} | {r.text}'
                            )
                    else:
                        loguru.logger.error(
                            f'"{title}" in guild "{channel_identifier}" - Could not create thread: {r} | {r.text}'
                        )
            else:
                loguru.logger.error(
                    f'"{title}" in guild "{channel_identifier}" - Could not post main message: {r} | {r.text}'
                )
                loguru.logger.error(json_body)

        if channels_posted_to:
            loguru.logger.info(
                f'Successfully posted "{title}" to {len(channels_posted_to)} channels ({", ".join(f'"{c}"' for c in channels_posted_to)})'
            )
        return not not channels_posted_to
    return False


def get_page_title(url: str) -> str:
    soup = bs4.BeautifulSoup(
        httpx.get(url).text,
        "html.parser",
    )
    title = ""
    if isinstance((t := soup.find(attrs={"id": "page-title"})), bs4.Tag):
        if isinstance((h1 := t.find("h1")), bs4.Tag):
            title += h1.text.strip()
        else:
            loguru.logger.error(
                f'Could not "h1" tag in id "page-title" on page "{url}"'
            )
            return ""
    else:
        loguru.logger.error(f'Could not find id "page-title" on page "{url}"')
        return ""
    if "#" in url:
        sub_id = url.rsplit("#", 1)[-1]
        if isinstance((t := soup.find(attrs={"id": sub_id})), bs4.Tag):
            title += f" - {t.text}"
        else:
            loguru.logger.error(f'Could not find id "{sub_id}" on page "{url}"')
            return ""
    return title


def extract_season(path: str) -> str | None:
    url = f"{WIKI_URL}{path}"
    try:
        r = httpx.get(url, timeout=60)
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        loguru.logger.error(f'Could not get article from "{url}": {e}')
        return None

    soup = bs4.BeautifulSoup(r.text, "html.parser")
    categories = soup.find("div", {"id": "pageCategories"})
    if isinstance(categories, bs4.Tag):
        for a in categories.find_all("a", href=True):
            if match := SEASON_RE.search(a["href"]):
                return match.group(0).replace("Category:", "")

    return None


def get_recent_history(skip_articles: list[str]) -> list[Article]:
    url: str | None = f"{WIKI_URL}/empire-wiki/{RECENT_HISTORY}"
    pages: list[tuple[str, str]] = []
    i = 0
    while url:
        try:
            r = httpx.get(url, timeout=60)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            loguru.logger.error(f'Could not get articles from "{url}": {e}')
            return []

        soup = bs4.BeautifulSoup(r.text, "html.parser")
        categories = soup.find("div", {"class": "mw-category"})
        if isinstance(categories, bs4.Tag):
            pages += [
                (a["title"], a["href"].rsplit("?")[0])
                for a in categories.find_all("a", href=True, title=True)
                if ":" not in a["href"] and a["href"] not in skip_articles
            ]
        else:
            loguru.logger.error(f'No "mw-category" in "{url}", adding no pages')

        next_page = soup.find("a", string="next page", href=True)
        if isinstance(next_page, bs4.Tag):
            url = f"{WIKI_URL}{next_page["href"]}"
        else:
            url = None

    return [Article(title=t, path=p, season=extract_season(p)) for t, p in pages]


def extract_wof_relevancy(season: str) -> dict[str, dict]:
    wof_article = f'{WIKI_URL}/empire-wiki/{season}_{"Solstice" if "Summer" in season or "Winter" in season else "Equinox"}_winds_of_fortune'
    while True:
        try:
            response = httpx.get(wof_article)
            break
        except httpx.ConnectError as e:
            loguru.logger.error(f'Could not get WoF relevancy for "{season}": {e}')
            return {}
        except httpx.ConnectTimeout as e:
            loguru.logger.warning(
                f'Getting WoF relevancy for "{season}" timed out, waiting 10s: {e}'
            )
            time.sleep(10)

    if not response.is_success:
        loguru.logger.warning(
            f'"{wof_article}" not valid page - articles will not have relevancy'
        )
        return {}

    page_soup = bs4.BeautifulSoup(response.text, "html.parser")
    _page_content = page_soup.find("div", {"id": "mw-content-text"})
    if isinstance(_page_content, bs4.Tag):
        page_content = _page_content.find(
            "div", {"class": "mw-parser-output"}
        ) or _page_content.find("div", {"id": "mw-content-text"})
        wof_relevancy: dict[str, dict] = {}
        current_wof = ""
        latest_link = ""
        if isinstance(page_content, bs4.Tag):
            content_children = page_content.find_all(recursive=False)
            for child in content_children:
                if child.name.startswith("h"):
                    if current_wof:
                        if (
                            "href" in wof_relevancy[current_wof]
                            and "relevancy" in wof_relevancy[current_wof]
                        ):
                            wof_relevancy[wof_relevancy[current_wof]["href"]] = (
                                wof_relevancy[current_wof]
                            )

                        del wof_relevancy[current_wof]

                    current_wof = re.sub(r"\(.*\)", "", child.text).strip()
                    wof_relevancy[current_wof] = {"name": current_wof}
                elif current_wof and child.name == "ul":
                    wof_relevancy[current_wof]["relevancy"] = child.text.strip()
                elif current_wof and child.name == "p":
                    if links := child.find_all("a", recursive=True):
                        wof_relevancy[current_wof]["href"] = links[-1]["href"]

    return wof_relevancy


def update_articles(
    new_articles: list[Article], wof_relevancy: dict[str, dict]
) -> list[Article]:
    loguru.logger.info(f"Updating with {len(new_articles)} new articles")
    for page in new_articles:
        url = f"{WIKI_URL}{page.path}"
        page_soup = bs4.BeautifulSoup(httpx.get(url).text, "html.parser")
        page_content = page_soup.find("div", {"id": "mw-content-text"})
        assert isinstance(page_content, bs4.Tag)
        ic = page_content.find("div", {"class": "ic"})
        if isinstance(ic, bs4.Tag):
            ic.decompose()

        # Get tag line 0 - TTS and page tags
        tags0 = [(page.title, url, "#️⃣")]
        page_categories = page_soup.find("div", {"id": "pageCategories"})

        if isinstance(page_categories, bs4.Tag):
            page_tags = [
                a
                for a in page_categories.find_all("a")
                if a.text.lower() not in TAGS_IGNORE_LINE1
            ]

            if banned_tags := [
                tag for tag in page_tags if tag.text.lower() in TAGS_BAN_LIST
            ]:
                loguru.logger.warning(
                    f'"{page.title}" ({page.path}) contains banned tags, removing ({banned_tags})'
                )
                articles.append(page)
                with open(ARTICLES_JSON, "w") as f:
                    json.dump([a.model_dump() for a in articles], f, indent=2)
                continue

            tags0 = [(a.text, f"{WIKI_URL}{a['href']}", "#️⃣") for a in page_tags]

        tags0.insert(0, ("Text to speech", f"{TTS_URL}{page.path}", "🎙️"))

        # Get tag line 1 - most linked pages
        paths: list[str] = [
            f'{WIKI_URL}{a.get("href").rsplit("#", 1)[0]}'
            for a in page_soup.find_all("a", recursive=True)
            if a.get("href") and a.get("href").startswith("/")
        ]
        tags1 = [
            (get_page_title(a), a, "📙")
            for _, a in sorted([(paths.count(p), p) for p in set(paths)], reverse=True)[
                :3
            ]
            if not any(
                re.match(t, get_page_title(a).lower()) for t in TAGS_IGNORE_LINE2
            )
        ]

        # Get description
        description = next(
            (
                p.text.strip()
                for p in page_content.find_all("p")
                if isinstance(p, bs4.Tag) and p.text.strip()
            ),
            None,
        )
        if not description:
            loguru.logger.warning(
                f'"{page.title}" ({page.path}) does not have a description yet, waiting'
            )
            continue

        # Lookup relevancy
        relevancy = ""
        if page.path in wof_relevancy:
            relevancy = wof_relevancy[page.path]["relevancy"]
        else:
            loguru.logger.warning(
                f'"{page.path}" not in WoF relevancy, assuming it is not a WoF'
            )

        # Get image
        img = page_content.find("img")
        img_url = f"{WIKI_URL}{img['src']}" if isinstance(img, bs4.Tag) else None

        # Create mesasge with mentions
        page_content_text = page_content.text.lower()
        mentions: list[dict] = []
        for l in MENTIONS:
            if l["search"].lower() in page_content_text and l["mention"] not in [
                m["mention"] for m in mentions
            ]:
                l["pretty"] = f'- {l["mention"]} ({l["search"]})'
                mentions.append(l)
        content = (
            f"Article mentions:\n{NEWLINE.join(m['pretty'] for m in mentions)}\n(contact <@133962599015383041> to be added to/removed from mentions search)"
            if mentions
            else ""
        )
        # Discord
        if post_to_discord(
            page.title,
            description,
            url,
            tags0,
            tags1,
            img_url,
            content,
            relevancy,
            page.path,
        ):
            articles.append(page)
            with open(ARTICLES_JSON, "w") as f:
                json.dump([a.model_dump() for a in articles], f, indent=2)

    return articles


def update_users(terminate: multiprocessing.sharedctypes.Synchronized) -> None:
    loguru.logger.info("User update process started")
    users = json.load(open(USERS_JSON))
    uri = f"wss://gateway.discord.gg/?v=8&encoding=json"
    while not terminate.value:
        try:
            with websockets.sync.client.connect(uri) as websocket:
                identify_event = {
                    "op": 2,
                    "d": {
                        "token": DISCORD_API_TOKEN,
                        "intents": 4096,  # Direct Messages
                        "properties": {
                            "$os": "linux",
                            "$browser": "my_bot",
                            "$device": "my_bot",
                        },
                        "presence": {
                            "status": "online",
                            "afk": False,
                            "since": 91879201,
                            "activities": [],
                        },
                    },
                }

                websocket.send(json.dumps(identify_event))

                for message in websocket:
                    if terminate.value:
                        break
                    message = json.loads(message)

                    # Check if it's MESSAGE_CREATE event and that it's a DM
                    if (
                        "t" in message
                        and message["t"] == "MESSAGE_CREATE"
                        and (
                            "guild_id" not in message["d"]
                            or not message["d"]["guild_id"]
                        )
                    ):
                        author = message["d"]["author"]
                        if author["id"] not in users:
                            loguru.logger.info(
                                f'Found new user "{author["username"]} ({author["id"]})"'
                            )
                            users[author["id"]] = author

                        if "channels" not in users[author["id"]]:
                            users[author["id"]]["channels"] = []

                        if (
                            message["d"]["channel_id"]
                            not in users[author["id"]]["channels"]
                        ):
                            users[author["id"]]["channels"].append(
                                message["d"]["channel_id"]
                            )
                            loguru.logger.info(
                                f'Found new channel "{message["d"]["channel_id"]} (DM: {author["username"]})"'
                            )

                    json.dump(users, open(USERS_JSON, "w"), indent=4)
        except Exception as e:
            loguru.logger.warning(
                f"User update process encountered an error, restarting in 10s: {e}"
            )
            time.sleep(10)

    loguru.logger.info("User update process ended")


terminate_user_update = multiprocessing.Value("i", 0)
user_update_process = multiprocessing.Process(
    target=update_users, args=(terminate_user_update,)
)
user_update_process.start()

loguru.logger.info("Empire Wiki Winds update process started")
articles = [Article(**a) for a in json.load(open(ARTICLES_JSON))]
while True:
    # if TMP_FORCE_ADD:
    #     if TMP_FORCE_ADD:
    #         articles = update_articles(TMP_FORCE_ADD, articles)

    new_articles = list(
        sorted(
            get_recent_history([a.path for a in articles]),
            key=lambda a: str(a.title),
        )
    )
    if new_articles:
        wof_relevancy: dict[str, dict] = {}
        for s in set([a.season for a in new_articles if a.season]):
            wof_relevancy |= extract_wof_relevancy(s)

        articles = update_articles(new_articles, wof_relevancy)

    time.sleep(10 * 60)  # Check every 10 minutes


def exit_handler():
    global user_update_process, terminate_user_update
    if user_update_process:
        loguru.logger.info("Terminating user update process...")
        terminate_user_update.value = 1
        user_update_process.join()
        loguru.logger.info("User update process stopped, exiting")


atexit.register(exit_handler)
