import datetime
import io
import json
import multiprocessing
import os
import re
import time
import typing
import urllib
from pathlib import Path

import dotenv
import httpx
import mutagen.mp3
import pydub
from bs4 import BeautifulSoup, Tag
from loguru import logger

dotenv.load_dotenv()

DISCORD_API_TOKEN: str | None = os.getenv("DISCORD_API_TOKEN") or None
DISCORD_API_ENDPOINT: str = "https://discordapp.com/api"
DISCORD_POSTING_CHANNEL_NAME: str = "winds-watch"
DISCORD_HEADERS = {
    "Authorization": f"Bot {DISCORD_API_TOKEN}",
    "Content-Type": "application/json",
}
DISCORD_BANNED_GUILDS: list[str] = []
DISCORD_MAX_UPLOAD_SIZE: int = 8_388_608  # 8 MiB

WIKI_URL = "https://www.profounddecisions.co.uk"
TTS_URL = "https://www.pprofounddecisions.co.uk"

TAGS_IGNORE_LINE1 = ["winds", "recent history", "text to speech"]
TAGS_IGNORE_LINE2 = ["imperial:.*", "special:.*", "file:.*"]
TAGS_BAN_LIST = ["senate motion"]
TMP_FORCE_ADD: list[tuple[str, str]] = []
TMP_FORCE_ADD_SEASON = ""

SEASONS = json.loads(os.environ["SEASONS"])
ARTICLES_JSON = Path(os.environ["ARTICLES_JSON"])
TTS_SEASON_JSON_DIR = Path(os.environ["TTS_SEASON_JSON_DIR"])
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MAX_TOKENS = 10_000
HEADER_REGEX = re.compile(r"^h[0-9]{1}$")
NEWLINE = "\n"
MENTIONS = [
    {"search": "Archmage of Day", "mention": "<@279953529991593986>"},
    {"search": "Ashborn Trosk", "mention": "<@250734912599097344>"},
    {"search": "Bloodcrow Knott", "mention": "<@133962599015383041>"},
    {"search": "Bloodcrow Rasp", "mention": "<@289762927718957056>"},
    {"search": "Bloodcrow Udoo", "mention": "<@234781274546372610>"},
    {"search": "Gralka", "mention": "<@279953529991593986>"},
    {"search": "Rykana", "mention": "<@279700839533379585>"},
    {"search": "Weigher of Worth", "mention": "<@423209028529815562>"},
    {"search": "Winds of War", "mention": "<@728023074527772733>"},
    {"search": "Winds of War", "mention": "<@279700839533379585>"},
    {"search": "Winds of War", "mention": "<@214070708676984833>"},
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

The Imperial Orcs are the previous slaves of the Empire. They freed themselves through a bloody slave-rebellion over 60 years ago.
The nations of the Empire have 1 senator per territory, all making up the senate; the most powerful institution in the Empire.
All nations have multiple territories and therefore multiple senators with the Imperial Orcs having only 2 territories called Skarsind and Mareave, its senators currently being Bloodcrow Rasp and Pathfinder Gaddak respectively.
The Imperial Orcs are one of the 10 nations that make up the Empire.
The Imperial Orcs have three armies the first legion "Winter Sun" and the second legion "Summer Storm" and the newest army the "Autumn Hammers".
The Imperial Orcs consists of the following banners: {", ".join(BANNERS)}.

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
    soup = BeautifulSoup(
        httpx.get(url).text,
        "html.parser",
    )
    content = soup.find(attrs={"id": "mw-content-text"})
    text = ""
    if not isinstance(content, Tag):
        return logger.error(
            f'Could not summarize article "{url}: no "mw-content-text" in soup'
        )

    while isinstance(content, Tag) and len(list(content.children)) == 1:
        content = list(content.children)[0]  # type: ignore

    for tag in content:
        if isinstance(tag, Tag) and tag.name:
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
                "model": "gpt-4.5-preview",
                "temperature": 0.2,
                # "max_tokens": 8192,
                "messages": [{"role": "user", "content": CHATGPT_INTRO + text}],
            },
            timeout=60,
        )

        if not resp.is_error:
            resp_json = resp.json()
            if "choices" not in resp_json:
                return logger.error(
                    f'Could not summarize article "{url}": No key "choices" in response: "{resp_json}"'
                )
            if not resp_json["choices"]:
                return logger.error(
                    f'Could not summarize article "{url}": Zero "choices" in response: "{resp_json}"'
                )
            if "message" not in resp_json["choices"][0]:
                return logger.error(
                    f'Could not summarize article "{url}": Choice 1 in response has no "message": "{resp_json}"'
                )
            if "content" not in resp_json["choices"][0]["message"]:
                return logger.error(
                    f'Could not summarize article "{url}": "message" in choice 1 in response has no "content": "{resp_json}"'
                )

            content = resp_json["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                return logger.error(
                    f'Could not summarize article "{url}": "content" in "message" in choice 1 in response is not string: "{resp_json}"'
                )

            return content
        else:
            try:
                logger.error(f'Could not summarize article "{url}": {resp.text}')
                if resp.json()["error"]["code"] == "rate_limit_exceeded":
                    logger.info("Hit OpenAI rate limiting, retrying in 20 seconds.")
                    time.sleep(10)
                    continue
                if resp.json()["error"]["code"] == "context_length_exceeded":
                    return "Article too long for AI to summarise... Sorry"
            except json.decoder.JSONDecodeError:
                logger.error(f'Could not summarize article "{url}": {resp}')
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
        summary = ""  # summarize(url)
        embed = {
            "type": "rich",
            "title": title,
            "description": description,
            "url": url,
            "fields": [],
            "timestamp": str(datetime.datetime.now(datetime.UTC)),
            "footer": {
                "text": (f"IO SUMMARY: {summary}\n\n" if summary else "")
                + (f"RELEVANT FOR: {relevancy}\n\n" if relevancy else "")
                + "Uploaded",
            },
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

        guilds_posted_to: list = []
        for guild in httpx.get(
            DISCORD_API_ENDPOINT + "/users/@me/guilds", headers=DISCORD_HEADERS
        ).json():
            if "id" in guild and guild["id"] not in DISCORD_BANNED_GUILDS:
                for channel in httpx.get(
                    f"{DISCORD_API_ENDPOINT}/guilds/{guild['id']}/channels",
                    headers=DISCORD_HEADERS,
                ).json():
                    if channel["name"] == DISCORD_POSTING_CHANNEL_NAME:
                        # Wait for TTS audio
                        for i in range(RETRIES):
                            r = httpx.get(f"{TTS_URL}/api/manuscript/{article_id}")
                            if r.is_error:
                                logger.warning(
                                    f'"{article_id}" API error "{r}", retrying in 10 min ({i+1}/{RETRIES} retries)'
                                )
                            if r.json()["state"] != "done":
                                logger.info(
                                    f'"{article_id}" not done, retrying in 10 min ({i+1}/{RETRIES} retries)'
                                )
                            elif "complete_audio_url" not in r.json():
                                logger.info(
                                    f'No complete audio URL in "{article_id}", retrying in 10 min ({i+1}/{RETRIES} retries)'
                                )
                            else:
                                # Download audio
                                complete_audio_url = (
                                    f'{TTS_URL}{r.json()["complete_audio_url"]}'
                                )

                                audio_data = httpx.get(complete_audio_url).content
                                audio_datas = split_audio(audio_data)

                                if len(audio_datas) > 1:
                                    logger.warning(
                                        f"Audio file too large ({sizeof_fmt(len(audio_data))}) splitting into {len(audio_datas)} files"
                                    )

                                json_body["attachments"] = upload_audios(
                                    title, channel["id"], audio_datas
                                )
                                break

                            time.sleep(10 * 60)

                        # Send message
                        r = httpx.post(
                            f"{DISCORD_API_ENDPOINT}/channels/{channel['id']}/messages",
                            headers=DISCORD_HEADERS,
                            json=json_body,
                        )
                        if r.is_success:
                            message = r.json()
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
                                    logger.warning(
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
                                if r.is_success:
                                    guilds_posted_to.append(f'"{guild["name"]}"')
                                else:
                                    logger.error(
                                        f'"{title}" in guild "{guild["name"]}" - Could not post message to thread: {r} | {r.text}'
                                    )
                            else:
                                logger.error(
                                    f'"{title}" in guild "{guild["name"]}" - Could not create thread: {r} | {r.text}'
                                )
                        else:
                            logger.error(
                                f'"{title}" in guild "{guild["name"]}" - Could not post main message: {r} | {r.text}'
                            )
                            logger.error(json_body)
        if guilds_posted_to:
            logger.info(
                f'Successfully posted "{title}" to {len(guilds_posted_to)} guilds ({", ".join( guilds_posted_to)})'
            )
        return not not guilds_posted_to
    return False


def get_page_title(url: str) -> str:
    soup = BeautifulSoup(
        httpx.get(url).text,
        "html.parser",
    )
    title = ""
    if isinstance((t := soup.find(attrs={"id": "page-title"})), Tag):
        if isinstance((h1 := t.find("h1")), Tag):
            title += h1.text.strip()
        else:
            logger.error(f'Could not "h1" tag in id "page-title" on page "{url}"')
            return ""
    else:
        logger.error(f'Could not find id "page-title" on page "{url}"')
        return ""
    if "#" in url:
        sub_id = url.rsplit("#", 1)[-1]
        if isinstance((t := soup.find(attrs={"id": sub_id})), Tag):
            title += f" - {t.text}"
        else:
            logger.error(f'Could not find id "{sub_id}" on page "{url}"')
            return ""
    return title


def extract_links(season: str) -> list[tuple[str, str]]:
    url = f"{WIKI_URL}/empire-wiki/Category:{season}"
    try:
        r = httpx.get(url, timeout=60)
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        logger.error(f'Could not get articles from "{season}": {e}')
        return []

    soup = BeautifulSoup(
        r.text,
        "html.parser",
    )

    pages = soup.find("div", {"id": "mw-pages"})
    if isinstance(pages, Tag):
        return [(a["title"], a["href"].rsplit("&", 1)[0]) for a in pages.find_all("a")]
    else:
        logger.error(f'No "mw-pages" in "{url}", returning no pages')
        return []


def extract_wof_relevancy(season: str) -> dict[str, dict]:
    wof_article = f'{WIKI_URL}/empire-wiki/{season}_{"Solstice" if "Summer" in season or "Winter" in season else "Equinox"}_winds_of_fortune'
    try:
        response = httpx.get(wof_article)
    except httpx.ConnectError as e:
        logger.error(f'Could not get WoF relevancy for "{season}": {e}')
        return {}

    if not response.is_success:
        logger.warning(
            f'"{wof_article}" not valid page - articles will not have relevancy'
        )
        return {}

    page_soup = BeautifulSoup(response.text, "html.parser")
    page_content = page_soup.find(
        "div", {"class": "mw-parser-output"}
    ) or page_soup.find("div", {"id": "mw-content-text"})

    wof_relevancy: dict[str, dict] = {}
    current_wof = ""
    latest_link = ""
    if isinstance(page_content, Tag):
        content_children = page_content.findChildren(recursive=False)
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
                if links := child.findChildren("a", recursive=True):
                    wof_relevancy[current_wof]["href"] = links[-1]["href"]

    return wof_relevancy


def update_articles(
    a_tags: list[tuple[str, str]],
    articles: list[tuple[str, str]],
    season: str,
) -> list[tuple[str, str]]:
    wof_relevancy = extract_wof_relevancy(season)

    for title, path in sorted(set(a_tags) - set(articles), key=lambda a: str(a[0])):
        url = f"{WIKI_URL}{path}"
        page_soup = BeautifulSoup(httpx.get(url).text, "html.parser")
        page_content = page_soup.find("div", {"id": "mw-content-text"})
        assert isinstance(page_content, Tag)
        ic = page_content.find("div", {"class": "ic"})
        if isinstance(ic, Tag):
            ic.decompose()

        # Get tag line 0 - TTS and page tags
        tags0 = [(title, url, "#️⃣")]
        page_categories = page_soup.find("div", {"id": "pageCategories"})

        if isinstance(page_categories, Tag):
            page_tags = [
                a
                for a in page_categories.find_all("a")
                if a.text.lower() not in TAGS_IGNORE_LINE1
            ]

            if banned_tags := [
                tag for tag in page_tags if tag.text.lower() in TAGS_BAN_LIST
            ]:
                logger.warning(
                    f'"{title}" ({path}) contains banned tags, removing ({banned_tags})'
                )
                articles.append((title, path))
                with open(ARTICLES_JSON, "w") as f:
                    json.dump(articles, f, indent=2)
                continue

            tags0 = [(a.text, f"{WIKI_URL}{a['href']}", "#️⃣") for a in page_tags]

        tags0.insert(0, ("Text to speech", f"{TTS_URL}{path}", "🎙️"))

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
                if isinstance(p, Tag) and p.text.strip()
            ),
            None,
        )
        if not description:
            logger.warning(
                f'"{title}" ({path}) does not have a description yet, waiting'
            )
            continue

        # Lookup relevancy
        relevancy = ""
        if path in wof_relevancy:
            relevancy = wof_relevancy[path]["relevancy"]
        else:
            logger.warning(f'"{path}" not in WoF relevancy, assuming it is not a WoF')

        # Get image
        img = page_content.find("img")
        img_url = f"{WIKI_URL}{img['src']}" if isinstance(img, Tag) else None

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
            title, description, url, tags0, tags1, img_url, content, relevancy, path
        ):
            articles.append((title, path))
            with open(ARTICLES_JSON, "w") as f:
                json.dump(articles, f, indent=2)

    return articles


logger.info("Empire Wiki Winds update process started")
with open(ARTICLES_JSON) as f:
    articles = json.load(f)
    articles = [tuple(a) for a in articles]

while True:
    if TMP_FORCE_ADD:
        if TMP_FORCE_ADD:
            articles = update_articles(TMP_FORCE_ADD, articles, TMP_FORCE_ADD_SEASON)

    for season in SEASONS:
        links = extract_links(season)
        if links:
            articles = update_articles(links, articles, season)

    time.sleep(10 * 60)  # Check every 10 minutes
