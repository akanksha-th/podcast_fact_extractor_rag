from aiogram import Bot, Dispatcher
from aiogram.filters import Command, CommandObject
from aiogram.types import Message
from bot.core.config import bot_settings
import asyncio
import httpx


settings = bot_settings()
bot = Bot(token=settings.bot_token)
dp = Dispatcher()

# run this function when /start is received
@dp.message(Command("start"))
async def handle_start(message: Message):
    await message.answer("Heyyyy\nEnter '/help' to know about the valid commands")

@dp.message(Command("help"))
async def get_help(message: Message):
    await message.answer("""
    /help: This commands shows the list of all the valid commands.

    All the other valid commands with description are mentioned below:
    1. /enter_url<youtub-video-url> : Enter youtube video url to begin a session.
    2. /ask<question-from-the-video-podcast-content>: You can ask any questions regarding the video content cntext.
    3. /get_notes: You get notes from the video transcript.
    4. /history: You can see their QnA history for the current session.
    5. /clear: End the current session and clear all of it's memory - You need to enter a new URL using the correct command to enter another session.
    """)

@dp.message(Command("enter_url"))
async def handle_enter_url(message: Message, command: CommandObject, http_client: httpx.AsyncClient):
    url = command.args      # Contains everything after "/enter_url"
    if not url:
        return await message.answer("Kindly enter the URL.\nUsage: /enter_url <url>")
    
    telegram_user_id = str(message.from_user.id)

    try:
        response = await http_client.post(
            settings.ingest_endpoint,
            json={
                "video_url": url,
                "user_id": telegram_user_id
                }
        )
        if response.status_code == 200:
            await message.answer("Processing...")
            for _ in range(30):
                await asyncio.sleep(2)
                status_resp = await http_client.get(
                    settings.status_endpoint,
                    params={"user_id": telegram_user_id}
                )
                data = status_resp.json()
                if data["status"] == "ready":
                    await message.answer("Podcast ready! Ask questions with /ask")
                    return
            await message.answer("⏳ This podcast is being transcribed. It may take 30-60 minutes for long videos. We'll process it in the background — try /ask in about an hour.")
        elif response.status_code == 400:
            await message.answer("Invalid YouTube URL")
        elif response.status_code == 429:
            await message.answer("You have reached your daily limit of 10 videos.")
        else:
            await message.answer(f"Server Error: {response.status_code}")
    except httpx.RequestError as e:
        await message.answer("Connection error: Could not reach the API.")

@dp.message(Command("ask"))
async def handle_queries(message: Message, command: CommandObject, http_client: httpx.AsyncClient):
    query = command.args
    if not query:
        return await message.answer("Please enter the question.")
    
    telegram_user_id = str(message.from_user.id)

    try:
        response = await http_client.post(
            settings.query_endpoint,
            json={
                "user_id": telegram_user_id,
                "question": query
            }
        )
        if response.status_code == 200:
            data = response.json()
            await message.answer(data["answer"])
        elif response.status_code == 400:
            await message.answer("Please enter a podcast URL first using /enter_url")
        elif response.status_code == 429:
            await message.answer("Rate limit exceeded.")
        else:
            await message.answer(f"Server Error: {response.status_code}")
    except httpx.RequestError as e:
        await message.answer("Connection error: Could not reach the API.")

@dp.message(Command("history"))
async def history(message: Message, http_client: httpx.AsyncClient):
    telegram_user_id = str(message.from_user.id)
    try:
        response = await http_client.get(
            "/api/v1/history",
            params={"user_id": telegram_user_id}
        )
        if response.status_code != 200:
            await message.answer(f"Server Error: {response.status_code}")
            return
        history_data = response.json()

        if not history_data["history"]:
            await message.answer("No history yet. Ask a question first!")
        else:
            formatted = "\n\n".join([
                f"Q: {item['question']}\nA: {item['answer']}" 
                for item in history_data["history"]
            ])
            await message.answer(formatted)
    except httpx.RequestError as e:
        await message.answer("Connection error: Could not reach the API ")

@dp.message(Command("clear"))
async def clear_session(message: Message, http_client: httpx.AsyncClient):
    telegram_user_id = str(message.from_user.id)
    try:
        response = await http_client.delete(
            "/api/v1/history",
            params={"user_id": telegram_user_id}
        )
        if response.status_code == 200:
            await message.answer("Successfully cleared the session history.")
        else:
            await message.answer("Failed to clear session history.")
    except httpx.RequestError as e:
        await message.answer("Connection error: Could not connect to the API.")

@dp.message(Command("get_notes"))
async def fetch_notes(message: Message, http_client: httpx.AsyncClient):
    telegram_user_id = str(message.from_user.id)
    await message.answer("⏳ Generating notes, this may take up to a minute...")
    
    try:
        response = await http_client.post(
            settings.notes_endpoint,
            params={"user_id": telegram_user_id},
            timeout=300.0
        )
        if response.status_code == 200:
            data = response.json()
            notes = data["notes"]
            # split into 4000 char chunks (leave buffer)
            chunk_size = 4000
            for i in range(0, len(notes), chunk_size):
                await message.answer(notes[i:i+chunk_size])
        elif response.status_code == 400:
            await message.answer("Please enter a podcast URL first using /enter_url")
        elif response.status_code == 429:
            await message.answer("Rate limit exceeded.")
        else:
            await message.answer(f"Server Error: {response.status_code}")
    except httpx.RequestError as e:
        await message.answer("Connection error: Could not connect to the API.")


async def main():
    async with httpx.AsyncClient(base_url=settings.api_base_url) as client:
        await dp.start_polling(bot, http_client=client)

if __name__ == "__main__":
    asyncio.run(main())