import os
import logging
import base64
import httpx
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN environment variable is not set!")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set!")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
)

# ── Socratic System Prompt ───────────────
SYSTEM_PROMPT = """You are Thinkr, a Socratic AI tutor for South African high school students Grades 8 to 12.

Your core philosophy: You NEVER just hand students the answer. You guide them to discover it themselves.

How you respond:
1. Acknowledge what the student is asking warmly and briefly
2. Ask ONE powerful guiding question that nudges them toward the answer
3. If they are stuck after 2 attempts give a small hint but not the full answer
4. Only reveal the full solution after the student has genuinely tried
5. End with a short note connecting the topic to real life

Subjects you cover: Mathematics, Physical Science, Life Sciences, Geography, History, English, Afrikaans, Economics, Accounting, Business Studies, Computer Applications Technology

Your tone: Warm and encouraging like a brilliant older sibling. Never condescending. Use simple South African English. Short sentences easy to read on a phone screen.

For photos: Read the image carefully and treat it like a typed question. Guide them Socratically.
For voice notes: Respond naturally as if they spoke to you.

Important rules:
- Never do a students homework outright
- Keep all responses under 200 words
- Use plain text only, no markdown, no bullet symbols
- Use numbered steps when showing working"""

# ── Conversation Memory ──────────────────
conversation_history: dict = {}

def get_history(user_id: int) -> list:
    return conversation_history.get(user_id, [])

def save_history(user_id: int, history: list):
    conversation_history[user_id] = history[-20:]

# ── Gemini API Call ──────────────────────
async def call_gemini(history: list, new_parts: list) -> str:
    contents = list(history)
    contents.append({"role": "user", "parts": new_parts})

    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 400,
            "temperature": 0.7,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(GEMINI_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

# ── Command Handlers ─────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversation_history[user_id] = []
    await update.message.reply_text(
        "Hey! I am Thinkr.\n\n"
        "I am your personal study buddy for Grades 8 to 12.\n\n"
        "I will not just give you answers. I will help you actually understand.\n\n"
        "Send me a typed question, a photo of your textbook, or a voice note.\n\n"
        "What subject are you working on today?"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "I can help with:\n\n"
        "Maths, Science, Biology, History, Geography, English, "
        "Afrikaans, Economics, Accounting, Business Studies, CAT\n\n"
        "Just send your question as text, photo, or voice note.\n\n"
        "/start - Start fresh\n"
        "/reset - Clear chat history\n"
        "/help - Show this message"
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conversation_history[update.effective_user.id] = []
    await update.message.reply_text("Fresh start! What would you like to study?")

# ── Message Handlers ─────────────────────
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    history = get_history(user_id)
    try:
        reply = await call_gemini(history, [{"text": user_text}])
        history.append({"role": "user", "parts": [{"text": user_text}]})
        history.append({"role": "model", "parts": [{"text": reply}]})
        save_history(user_id, history)
        await update.message.reply_text(reply)
    except Exception as e:
        logger.error(f"Text error: {e}")
        await update.message.reply_text("Something went wrong. Please try again in a moment.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    photo = update.message.photo[-1]
    caption = update.message.caption or "Please help me with this question in the image."
    photo_file = await context.bot.get_file(photo.file_id)

    async with httpx.AsyncClient(timeout=30.0) as client:
        img_response = await client.get(photo_file.file_path)
        img_base64 = base64.b64encode(img_response.content).decode("utf-8")

    history = get_history(user_id)
    new_parts = [
        {"text": caption},
        {"inline_data": {"mime_type": "image/jpeg", "data": img_base64}},
    ]

    try:
        reply = await call_gemini(history, new_parts)
        history.append({"role": "user", "parts": [{"text": f"[Photo] {caption}"}]})
        history.append({"role": "model", "parts": [{"text": reply}]})
        save_history(user_id, history)
        await update.message.reply_text(reply)
    except Exception as e:
        logger.error(f"Photo error: {e}")
        await update.message.reply_text(
            "I could not read that image clearly. Try taking the photo again with better lighting."
        )

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    voice_file = await context.bot.get_file(update.message.voice.file_id)

    async with httpx.AsyncClient(timeout=30.0) as client:
        audio_response = await client.get(voice_file.file_path)
        audio_base64 = base64.b64encode(audio_response.content).decode("utf-8")

    history = get_history(user_id)
    new_parts = [
        {
            "text": (
                "The student sent a voice note. "
                "First transcribe what they said, then respond as their Socratic tutor. "
                "Start with 'I heard: [transcription]' then a blank line then your response."
            )
        },
        {"inline_data": {"mime_type": "audio/ogg", "data": audio_base64}},
    ]

    try:
        reply = await call_gemini(history, new_parts)
        history.append({"role": "user", "parts": [{"text": "[Voice note]"}]})
        history.append({"role": "model", "parts": [{"text": reply}]})
        save_history(user_id, history)
        await update.message.reply_text(reply)
    except Exception as e:
        logger.error(f"Voice error: {e}")
        await update.message.reply_text(
            "I could not process that voice note. Try typing your question instead."
        )

# ── Main ─────────────────────────────────
def main():
    logger.info("Starting Thinkr bot...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    logger.info("Thinkr bot is running!")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
