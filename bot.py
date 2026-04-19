import os
import asyncio
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
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # only for photos

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN is not set!")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set!")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
)

# ── Socratic System Prompt ───────────────
SYSTEM_PROMPT = """You are Thinkr, a Socratic AI tutor for South African students from Grade 8 all the way through university level.

YOUR CORE PHILOSOPHY:
You NEVER just hand students the answer. You guide them to discover it themselves.
This is what makes you different from every other AI tutor.

DETECTING STUDENT LEVEL:
- If the student mentions matric, grade, high school, NSC or CAPS subjects, treat them as a high school student
- If the student mentions university, varsity, degree, module, semester, first year, second year, third year, honours, or mentions a university subject, treat them as a university student
- If unsure, ask: "Are you in high school or university?" before answering
- Adapt your depth, vocabulary and complexity accordingly

FOR HIGH SCHOOL STUDENTS (Grades 8-12):
- CAPS-aligned for all subjects
- Simple language, relatable examples, real-life connections
- Subjects: Mathematics, Mathematical Literacy, Physical Science, Life Sciences, Geography, History, English, Afrikaans, Economics, Accounting, Business Studies, Computer Applications Technology, Tourism, Agriculture

FOR UNIVERSITY STUDENTS:
- Go deeper, use academic language and proper terminology
- Reference relevant theories, frameworks, scholars and models where appropriate
- Subjects include but are not limited to:
  Calculus, Linear Algebra, Statistics, Physics, Chemistry, Thermodynamics, Mechanics, Electronics, Computer Science, Data Structures, Algorithms, Machine Learning, Microeconomics, Macroeconomics, Financial Accounting, Management Accounting, Corporate Finance, Business Law, Marketing, Strategic Management, Auditing, Taxation, Anatomy, Physiology, Pharmacology, Biochemistry, Pathology, Nursing Science, Public Health, Sociology, Psychology, Philosophy, Political Science, Law, History, Linguistics, Media Studies, Development Studies, Real Analysis, Abstract Algebra, Differential Equations, Numerical Methods, Probability Theory

HOW YOU RESPOND (both levels):
1. Acknowledge what the student is asking warmly and briefly
2. Ask ONE powerful guiding question that nudges them toward the answer
3. If they are stuck after 2 attempts give a small hint but not the full answer
4. Only reveal the full solution after the student has genuinely tried
5. For university students add relevant theory or scholar and a conceptual follow-up question
6. End with a short note connecting the topic to real life or their career

YOUR TONE:
- High school: warm and encouraging like a brilliant older sibling
- University: collegial and intellectually stimulating like a sharp postgrad tutor
- Always respectful, never condescending
- Use South African context and examples where possible such as JSE, Eskom, Constitutional Court, SARB

IMPORTANT RULES:
- Never write an essay or assignment for a student, guide the structure instead
- Never solve a full past paper, work through one question at a time
- Keep responses under 250 words, this is a messaging app not a textbook
- Use plain text only, no markdown, no bullet symbols, no bold
- Use numbered steps when showing mathematical or scientific working
- If a student is stressed about exams, acknowledge it warmly before tutoring"""

# ── Conversation Memory ──────────────────
conversation_history: dict = {}

def get_history(user_id: int) -> list:
    return conversation_history.get(user_id, [])

def save_history(user_id: int, history: list):
    conversation_history[user_id] = history[-20:]

# ── Groq API Call (text + voice) ─────────
async def call_groq(history: list, user_message: str) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "max_tokens": 500,
        "temperature": 0.7,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            GROQ_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

# ── Groq Whisper (voice transcription) ───
async def transcribe_voice(audio_bytes: bytes) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": ("voice.ogg", audio_bytes, "audio/ogg")},
            data={"model": "whisper-large-v3"},
        )
        response.raise_for_status()
        return response.json()["text"]

# ── Gemini API Call (photos only) ────────
async def call_gemini_vision(caption: str, img_base64: str) -> str:
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{
            "role": "user",
            "parts": [
                {"text": caption},
                {"inline_data": {"mime_type": "image/jpeg", "data": img_base64}},
            ]
        }],
        "generationConfig": {"maxOutputTokens": 500, "temperature": 0.7},
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(GEMINI_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

# ── Command Handlers ─────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conversation_history[update.effective_user.id] = []
    await update.message.reply_text(
        "Hey! I am Thinkr.\n\n"
        "I am your personal study buddy, whether you are in high school or at university.\n\n"
        "I cover everything from Grade 8 Maths all the way to university level "
        "Finance, Engineering, Law, Medicine, and more.\n\n"
        "I will not just give you answers. I will help you actually understand.\n\n"
        "Send me a typed question, a photo of your notes or textbook, or a voice note.\n\n"
        "Are you in high school or university?"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "I cover two levels:\n\n"
        "HIGH SCHOOL (Grades 8-12)\n"
        "Maths, Science, Biology, History, Geography, English, "
        "Afrikaans, Economics, Accounting, Business Studies, CAT\n\n"
        "UNIVERSITY\n"
        "Calculus, Statistics, Physics, Chemistry, Accounting, "
        "Finance, Economics, Law, Psychology, Anatomy, Computer Science, "
        "Engineering, Marketing, and much more\n\n"
        "Send your question as text, photo, or voice note.\n\n"
        "/start - Start fresh\n"
        "/reset - Clear chat history\n"
        "/help - Show this message"
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conversation_history[update.effective_user.id] = []
    await update.message.reply_text("Fresh start! High school or university?")

# ── Message Handlers ─────────────────────
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    history = get_history(user_id)
    try:
        reply = await call_groq(history, user_text)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})
        save_history(user_id, history)
        await update.message.reply_text(reply)
    except Exception as e:
        logger.error(f"Text error: {e}")
        await update.message.reply_text("Something went wrong. Please try again in a moment.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    if not GEMINI_API_KEY:
        await update.message.reply_text(
            "Photo questions are not set up yet. Please type your question instead."
        )
        return

    photo = update.message.photo[-1]
    caption = update.message.caption or "Please help me with this question in the image."
    photo_file = await context.bot.get_file(photo.file_id)

    async with httpx.AsyncClient(timeout=30.0) as client:
        img_response = await client.get(photo_file.file_path)
        img_base64 = base64.b64encode(img_response.content).decode("utf-8")

    try:
        reply = await call_gemini_vision(caption, img_base64)
        history = get_history(user_id)
        history.append({"role": "user", "content": f"[Photo] {caption}"})
        history.append({"role": "assistant", "content": reply})
        save_history(user_id, history)
        await update.message.reply_text(reply)
    except Exception as e:
        logger.error(f"Photo error: {e}")
        await update.message.reply_text(
            "I could not read that image. Try taking the photo again with better lighting, or type your question."
        )

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    voice_file = await context.bot.get_file(update.message.voice.file_id)
    async with httpx.AsyncClient(timeout=30.0) as client:
        audio_response = await client.get(voice_file.file_path)
        audio_bytes = audio_response.content

    try:
        transcript = await transcribe_voice(audio_bytes)
        logger.info(f"Transcribed: {transcript}")
        history = get_history(user_id)
        reply = await call_groq(history, transcript)
        history.append({"role": "user", "content": transcript})
        history.append({"role": "assistant", "content": reply})
        save_history(user_id, history)
        await update.message.reply_text(f"I heard: {transcript}\n\n{reply}")
    except Exception as e:
        logger.error(f"Voice error: {e}")
        await update.message.reply_text(
            "I could not process that voice note. Try typing your question instead."
        )

# ── Main ─────────────────────────────────
async def run():
    logger.info("Starting Thinkr bot...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    logger.info("Thinkr bot is running!")
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(run())
