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

# ─────────────────────────────────────────
# CONFIG — paste your keys here
# ─────────────────────────────────────────
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_TOKEN_HERE")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
)

# ─────────────────────────────────────────
# SOCRATIC TUTOR SYSTEM PROMPT
# This is your #1 competitive advantage
# ─────────────────────────────────────────
SYSTEM_PROMPT = """You are Thinkr — a Socratic AI tutor for South African high school students (Grades 8–12).

YOUR CORE PHILOSOPHY:
You NEVER just hand students the answer. You guide them to discover it themselves.
This is what makes you different from every other AI tutor.

HOW YOU RESPOND:
1. First, acknowledge what the student is asking warmly and briefly
2. Ask ONE powerful guiding question that nudges them toward the answer
3. If they're stuck after 2 attempts, give a small hint — not the full answer
4. Only reveal the full solution after the student has genuinely tried
5. Always end with a short "Why does this matter?" to connect learning to real life

SUBJECTS YOU COVER (CAPS-aligned):
Mathematics, Physical Science, Life Sciences, Geography, History, English, 
Afrikaans, Economics, Accounting, Business Studies, Computer Applications Technology

YOUR TONE:
- Warm, encouraging, like a brilliant older sibling
- Never condescending, never preachy
- Use simple South African English (avoid jargon)
- Short sentences. Easy to read on a phone screen.
- Add a relevant emoji occasionally to keep it friendly

FOR PHOTOS/IMAGES:
- If a student sends a photo of a question or formula, read it carefully
- Treat it exactly like a typed question — guide them Socratically
- If the image is unclear, ask them to retake it

FOR VOICE NOTES:
- You will receive a transcription of what the student said
- Respond naturally as if they spoke to you

IMPORTANT RULES:
- Never do a student's homework for them outright
- Never give a full essay or assignment — guide the structure instead
- If a student is rude or frustrated, respond with extra warmth and patience
- Keep all responses under 200 words — this is WhatsApp, not a textbook
- Format using plain text only — no markdown, no bullet symbols, no bold
- Use numbered steps when showing working (1. 2. 3.)

Remember: Your goal is not to give answers. Your goal is to build a student who doesn't need you anymore."""

# ─────────────────────────────────────────
# CONVERSATION MEMORY (in-memory per user)
# Stores last 10 messages per student
# ─────────────────────────────────────────
conversation_history: dict[int, list[dict]] = {}

def get_history(user_id: int) -> list[dict]:
    return conversation_history.get(user_id, [])

def save_history(user_id: int, history: list[dict]):
    # Keep last 10 exchanges to stay within token limits
    conversation_history[user_id] = history[-20:]

def build_gemini_payload(history: list[dict], new_content: list[dict]) -> dict:
    """Build the full Gemini API request payload."""
    contents = []
    
    # Add conversation history
    for msg in history:
        contents.append(msg)
    
    # Add new user message
    contents.append({"role": "user", "parts": new_content})
    
    return {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 400,
            "temperature": 0.7,
        },
    }

async def call_gemini(payload: dict) -> str:
    """Call Gemini API and return the text response."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(GEMINI_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

# ─────────────────────────────────────────
# COMMAND HANDLERS
# ─────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversation_history[user_id] = []  # Reset history on /start
    
    await update.message.reply_text(
        "Hey! I'm Thinkr 🧠\n\n"
        "I'm your personal study buddy for Grades 8-12.\n\n"
        "I won't just give you answers — I'll help you actually understand.\n\n"
        "Send me:\n"
        "📝 A typed question\n"
        "📸 A photo of your textbook or notes\n"
        "🎤 A voice note\n\n"
        "What subject are you working on today?"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Here's what I can help with:\n\n"
        "Maths, Science, Biology, History, Geography, English, "
        "Afrikaans, Economics, Accounting, Business Studies, CAT\n\n"
        "Just ask your question in any format — text, photo, or voice note.\n\n"
        "Commands:\n"
        "/start - Start fresh conversation\n"
        "/help - Show this message\n"
        "/reset - Clear our chat history"
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conversation_history[user_id] = []
    await update.message.reply_text(
        "Done! Fresh start 🔄\n\nWhat would you like to study?"
    )

# ─────────────────────────────────────────
# MESSAGE HANDLERS
# ─────────────────────────────────────────
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_text = update.message.text
    
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )
    
    history = get_history(user_id)
    payload = build_gemini_payload(
        history, [{"text": user_text}]
    )
    
    try:
        reply = await call_gemini(payload)
        
        # Save to history
        history.append({"role": "user", "parts": [{"text": user_text}]})
        history.append({"role": "model", "parts": [{"text": reply}]})
        save_history(user_id, history)
        
        await update.message.reply_text(reply)
    except Exception as e:
        logging.error(f"Gemini error: {e}")
        await update.message.reply_text(
            "Hmm, something went wrong on my end. Try again in a moment! 🙏"
        )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )
    
    # Get the highest resolution photo
    photo = update.message.photo[-1]
    caption = update.message.caption or "Please help me with this question in the image."
    
    # Download the image from Telegram
    photo_file = await context.bot.get_file(photo.file_id)
    
    async with httpx.AsyncClient() as client:
        img_response = await client.get(photo_file.file_path)
        img_bytes = img_response.content
    
    # Encode to base64 for Gemini
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    
    history = get_history(user_id)
    new_content = [
        {"text": caption},
        {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": img_base64,
            }
        },
    ]
    
    payload = build_gemini_payload(history, new_content)
    
    try:
        reply = await call_gemini(payload)
        
        # Save text version to history
        history.append({
            "role": "user",
            "parts": [{"text": f"[Student sent a photo] {caption}"}],
        })
        history.append({"role": "model", "parts": [{"text": reply}]})
        save_history(user_id, history)
        
        await update.message.reply_text(reply)
    except Exception as e:
        logging.error(f"Photo error: {e}")
        await update.message.reply_text(
            "I couldn't read that image clearly. "
            "Could you try taking the photo again with better lighting? 📸"
        )

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id, action="typing"
    )
    
    # Download the voice note
    voice_file = await context.bot.get_file(update.message.voice.file_id)
    
    async with httpx.AsyncClient() as client:
        audio_response = await client.get(voice_file.file_path)
        audio_bytes = audio_response.content
    
    # Encode to base64 for Gemini
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    history = get_history(user_id)
    
    # Use Gemini to transcribe AND respond in one call
    new_content = [
        {
            "text": (
                "The student sent a voice note. "
                "First transcribe what they said, then respond as their Socratic tutor. "
                "Start your reply with 'I heard: [transcription]' on the first line, "
                "then a blank line, then your tutoring response."
            )
        },
        {
            "inline_data": {
                "mime_type": "audio/ogg",
                "data": audio_base64,
            }
        },
    ]
    
    payload = build_gemini_payload(history, new_content)
    
    try:
        reply = await call_gemini(payload)
        
        history.append({
            "role": "user",
            "parts": [{"text": "[Student sent a voice note]"}],
        })
        history.append({"role": "model", "parts": [{"text": reply}]})
        save_history(user_id, history)
        
        await update.message.reply_text(reply)
    except Exception as e:
        logging.error(f"Voice error: {e}")
        await update.message.reply_text(
            "I couldn't process that voice note. "
            "Try typing your question instead? ✍️"
        )

# ─────────────────────────────────────────
# MAIN — start the bot
# ─────────────────────────────────────────
def main():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset))
    
    # Messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    
    print("Thinkr bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
