import os
import logging
import openai
import fitz  # PyMuPDF
import tempfile
from dotenv import load_dotenv
from flask import Flask, render_template
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes
)
import asyncio
import threading

# --- Load .env ---
load_dotenv()
BOT_TOKEN = os.getenv("BOT_KEY")
AI_KEY = os.getenv("AI_MODEL_KEY")
AI_MODEL = "meta-llama/llama-3-8b-instruct:free"

# --- OpenRouter AI Client ---
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=AI_KEY
)

# --- Logging ---
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# --- Flask App (for Render or other hosting) ---
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")  # Create a templates/index.html if needed

# --- Telegram Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to *ReadNova Bot*!\n\n"
        "Upload a PDF and ask questions about its content using `/ask Your question`.",
        parse_mode="Markdown"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "💡 *How to use ReadNova:*\n"
        "1. Upload a PDF.\n"
        "2. Wait for processing.\n"
        "3. Ask questions with `/ask What is this about?`\n\n"
        "I’ll give answers based only on the uploaded PDF.",
        parse_mode="Markdown"
    )

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📄 Reading your PDF, please wait...")

    try:
        file = await context.bot.get_file(update.message.document.file_id)
        file_bytes = await file.download_as_bytearray()

        text = ""
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()

        context.user_data["pdf_text"] = text
        logger.info(f"Extracted {len(text)} characters from PDF.")
        await update.message.reply_text("✅ PDF processed! Use `/ask` to ask questions.")

    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        await update.message.reply_text("❌ Failed to read the PDF. Try another file.")

async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pdf_text = context.user_data.get("pdf_text")
    question = " ".join(context.args)

    if not pdf_text:
        await update.message.reply_text("❗ Please upload a PDF first.")
        return
    if not question:
        await update.message.reply_text("❗ Please provide a question after `/ask`.")
        return

    await update.message.reply_text("🤔 Thinking...")

    try:
        system_prompt = (
            "You are an assistant that answers based only on the provided PDF text. "
            "If the answer cannot be found, say 'I couldn't find that in the document.'"
        )
        user_prompt = f"DOCUMENT:\n---\n{pdf_text}\n---\n\nQUESTION:\n{question}"

        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5
        )
        answer = response.choices[0].message.content
        for i in range(0, len(answer), 4096):
            await update.message.reply_text(answer[i:i+4096])

    except Exception as e:
        logger.error(f"AI error: {e}")
        await update.message.reply_text("⚠️ Failed to generate an answer. Try again later.")

# --- Bot Setup ---
def setup_bot():
    app_builder = (
        Application.builder()
        .token(BOT_TOKEN)
        .read_timeout(30)
        .write_timeout(30)
        .connect_timeout(30)
        .pool_timeout(10)
    )
    application = app_builder.build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("ask", ask))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))

    return application

def run_bot():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        application = setup_bot()
        logger.info("🤖 ReadNova Bot is running...")

        loop.run_until_complete(
            application.run_polling(
                stop_signals=None,
                timeout=30,
                bootstrap_retries=3,
                read_timeout=20,
                write_timeout=20,
                connect_timeout=20,
                pool_timeout=10
            )
        )

    except Exception as e:
        logger.error(f"Bot crashed: {e}")
        import time
        time.sleep(10)

    finally:
        loop.close()

# --- Start Everything ---
if __name__ == "__main__":
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()

    PORT = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=PORT, debug=False)
