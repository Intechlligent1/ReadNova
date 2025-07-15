import os
import logging
import threading
import asyncio
import openai
from dotenv import load_dotenv
import fitz  # PyMuPDF

from telegram import Update
from telegram.ext import (
    MessageHandler, Application, CommandHandler,
    filters, ContextTypes
)

# --- Configuration & Setup ---
load_dotenv()
BOT_TOKEN = os.getenv("BOT_KEY")
AI_MODEL_KEY = os.getenv("AI_MODEL_KEY")
AI_MODEL = "meta-llama/llama-3-8b-instruct:free"  # Recommended free model

# Set up the OpenAI client for OpenRouter
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=AI_MODEL_KEY,
)

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# --- Bot Command Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends a welcome message when the /start command is issued."""
    await update.message.reply_text(
        "Welcome to ReadNova Bot! 📚\n\n"
        "Please upload a PDF file, and I'll help you ask questions about it."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Shows the help message."""
    help_text = (
        "Here's how to use me:\n\n"
        "1. Upload any PDF document.\n"
        "2. Wait for me to confirm I've read it.\n"
        "3. Use the `/ask` command to ask a question about the document.\n\n"
        "*Example:*\n`/ask What is the main conclusion of this document?`"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles PDF file uploads, extracts text, and stores it."""
    logger.info("PDF received, starting processing...")
    await update.message.reply_text("Reading your PDF... This might take a moment. ⏳")

    try:
        # Download the file into memory
        file = await context.bot.get_file(update.message.document.file_id)
        file_bytes = await file.download_as_bytearray()

        # Extract text using PyMuPDF
        pdf_text = ""
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                pdf_text += page.get_text()

        # Store the extracted text in user_data for this specific user
        context.user_data['pdf_text'] = pdf_text
        logger.info(f"Successfully extracted {len(pdf_text)} characters from the PDF.")

        await update.message.reply_text(
            "✅ I've finished reading your document!\n\n"
            "You can now ask me questions using the `/ask` command."
        )
    except Exception as e:
        logger.error(f"Failed to process PDF: {e}")
        await update.message.reply_text(
            "❌ Sorry, I couldn't read that PDF. Please try another one."
        )

async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Answers a user's question based on the stored PDF text."""
    pdf_text = context.user_data.get('pdf_text')
    question = " ".join(context.args)

    # Check if a PDF has been uploaded
    if not pdf_text:
        await update.message.reply_text("Please upload a PDF document first before asking questions.")
        return

    # Check if the user provided a question
    if not question:
        await update.message.reply_text("Please provide a question after the /ask command.\n\n*Example:*\n`/ask What is the main topic?`", parse_mode="Markdown")
        return

    logger.info(f"Received question: {question}")
    await update.message.reply_text("Thinking... 🧠")

    try:
        # Construct the prompt for the AI
        system_prompt = (
            "You are an intelligent assistant. Based *only* on the provided document text, "
            "answer the user's question. If the answer is not found in the text, say so."
        )
        user_prompt = f"DOCUMENT TEXT:\n---\n{pdf_text}\n---\n\nQUESTION:\n{question}"

        # Call the OpenRouter API
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5,
        )

        answer = response.choices[0].message.content
        await update.message.reply_text(answer)

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        await update.message.reply_text("Sorry, I encountered an error while trying to answer. Please try again.")

# --- Main Bot Execution ---
def main() -> None:
    """Sets up the bot and starts polling for updates."""
    application = Application.builder().token(BOT_TOKEN).build()

    # Add handlers for commands and messages
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("ask", ask_question))
    application.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))

    logger.info("Bot is starting up...")
    application.run_polling()

if __name__ == '__main__':
    main()