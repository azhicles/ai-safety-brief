"""Application entrypoint."""

from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler

from ai_safety_brief.bot.handlers import (
    brief_handler,
    callback_handler,
    get_runtime,
    help_handler,
    history_handler,
    more_handler,
    pause_handler,
    resume_handler,
    settings_handler,
    sources_handler,
    start_handler,
    status_handler,
    why_handler,
)
from ai_safety_brief.bot.runtime import Runtime
from ai_safety_brief.bot.scheduler import schedule_jobs
from ai_safety_brief.config import load_settings
from ai_safety_brief.db import Database
from ai_safety_brief.services import DigestPipeline, GroqRefiner, SourceCollector, Summarizer

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


async def error_handler(update: object, context) -> None:
    logger.exception("Update %s failed with %s", update, context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text("Something went wrong. Please try again.")


async def post_init(application: Application) -> None:
    runtime = get_runtime(application)
    await runtime.db.init()
    schedule_jobs(application)


def build_runtime() -> Runtime:
    settings = load_settings()
    db = Database(settings.db_path)
    collector = SourceCollector(settings)
    summarizer = Summarizer()
    refiner = GroqRefiner(settings.groq_api_key, settings.groq_model)
    pipeline = DigestPipeline(settings, db, collector, summarizer, refiner)
    return Runtime(settings=settings, db=db, collector=collector, pipeline=pipeline)


def main() -> None:
    runtime = build_runtime()
    if not runtime.settings.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN is not set.")
        return

    application = (
        Application.builder()
        .token(runtime.settings.telegram_bot_token)
        .post_init(post_init)
        .build()
    )
    application.bot_data["runtime"] = runtime

    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(CommandHandler("brief", brief_handler))
    application.add_handler(CommandHandler("more", more_handler))
    application.add_handler(CommandHandler("why", why_handler))
    application.add_handler(CommandHandler("status", status_handler))
    application.add_handler(CommandHandler("history", history_handler))
    application.add_handler(CommandHandler("sources", sources_handler))
    application.add_handler(CommandHandler("settings", settings_handler))
    application.add_handler(CommandHandler("pause", pause_handler))
    application.add_handler(CommandHandler("resume", resume_handler))
    application.add_handler(CallbackQueryHandler(callback_handler))
    application.add_error_handler(error_handler)

    logger.info("Starting AI Safety Brief")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
