#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This program is dedicated to the public domain under the CC0 license.

"""
Simple Bot to reply to Telegram messages.
First, a few handler functions are defined. Then, those functions are passed to
the Dispatcher and registered at their respective places.
Then, the bot is started and runs until we press Ctrl-C on the command line.
Usage:
Basic Echobot example, repeats messages.
Press Ctrl-C on the command line or send a signal to the process to stop the
bot.
"""
import os
import logging
import subprocess
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

label = ''
# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.

def echo(update, context):
    """Echo the user message."""
    update.message.reply_text('Hi! Type /train , /predict, /video or /classes')

def classes(update, context):
    """Reply with current classes"""
    update.message.reply_text(os.listdir(os.getcwd()+'/images'))

def train(update, context):
    update.message.reply_text('Training model... This may take a while, you can come back later for the results.')
    proc = subprocess.Popen(['python3', 'train_torch.py', '-tTrue'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.communicate()[0]
    update.message.reply_text(f'Done! Model with {out.decode("utf-8")[-21:-2]}%')

def predict(update, context):
    update.message.reply_text('Go ahead, send a picture to predict.')

def video(update, context):
    global label
    label = ' '.join(context.args)
    if label == '':
        update.message.reply_text('Add the label of the object after /video')
    else:
        update.message.reply_text('Go ahead, send a video to save.')

def process_video(update, context):
    global label
    video_file = update.message.video.get_file()
    video_file.download(f'input/{label}')
    update.message.reply_text(f'Video of a {label} received! Processing...')
    label = ''
    proc = subprocess.Popen(['python3', 'video2frames.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.communicate()[0]
    update.message.reply_text(out.decode("utf-8"))

def save_img(update, context):
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('input/photo')
    update.message.reply_text('Photo received! Processing...')

    proc = subprocess.Popen(['python3', 'predict.py',  '-iphoto'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = proc.communicate()[0]

    update.message.reply_text(out.decode("utf-8"))

def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater(open('token.txt').read()[:-1], use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram

    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(CommandHandler("train", train))
    dp.add_handler(CommandHandler("predict", predict))
    dp.add_handler(CommandHandler("video", video))
    dp.add_handler(CommandHandler("classes", classes))

    dp.add_handler(MessageHandler(Filters.text, echo))
    dp.add_handler(MessageHandler(Filters.video, process_video))
    dp.add_handler(MessageHandler(Filters.photo, save_img))


    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
