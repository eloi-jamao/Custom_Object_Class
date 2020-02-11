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
import subprocess
import logging
#from predict import Custom_resnet, image_loader, decode_preds
import predict as pred

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

def execute(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = process.communicate()[0]
    exitCode = process.returncode

    if (exitCode == 0):
        return output
    else:
        raise ProcessException(command, exitCode, output)

# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.

def echo(update, context):
    """Echo the user message."""
    update.message.reply_text('Hi! Type /train , /predict, or /video')

def train(update, context):
    update.message.reply_text('Training model...')
    #Add training here somehow
    update.message.reply_text('Done!')

def predict(update, context):
    update.message.reply_text('Go ahead, send a picture to predict.')

def video(update, context):
    update.message.reply_text('Go ahead, send a video to save.')

def process_video(update, context):
    pass

def save_img(update, context):
    photo_file = update.message.photo[-1].get_file()
    photo_file.download('input/user_photo.jpg')
    update.message.reply_text('Photo received! Processing...')
    #out = os.system(f'python3 hello.py -i user_photo.jpg')
    out = execute('python3 hello.py -i /input/user_photo.jpg')
    update.message.reply_text(out)
    '''
    model = pred.Custom_resnet()
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    img_path = root + '/input/user_photo.jpg'
    data = pred.image_loader(img_path)
    preds = model(data)
    #Print Results
    label, confidence = pred.decode_preds(preds)
    update.message.reply_text(f"Class: {label} ; Probability: {100*confidence}")
    '''

def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater("TOKEN", use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram

    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(CommandHandler("train", train))
    dp.add_handler(CommandHandler("predict", predict))
    dp.add_handler(CommandHandler("video", video))

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
