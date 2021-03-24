#Importing librarys
import os
import sys

original_cwd = os.getcwd()
os.chdir(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/nmt")
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/setup")
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + "/core")
import argparse

from nmt import nmt
from setup.settings import hparams, out_dir, preprocessing
from setup.settings import score as score_settings

sys.path.remove(os.path.dirname(os.path.realpath(__file__)) + "/setup")
import tensorflow as tf

from core.scorer import score_answers
from core.sentence import normalize_new_lines, replace_in_answers
from core.tokenizer import apply_bpe, apply_bpe_load, detokenize, tokenize

sys.path.remove(os.path.dirname(os.path.realpath(__file__)) + "/core")
import random

import colorama
import discord
import nltk
from discord.ext import commands
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
nltk.download()
import json
import pickle
import tkinter
from tkinter import *

import numpy
import tflearn
from tensorflow.python.framework import ops

current_stdout = None


# Start inference "engine"
def do_start_inference(out_dir, hparams):

    # Silence all outputs
    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    global current_stdout
    current_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")

    # Modified autorun from nmt.py (bottom of the file)
    # We want to use original argument parser (for validation, etc)
    nmt_parser = argparse.ArgumentParser()
    nmt.add_arguments(nmt_parser)
    # But we have to hack settings from our config in there instead of commandline options
    flags, unparsed = nmt_parser.parse_known_args(['--'+k+'='+str(v) for k,v in hparams.items()])
    # And now we can run TF with modified arguments
    #tf.app.run(main=nmt.main, argv=[os.getcwd() + '\nmt\nmt\nmt.py'] + unparsed)

    # Add output (model) folder to flags
    flags.out_dir = out_dir

    # Make hparams
    hparams = nmt.create_hparams(flags)

    ## Train / Decode
    if not tf.gfile.Exists(flags.out_dir):
        nmt.utils.print_out("# Model folder (out_dir) doesn't exist")
        sys.exit()

    # Load hparams from model folder
    hparams = nmt.create_or_load_hparams(flags.out_dir, hparams, flags.hparams_path, save_hparams=False)

    # Choose checkpoint (provided with hparams or last one)
    if not flags.ckpt:
        flags.ckpt = tf.train.latest_checkpoint(flags.out_dir)

    # Create model
    model_creator = nmt.inference.get_model_creator(hparams)
    infer_model = nmt.inference.model_helper.create_infer_model(model_creator, hparams, None)
    sess, loaded_infer_model = nmt.inference.start_sess_and_load_model(infer_model, flags.ckpt)

    return (sess, infer_model, loaded_infer_model, flags, hparams)


# Inference
def do_inference(infer_data, sess, infer_model, loaded_infer_model, flags, hparams):

    # Disable TF logs for a while
    # Workaround for bug: https://github.com/tensorflow/tensorflow/issues/12414
    # Already fixed, available in nightly builds, but not in stable version
    # Maybe that will stay here to silence any outputs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    global current_stdout
    if not current_stdout:
        current_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    # With existing session
    with infer_model.graph.as_default():

        # Run model (translate)
        sess.run(
            infer_model.iterator.initializer,
            feed_dict={
                infer_model.src_placeholder: infer_data,
                infer_model.batch_size_placeholder: hparams.infer_batch_size
            })


        # calculate number of translations to be returned
        if hparams.infer_mode == "greedy":
            num_translations_per_input = 1
        elif hparams.infer_mode == "beam_search":
            num_translations_per_input = min(hparams.num_translations_per_input, hparams.beam_width)

        answers = []
        while True:
            try:

                nmt_outputs, _ = loaded_infer_model.decode(sess)

                if hparams.infer_mode != "beam_search":
                    nmt_outputs = nmt.inference.nmt_model.np.expand_dims(nmt_outputs, 0)

                batch_size = nmt_outputs.shape[1]

                for sent_id in range(batch_size):

                    # Iterate through responses
                    translations = []
                    for beam_id in range(num_translations_per_input):

                        if hparams.eos:
                            tgt_eos = hparams.eos.encode("utf-8")

                        # Select a sentence
                        output = nmt_outputs[beam_id][sent_id, :].tolist()

                        # If there is an eos symbol in outputs, cut them at that point
                        if tgt_eos and tgt_eos in output:
                            output = output[:output.index(tgt_eos)]
                        print(output)

                        # Format response
                        if hparams.subword_option == "bpe":  # BPE
                            translation = nmt.utils.format_bpe_text(output)
                        elif hparams.subword_option == "spm":  # SPM
                            translation = nmt.utils.format_spm_text(output)
                        else:
                            translation = nmt.utils.format_text(output)

                        # Add response to the list
                        translations.append(translation.decode('utf-8'))

                    answers.append(translations)

            except tf.errors.OutOfRangeError:
                print("end")
                break

        # bug workaround end
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        sys.stdout.close()
        sys.stdout = current_stdout
        current_stdout = None

        return answers


# Fancy way to start everything on first inference() call
def start_inference(question):

    global inference_helper, inference_object

    # Start inference, set global tuple with model, flags and hparams
    inference_object = do_start_inference(out_dir, hparams)

    # First inference() call calls that method
    # Now we have everything running, so replace inference() with actual function call
    inference_helper = lambda question: do_inference(question, *inference_object)

    # Load BPE join pairs
    if preprocessing['use_bpe']:
        apply_bpe_load()

    # Rerun inference() call
    return inference_helper(question)

# Model, flags and hparams
inference_object = None

# Function call helper (calls start_inference on first call, then do_inference)
inference_helper = start_inference

# Main inference function
def inference(questions, print = False):

    # Change current working directory (needed to load relative paths properly)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # Process questions
    answers_list = process_questions(questions)
    #answers = answers_list[0]

    # Revert current working directory
    os.chdir(original_cwd)

    # Return (one or more?)
    if not isinstance(questions, list):
        return answers_list[0]
    else:
        return answers_list

# Internal inference function (for direct call)
def inference_internal(questions):

    # Process questions and return
    return process_questions(questions, True)


# Get index and score for best answer
def get_best_score(answers_score):

    # Return first best scored response
    if score_settings['pick_random'] is None:
        max_score = max(answers_score)
        if max_score >= score_settings['bad_response_threshold']:
            return (answers_score.index(max_score), max_score)
        else:
            return (-1, None)

    # Return random best scored response
    elif score_settings['pick_random'] == 'best_score':
        indexes = [index for index, score in enumerate(answers_score) if score == max(answers_score) and score >= score_settings['bad_response_threshold']]
        if len(indexes):
            index = random.choice(indexes)
            return (index, answers_score[index])
        else:
            return (-1, None)

    # Return random response with score above threshold
    elif score_settings['pick_random'] == 'above_threshold':
        indexes = [index for index, score in enumerate(answers_score) if score > (score_settings['bad_response_threshold'] if score_settings['bad_response_threshold'] >= 0 else max(score)+score_settings['bad_response_threshold'])]
        if len(indexes):
            index = random.choice(indexes)
            return (index, answers_score[index])
        else:
            return (-1, None)

    return (0, score_settings['starting_score'])


# Process question or list of questions
def process_questions(questions, return_score_modifiers = False):

    # Make a list
    if not isinstance(questions, list):
        questions = [questions]

    # Clean and tokenize
    prepared_questions = []
    for question in questions:
        question = question.strip()
        prepared_questions.append(apply_bpe(tokenize(question)) if question else '##emptyquestion##')

    # Run inference
    answers_list = inference_helper(prepared_questions)

    # Process answers
    prepared_answers_list = []
    for index, answers in enumerate(answers_list):
        answers = detokenize(answers)
        answers = replace_in_answers(answers)
        answers = normalize_new_lines(answers)
        answers_score = score_answers(questions[index], answers)
        best_index, best_score = get_best_score(answers_score['score'])

        if prepared_questions[index] == '##emptyquestion##':
            prepared_answers_list.append(None)
        elif return_score_modifiers:
            prepared_answers_list.append({'answers': answers, 'scores': answers_score['score'], 'best_index': best_index, 'best_score': best_score, 'score_modifiers': answers_score['score_modifiers']})
        else:
            prepared_answers_list.append({'answers': answers, 'scores': answers_score['score'], 'best_index': best_index, 'best_score': best_score})

    return prepared_answers_list


def bagOWords(string,words):
    """
    """

    bag=[0 for _ in range(len(words))]
    stringWords = nltk.word_tokenize(string)
    stringWords =[stemmer.stem(word.lower()) for word in stringWords]

    for x in stringWords:
        for i, y in enumerate(words):
            if y == x:
                bag[i]=1
    return numpy.array(bag)


def send():
    """
    """

    msg =EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg !='':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END,"You: "+msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        res = start(msg)
        ChatLog.insert(END,"Alan: "+ res +'\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


def start(msg):
    """
    """

    # print('Welcome to Alan, Here you can begin your converstaion.(Enter "exit" to quit)')
    # while True:
    #     inp =input("User: ")
    #     if inp.lower()=="exit":
    #         break
    #
    #     conclusion=model.predict([bagOWords(inp,words)])
    #     conclusionIndex = numpy.argmax(conclusion)
    #     tag = labels[conclusionIndex]
    #     for x in data["Library"]:
    #         if x['tag'] == tag:
    #             responses =x['responses']
    #
    #     print(random.choice(responses))

    msg = msg.lower()
    conclusion=model.predict([bagOWords(msg,words)])[0]
    conclusionIndex = numpy.argmax(conclusion)
    tag = labels[conclusionIndex]
    print(conclusion[conclusionIndex])
    if conclusion[conclusionIndex]>0.7:
        for x in data["Library"]:
            if x['tag'] == tag:
                responses =x['responses']
        return random.choice(responses)
    else:
        return "Sorry I do not understand"


"""
base=Tk()
base.title("Alan Turing")
base.geometry("600x700")
base.resizable(width=FALSE, height=FALSE)

ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")

ChatLog.config(state=NORMAL)
ChatLog.insert(END,"Welcome to Alan, Here you can begin your converstaion.\n\n")
ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
ChatLog.config(state=DISABLED)
ChatLog.yview(END)
scrollbar.place(x=576,y=6, height=586)
ChatLog.place(x=6,y=6, height=586, width=570)
EntryBox.place(x=128, y=601, height=90, width=465)
SendButton.place(x=3, y=601, height=90)
"""

# Discord Implementation
client = commands.Bot(command_prefix = '.')

@client.event
async def on_ready():
    print('Bot is ready.')

@client.command(aliases=['speak'])
async def talk(ctx, *, question):
    msg =question
    if msg !='':
        res = start(msg)
        await ctx.send(f'{res}')


#client.run('ODIyMjU5MzY2OTE2MTI4Nzc4.YFPqtQ.XLvGeE2kX9LxBFwEFCQNNvV3Rhw') #comment out if you want to use UI
#base.mainloop() #comment out if you want to run on discord


# Interactive mode
if __name__ == "__main__":

    # Input file
    if sys.stdin.isatty() == False:

        # Process questions
        answers_list = process_questions(sys.stdin.readlines())

        # Print answers
        for answers in answers_list:
            print(answers['answers'][answers['best_index']])

        sys.exit()

    # Interactive mode
    colorama.init()
    print("\n\nStarting interactive mode (first response will take a while):")

    # Specified model
    if len(sys.argv) >= 2 and sys.argv[1]:
        checkpoint = hparams['out_dir'] + str(sys.argv[1])
        hparams['ckpt'] = checkpoint
        print("Using checkpoint: {}".format(checkpoint))

# Prototype Deep Neural Net Learning using tflearn
    with open('library.json') as fp:
        data = json.load(fp)

    try:
        with open("pickle.pickle","rb") as f:
            words, labels, learning, output = pickle.load(f)
    except:
        #Creating arrays for  words labels and docx and docy
        #docx list of patternes
        #docy tag for words
        words =[]
        labels = []
        docsx =[]
        docsy=[]

        #Stemming words for json library
        for library in data["Library"]:
            for pattern in library["patterns"]:
                keyword = nltk.word_tokenize(pattern)
                words.extend(keyword)
                docsx.append(keyword)
                docsy.append(library["tag"])

                #if tag is in labels array do not add duplicates
                if library["tag"] not in labels:
                    labels.append(library["tag"])
        words = [stemmer.stem(single.lower()) for single in words if single not in "?"]
        words = sorted(list(set(words)))

        labels = sorted(labels)

        learning = []
        output = []
        emptyOut = [0 for _ in range(len(labels))]

        # for loop over enumerate(docsx) to create a bag of words
        for x, doc in enumerate(docsx):
            bag = []

            keyword = [stemmer.stem(single) for single in doc]
            for single in words:
                if single in keyword:
                    bag.append(1)
                else:
                    bag.append(0)

            outputRow = emptyOut[:]
            outputRow[labels.index(docsy[x])] = 1
            learning.append(bag)
            output.append(outputRow)
            with open("pickle.pickle","wb") as f:
                pickle.dump((words, labels, learning, output),f)

    learning = numpy.array(learning)
    output = numpy.array(output)

    ops.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(learning[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    model.fit(learning, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("ChatbotModel.tflearn")

    # QAs
    while True:
        question = input("\n> ")
        answers = inference_internal(question)[0]

        conclusion=model.predict([bagOWords(question,words)])[0]
        conclusionIndex = numpy.argmax(conclusion)
        tag = labels[conclusionIndex]

        # execute the main.py prototype
        if conclusion[conclusionIndex] > 0.7:
            for x in data["Library"]:
                if x['tag'] == tag:
                    responses =x['responses']
            print(random.choice(responses))

        # execute the NMT Alanbot
        else:

            if answers is None:
                print(colorama.Fore.RED + "! Question can't be empty" + colorama.Fore.RESET)
            else:
                for i, _ in enumerate(answers['scores']):
                    print("{}- {}{} [{}] {}{}{}".format(colorama.Fore.GREEN if answers['scores'][i] == max(answers['scores']) and answers['scores'][i] >= score_settings['bad_response_threshold'] else colorama.Fore.YELLOW if answers['scores'][i] >= score_settings['bad_response_threshold'] else colorama.Fore.RED, answers['answers'][i], colorama.Fore.RESET, answers['scores'][i], colorama.Fore.BLUE, answers['score_modifiers'][i] if score_settings['show_score_modifiers'] else '', colorama.Fore.RESET))

os.chdir(original_cwd)
