import os
import sys

from tensorflow.python.framework import ops

sys.path.append(os.path.realpath(os.path.dirname(__file__)))
sys.path.append(os.path.realpath(os.path.dirname(__file__)) + "/nmt")
import argparse
import random

import colorama
import discord
import nltk
import tensorflow as tf
import tflearn
from discord.ext import commands
from nltk.stem.lancaster import LancasterStemmer

import scoring
from core.sentence import replace_in_answers, score_answers
from core.tokenizer import detokenize, tokenize
from nmt import nmt
from setup.settings import hparams, out_dir

stemmer = LancasterStemmer()
nltk.download()
# from tensorflow.python.framework import ops
import json
import pickle
import tkinter
from tkinter import *

import numpy

current_stdout = None

# That will not be as easy as training script, as code relies on input and output file in deep levels of code
# It also outputs massive amount of info
# We have to make own script for inference, so we could:cd ..
# - use it in interactive mode
# - import for use in other code
# - use input and output of our choice (so, for example, file as input and console as output,
#   or even console as input and file as output (but why? ;) ), etc)
# Why that nmt module doesn't give us some easy to use interface?

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
    hparams = nmt.create_or_load_hparams(flags.out_dir, hparams, flags.hparams_path, save_hparams=True)

    # Choose checkpoint (provided with hparams or last one)
    if not flags.ckpt:
        flags.ckpt = tf.train.latest_checkpoint(flags.out_dir)

    # Create model
    if not hparams.attention:
        model_creator = nmt.inference.nmt_model.Model
    elif hparams.attention_architecture == "standard":
        model_creator = nmt.inference.attention_model.AttentionModel
    elif hparams.attention_architecture in ["gnmt", "gnmt_v2"]:
        model_creator = nmt.inference.gnmt_model.GNMTModel
    else:
        raise ValueError("Unknown model architecture")
    infer_model = nmt.inference.model_helper.create_infer_model(model_creator, hparams, None)

    return (infer_model, flags, hparams)

# Inference
def do_inference(phrase, infer_model, flags, hparams):

    infer_data = [phrase]

    # Disable TF logs for a while
    # Workaround for bug: https://github.com/tensorflow/tensorflow/issues/12414
    # Already fixed, available in nightly builds, but not in stable version
    # Maybe that will stay here to silence any outputs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    global current_stdout
    if not current_stdout:
        current_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    # Spawn new session
    with tf.Session(graph=infer_model.graph, config=nmt.utils.get_config_proto()) as sess:

        # Load model
        loaded_infer_model = nmt.inference.model_helper.load_model(infer_model.model, flags.ckpt, sess, "infer")

        # Run model (translate)
        sess.run(
            infer_model.iterator.initializer,
            feed_dict={
                infer_model.src_placeholder: infer_data,
                infer_model.batch_size_placeholder: hparams.infer_batch_size
            })


        # calculate number of translations to be returned
        num_translations_per_input = max(min(hparams.num_translations_per_input, hparams.beam_width), 1)
        translations = []

        try:
            nmt_outputs, _ = loaded_infer_model.decode(sess)
            if hparams.beam_width == 0:
                nmt_outputs = nmt.inference.nmt_model.np.expand_dims(nmt_outputs, 0)

            # Iterate through responses
            for beam_id in range(num_translations_per_input):

                if hparams.eos: tgt_eos = hparams.eos.encode("utf-8")

                # Select a sentence
                output = nmt_outputs[beam_id][0, :].tolist()

                # If there is an eos symbol in outputs, cut them at that point
                if tgt_eos and tgt_eos in output:
                    output = output[:output.index(tgt_eos)]

                # Format response
                if hparams.subword_option == "bpe":  # BPE
                    translation = nmt.utils.format_bpe_text(output)
                elif hparams.subword_option == "spm":  # SPM
                    translation = nmt.utils.format_spm_text(output)
                else:
                    translation = nmt.utils.format_text(output)

                # Add response to array
                translations.append(translation.decode('utf-8'))

        except tf.errors.OutOfRangeError:
            pass

        # bug workaround end
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        sys.stdout.close()
        sys.stdout = current_stdout
        current_stdout = None

        return translations

# Fancy way to start everything on first inference() call
def start_inference(question):

    global inference_helper, inference_object

    # Start inference, set global tuple with model, flags and hparams
    inference_object = do_start_inference(out_dir, hparams)

    # First inference() call calls that method
    # Now we have everything running, so replace inference() with actual function call
    inference_helper = lambda question: do_inference(tokenize(question), *inference_object)

    # Rerun inference() call
    return inference_helper(question)

# Model, flags and hparams
inference_object = None

# Function call helper (calls start_inference on first call, then do_inference)
inference_helper = start_inference

# Main inference function
def inference(question, include_blacklisted = True):
    answers = inference_helper(question)
    answers = detokenize(answers)
    answers = replace_in_answers(answers, 'answers')
    answers_rate = score_answers(answers)

    try:
        index = answers_rate.index(1)
        score = 1
    except:
        index = None

    if index is None and include_blacklisted:
        try:
            index = answers_rate.index(0)
            score = 0
        except:
            index = 0
            score = -1

    if index is None:
        index = 0
        score = -1

    return {'answers': answers, 'index': index, 'score': score}

# Internal inference function (for direct call)
def inference_internal(question):
    answers = inference_helper(question)
    answers = detokenize(answers)
    answers = replace_in_answers(answers, 'answers')
    answers_rate = score_answers(answers, 'answers')
    return (answers, answers_rate)

def bagOWords(string,words):
    bag=[0 for _ in range(len(words))]
    stringWords = nltk.word_tokenize(string)
    stringWords =[stemmer.stem(word.lower()) for word in stringWords]

    for x in stringWords:
        for i, y in enumerate(words):
            if y == x:
                bag[i]=1
    return numpy.array(bag)

# interactive mode
if __name__ == "__main__":

    print("\n\nStarting interactive mode (first response will take a while):")
    colorama.init()
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
        answers, answers_rate = inference_internal(question)
        ans_score = {}
        answers = inference_internal(question)[0]

        conclusion=model.predict([bagOWords(question,words)])[0]
        conclusionIndex = numpy.argmax(conclusion)
        tag = labels[conclusionIndex]

        # execute the main.py prototype
        if conclusion[conclusionIndex] > 0.85:
            for x in data["Library"]:
                if x['tag'] == tag:
                    responses =x['responses']
            print(">"+random.choice(responses))
        # execute the NMT Alanbot
        else:
            for i, answer in enumerate(answers):

                score = scoring.do_scoring(question, answer, answers_rate[i])
                ans_score[answer] = score

            scores = [v for k,v in ans_score.items()]
            max_score = max(scores)
            options = [k for k,v in ans_score.items() if v == max_score]
            choice_answer = random.choice(options)

            print(choice_answer)
                # maybe print the others? Anything else with a matching highscore green, yellow mid-range... red lowest?
