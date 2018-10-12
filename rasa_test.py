#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 10:21:13 2018

@author: lawrence
"""
import matplotlib.pyplot as plt
import spacy
import logging, io, json, warnings
logging.basicConfig(level="INFO")
warnings.filterwarnings('ignore')


from rasa_core.actions import Action
from rasa_core.events import SlotSet

import requests

class CallImageAPI(Action):
    def name(self):
        return "action_call_image_api"
    
    def run(self, dispatcher, tracker, domain):
        pass
def pprint(o):
    # small helper to make dict dumps a bit prettier
    print(json.dumps(o, indent=2))
    
    
import rasa_nlu
import rasa_core
import spacy

print("rasa_nlu: {} rasa_core: {}".format(rasa_nlu.__version__, rasa_core.__version__))
print("Loading spaCy language model...")
print(spacy.load("en")("Hello world!"))


from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu import config

# loading the nlu training samples
training_data = load_data("nlu.md")

# trainer to educate our pipeline
trainer = Trainer(config.load("config.yml"))

# train the model!
interpreter = trainer.train(training_data)
    
# store it for future use
model_directory = trainer.persist("./models/nlu", fixed_model_name="current")
pprint(interpreter.parse("i'm sad"))


from rasa_nlu.evaluate import run_evaluation

run_evaluation("nlu.md", model_directory)



#from IPython.display import Image
#from rasa_core.agent import Agent
#
#agent = Agent('domain.yml')
#agent.visualize("stories.md", "story_graph.png", max_history=2)
#Image(filename="story_graph.png")




from rasa_core.policies import FallbackPolicy, KerasPolicy, MemoizationPolicy
from rasa_core.agent import Agent

# this will catch predictions the model isn't very certain about
# there is a threshold for the NLU predictions as well as the action predictions
fallback = FallbackPolicy(fallback_action_name="utter_unclear",
                          core_threshold=0.2,
                          nlu_threshold=0.6)

agent = Agent('domain.yml', policies=[MemoizationPolicy(), KerasPolicy()])

# loading our neatly defined training dialogues
training_data = agent.load_data('stories.md')

agent.train(
    training_data,
    validation_split=0.2,
    epochs=400
)

agent.persist('models/dialogue')

agent = Agent.load('models/dialogue', interpreter=model_directory)

print("Your bot is ready to talk! Type your messages here or send 'stop'")
while True:
    a = input()
    if a == 'stop':
        break
    responses = agent.handle_text(a)
    for response in responses:
        print(response["text"])