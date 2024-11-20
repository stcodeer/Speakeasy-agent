import method
from recommendation import func_3
import llama
import subprocess

from speakeasypy import Speakeasy, Chatroom
from typing import List
import time

from rdflib.namespace import Namespace, RDF, RDFS, XSD
from rdflib.term import URIRef, Literal
import csv
import json
import rdflib
# from collections import defaultdict, Counter
# import locale
# _ = locale.setlocale(locale.LC_ALL, '')
# from _plotly_future_ import v4_subplots
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# import plotly.graph_objs as go
# init_notebook_mode(connected=True)
# import plotly.io as pio
# pio.renderers.default = 'jupyterlab+svg'

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2

username = "stormy-dragon"
password = "G5ndX2X8"

def to_text(x):
    x = str(x)
    if x.isdigit():
        x = int(x)
    return x

def adjust_format(ret):
    multi_ans = False

    for i in range(len(ret)):
        if len(ret[i]) == 1:
            ret[i] = to_text(ret[i][0])
            continue
        else:
            multi_ans = True
            ret[i] = list(ret[i])
            for j in range(len(ret[i])):
                ret[i][j] = to_text(ret[i][j])
            ret[i] = tuple(ret[i])
            
    if multi_ans:
        ret = str(ret).replace("), ", "),\n ")
    else:
        ret = str(ret).replace("', ", "',\n ").replace('", ', '",\n ')
    
    return ret.replace('–','-')

class Agent:
    def __init__(self, username, password):
        self.username = username
        # Initialize the Speakeasy Python framework and login.
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.

    def listen(self):
        while True:
            # only check active chatrooms (i.e., remaining_time > 0) if active=True.
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    # send a welcome message if room is not initiated
                    room.post_messages(f'Hello! This is a welcome message from {room.my_alias}.')
                    room.initiated = True
                # Retrieve messages from this chat room.
                # If only_partner=True, it filters out messages sent by the current bot.
                # If only_new=True, it filters out messages that have already been marked as processed.
                for message in room.get_messages(only_partner=True, only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new message #{message.ordinal}: '{message.message}' "
                        f"- {self.get_time()}")

                    # Implement your agent here #

                    # Send a message to the corresponding chat room using the post_messages method of the room object.
                    
                    try:
                        if "Y" in llama.classify_question(message.message):
                            # Recommendation
                            print("!type:! ", "Recommendation")
                            ret = func_3(llama.entitylist_extract(message.message))
                        else:
                            # Fatual & Embedding
                            print("!type:! ", "Fatual & Embedding")
                            ret = method.template_1(message.message)
                    except:
                        ret = "Apologies, but there is no corresponding answer in the database for your question."
                        
                    # ret = llama.refine_wording(ret)

                    room.post_messages(ret)
                    # room.post_messages(f"Received your message: '{message.message}' ")
                    
                    # Mark the message as processed, so it will be filtered out when retrieving new messages.
                    room.mark_as_processed(message)

                # Retrieve reactions from this chat room.
                # If only_new=True, it filters out reactions that have already been marked as processed.
                for reaction in room.get_reactions(only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new reaction #{reaction.message_ordinal}: '{reaction.type}' "
                        f"- {self.get_time()}")

                    # Implement your agent here #

                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)

            time.sleep(listen_freq)

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    command = ["ollama", "serve"]
    process = subprocess.Popen(command)
    time.sleep(10)
    
    demo_bot = Agent(username, password)
    demo_bot.listen()
    
    process.terminate()
