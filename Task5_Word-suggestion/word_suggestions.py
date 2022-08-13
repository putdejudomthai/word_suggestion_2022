import numpy as np
import pandas as pd
import json
from pprint import pprint
import requests
from doctest import OutputChecker
from gensim.models import Word2Vec

class Suggestions:
    def __init__(self):
        self.model_path = 'word2vec_TNC.model'
        self.model = Word2Vec.load(self.model_path)
        self.max_sug = 3
        self.vec_base = 0.5
        self.pos_use = ['VV','AV','AJ']

    def word_seg(self,text):
        url = 'http://nlp1.exp.superai.me:10100/seg'
        data = {"payload": text}
        r = requests.post(url=url,json = data)
        return r.json()['segged']
        
    def pos_tag(self,word_list):
        url = 'http://nlp1.exp.superai.me:10200/pos'
        data = {"payload": word_list}
        r = requests.post(url=url,json = data)
        return r.json()['pos']
     

    def segment_index(self,text, seg):
        segmented_list = []
        cind = 0
        id = 0
        for segmented in seg:
            cind += text.find(segmented)
            idx_start = cind
            cind += len(segmented)
            idx_end = cind
            segmented_list.append((id,idx_start, idx_end, segmented))
            id += 1
            text = text[len(segmented):]
        return segmented_list

    def select_word(self,pos_list,index_list):
        sug_list = []
        for index in range(len(pos_list)):
            pos = pos_list[index]
            if pos in self.pos_use:
                sug_list.append(index_list[index])
        return sug_list

    def word2vec(self,sug_list):
        sug_output = []
        for index in range(len(sug_list)):
            # print(index)
            try:
                word = sug_list[index][3]
                # print(word)
                output = self.model.wv.most_similar(word)
                sug_select = []
                # print(output)
                for sug,vec in output:
                    if vec > self.vec_base :
                        sug_select.append(sug)
                    if len(sug_select) == self.max_sug:
                        break
                # print(sug_select)
                if len(sug_select) != 0:
                    sug_dict = {
                        'id':sug_list[index][0],
                        'word':word,
                        'start_idx':sug_list[index][1],
                        'end_idx': sug_list[index][2],
                        'suggest': sug_select,
                        }

                    sug_output.append(sug_dict)

            except:
                pass
        return sug_output

    def word_suggest(self,input_json):
        #json to text
        # text = input_json.json()
        text = input_json

        #text to list of word by Word Segmentation
        word_list = self.word_seg(text)
        # print(word_list)

        #label index position from word of sentent
        index_list = self.segment_index(text,word_list)
        # print(index_list)

        #word to part of speech
        pos_list = self.pos_tag(word_list)
        # print(pos_list)
        
        #select word from rule
        sug_list = self.select_word(pos_list,index_list)
        # print(sug_list)
        
        #word to vec and 
        final_list = self.word2vec(sug_list)
        return final_list
        