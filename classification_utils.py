import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
import re
import csv
import sqlite3
from shutil import copyfile

# from keras.models import Sequential
# from keras.layers import Dropout, MaxPool1D, Dense, Conv1D, LSTM, GRU, Embedding, Flatten, GlobalMaxPool1D, MaxPool2D
# from keras.preprocessing.text import Tokenizer, text_to_word_sequence, one_hot
# from keras.utils import to_categorical
# from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict


def load_from_database():
    db_file = '../Data/chatbot_clean_dataset.db'
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute("select * from multilabel_dataset")
        rows = cur.fetchall()
        copyfile('./data/augmentation.csv', './data/classifier_ds.csv')
        with open('./data/classifier_ds.csv', 'a') as file:
            writer = csv.writer(file)
            # writer.writerow(["text", "greet", "schedule_meeting", "plan_task", "query_plan", "discuss_task",
            #                 "report_progress", "query_progress", "request_feedback", "give_feedback",
            #                 "query_tool", "state_tool", "report_issue"])
            writer.writerows(rows)
    except sqlite3.Error as e:
        print(e)


def load_dataset(filepath):
    # data_frame = pd.read_csv(filepath, '\t\t\t', header=None, names=['id', 'start', 'end', 'msg', 'tag', 'out'])
    # data_frame = data_frame.drop(columns=['id', 'start', 'end', 'out'])
    data_frame = pd.read_csv(filepath, ',', header=None, names=['msg', 'tag'])
    data_frame = data_frame.sample(frac=1).reset_index(
        drop=True)  # shuffle the dataset for better performance in case it's sorted
    return data_frame


def plot_history(history, multilabel=False):
    if multilabel:
        acc = history.history['binary_accuracy']
        val_acc = history.history['val_binary_accuracy']
    else:
        acc = history.history['categorical_accuracy']
        val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'g', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    # ax = plt.gca()
    # ax.set_facecolor('white')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'g', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    # ax = plt.gca()
    # ax.set_facecolor('white')
    plt.title('Training and validation loss')
    plt.legend()
    plt.grid()


def get_weight_matrix(embed_model, vocab_size, dims, all_words):
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, dims))
    # step vocab, store vectors using the Tokenizer's integer mapping
    unk = []
    for i in range(len(all_words)):
        if all_words[i] in embed_model:
            weight_matrix[i + 1] = embed_model[all_words[i]]
        else:
            unk.append(i + 1)
            weight_matrix[i + 1] = np.zeros(dims)

    print("Found unknown words: ", [all_words[x - 1] for x in unk])
    unk_vec = np.average(weight_matrix, axis=0)
    for u in unk:
        weight_matrix[u] = unk_vec

    return weight_matrix


tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

Lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

normal_word = {
    r"\byeah\b": "yes",
    r"\byup\b": "yes",
    r"\byep\b": "yes",
    r"\bnope\b": "no",
    r"\bu\b": "you",
    r"\bur\b": "your",
    r"\bplz\b": "please",
    r"\bpls\b": "please",
    r"\bgr8\b": "great",
    r"\bcan't\b": "cannot",
    r"\bgonna\b": "going to",
    r"\bgunna\b": "going to",
    r"\bcuz\b": "because",
    r"\bi'm\b": "I am",
    r"\blet's\b": "let us",
    r"\bcant\b": "cannot",
    r"'s\b": "",
    r"'ll": " will",
    r"'ve": " have",
    r"n't": " not",
}


def normalize_msg(txt, remove_stop=True, normal=None, entities=None):
    """
        Takes a raw text message and returns a feature representation for it:
            abs_len : number of tokens in the message
            past_count : number of occurrences of past tense verbs.
            present_count : number of occurrences of present tense verbs.
            future_count : number of occurrences of future tense verbs.
            first_person : number of occurrences of first person pronouns.
            second_person : number of occurrences of second person pronouns.
            third_person : number of occurrences of third person pronouns.
            n_text : normalized text
    """
    txt = str(txt).lower().strip('"')
    txt = replace_patterns(txt, entities)
    tokens = word_tokenize(txt)
    tokens = replace_entities(tokens, entities)
    words = []
    first_person = 0
    second_person = 0
    third_person = 0
    past_count = 0
    present_count = 0
    future_count = 0
    q_score = 0
    req_score = 0
    sug_score = 0
    pos_score = 0
    neg_score = 0
    prev_token = (None, None)
    if len(tokens) > 0:
        token_tags = pos_tag(tokens)
        q_score = get_question_score(token_tags)
        req_score = get_request_score(token_tags)
        sug_score = get_suggest_score(token_tags)
        pos_score, neg_score = get_agreement_score(token_tags)
        for word, tag in token_tags:
            if re.fullmatch(r"\W+", word):  # no alphanumeric characters
                if word[0] == '?':  # question marks (one or more)
                    word = '?'
                else:
                    prev_token = (word, tag)
                    continue
            # count pronouns (must be done before stop words removal if needed)
            if tag in ['NN', 'PRP', 'PRP$', 'JJ']:
                # Note: lowercase 'I' is recognized as NN
                if word in ['i', 'we', 'me', 'us', 'my', 'our', 'mine', 'ours']:
                    first_person += 1
                elif word in ['you', 'your', 'yours']:
                    second_person += 1
                elif word in ['he', 'him', 'his', 'she', 'her', 'they', 'them', 'their']:
                    third_person += 1
            elif tag in ['VBD', 'VBN']:  # Past tense
                past_count += 1
            elif tag in ['VBP', 'VBZ'] or (
                    tag == 'VBG' and prev_token[0] in ['am', 'is', 'are'] and word != 'going'):  # Present tense
                present_count += 1
            elif (tag == 'MD' and word in ['will']) or word == 'to' and prev_token[0] == 'going':  # Future tense
                future_count += 1
            # remove stop words
            if remove_stop and word in stopwords.words('english'):
                prev_token = (word, tag)
                continue
            # normalize remaining words if required
            if normal == 'lemma':
                word = Lemmatizer.lemmatize(word, tag_map[tag[0]])
            elif normal == 'stem':
                word = stemmer.stem(word)
            prev_token = (word, tag)
            words.append(word)
    abs_len = len(words)
    out_text = " ".join(words) if (len(words) > 0) else None
    # print(out_text)
    return out_text, first_person, second_person, third_person, past_count, present_count, future_count, \
           q_score, req_score, sug_score, pos_score, neg_score, abs_len


ent_dict = {}

replacement_dict = {
    'person': " MABotPerson ",
    'subteam': ' MABotSubteam ',
    'tool': " MABotTool ",
    'artifact': " MABotArtifact ",
    'time': " MABotEntTime ",
    'date': " MABotEntDate ",
    'email': " MABotEntEmail ",
    'number': " MABotEntNumber ",
    'url': " MABotEntURL "
}


def load_dictionary(path):
    reader = csv.reader(open(path))
    for row in reader:
        key = row[0].lower()
        val = row[1].lower()
        ent_dict[key] = val


def filter_dict(expr_dict: dict, word: str):
    filtered_dict = {}
    for key in expr_dict.keys():
        if key.startswith(word):
            filtered_dict[key] = expr_dict[key]
    return filtered_dict


def replace_entities(tokens: list, actions=None):
    if actions is None:
        actions = {'person': 'remove', 'subteam': 'remove', 'tool': 'replace', 'artifact': 'replace', 'date': 'replace', 'time': 'replace'}
    mtoken = 0
    while mtoken < len(tokens):
        largest_match = ''  # index of largest matched key in filtered entities dictionary
        largest_len = 0
        tokens[mtoken] = re.sub(r"[^?\w]+", " ", tokens[mtoken])  # replace any other punctuation with a space
        if str(tokens[mtoken]).strip(' ') == '':
            del tokens[mtoken]
            continue  # skip to next token
        filtered_dict = filter_dict(ent_dict, tokens[mtoken])
        for key in filtered_dict:
            key_tokens = str(key).split(' ')
            match = False
            for ktoken in range(len(key_tokens)):
                if mtoken + ktoken >= len(tokens) or tokens[mtoken + ktoken] != key_tokens[ktoken]:
                    match = False
                    break
                else:
                    match = True
            if match:  # matched all key tokens
                if len(key_tokens) > largest_len:
                    largest_match = key  # save matched key
                    largest_len = len(key_tokens)
        if largest_match:  # found matching key
            entity_type = filtered_dict[largest_match]
            if entity_type in actions.keys():
                if actions[entity_type] == 'replace' and entity_type in replacement_dict.keys():
                    tokens[mtoken] = replacement_dict[entity_type]
                    del tokens[mtoken + 1: mtoken + largest_len]
                elif actions[entity_type] == 'remove':
                    del tokens[mtoken: mtoken + largest_len]
                    mtoken -= 1
                else:
                    pass
        mtoken += 1  # while loop counter
    return tokens


def replace_patterns(text, actions=None):
    """
    Takes a raw (lower case) text message and returns the text after replacing common patterns
    :param actions: action performed on every entity type
    :param text: Original text message
    :return: the text after replacing common patterns
    """
    if actions is None:
        actions = {
            'time': 'replace',
            'date': 'replace',
            'url': 'replace',
            'email': 'remove',
            'number': 'replace'
        }
    for word_regex in normal_word.keys():
        text = re.sub(word_regex, normal_word[word_regex], text)

    date_re = r'\b\d\d?[\\/\-.]\d\d?[\\/\-.]\d\d\d\d\b|\b\d\d\d\d[\\/\-.]\d\d?[\\/\-.]\d\d\?\b'  # DD/MM/YYYY or reversed
    time_re = r'\b\d\d? ?(: ?\d\d?( ?: ?\d\d?)? ?(pm|am)?|(pm|am))\b'  # hh:mm:ss am

    numeral_re = r'\b\d+(,?\d\d\d)*(\.\d+)?\b'  # real number with decimal places or thousand separator
    # url_re = r'\[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    url_re = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    email_re = r"\b[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}\b"
    repeated_re = r"(.)\1{2,}"

    if actions['url'] == 'replace':
        text = re.sub(url_re, replacement_dict['url'], text)
    elif actions['url'] == 'remove':
        text = re.sub(url_re, '', text)

    if actions['email'] == 'replace':
        text = re.sub(email_re, replacement_dict['email'], text)
    elif actions['email'] == 'remove':
        text = re.sub(email_re, '', text)

    if actions['date'] == 'replace':
        text = re.sub(date_re, replacement_dict['date'], text)
    elif actions['date'] == 'remove':
        text = re.sub(date_re, '', text)

    if actions['time'] == 'replace':
        text = re.sub(time_re, replacement_dict['time'], text)
    elif actions['time'] == 'remove':
        text = re.sub(time_re, '', text)

    if actions['number'] == 'replace':
        text = re.sub(numeral_re, replacement_dict['number'], text)
    elif actions['number'] == 'remove':
        text = re.sub(numeral_re, '', text)

    # replace repeated characters
    rep = re.findall(repeated_re, text)
    for rs in rep:
        text = re.sub(re.escape(rs[0]) + "{3,}", rs[0] * 2, text)

    return text


def score_parse_tree(t, tags_score):
    score = 0
    for st in t:
        if type(st) == nltk.tree.Tree:
            lbl = st.label()
            if lbl in tags_score.keys():
                score += tags_score[lbl]
            score_parse_tree(st, tags_score)
        elif type(st) == tuple:
            lbl = st[1]
            if lbl in tags_score.keys():
                score += tags_score[lbl]
    return score


def get_question_score(tokens: list):
    # set custom tags for parsing
    question_tags = {
        'what': 'WTQ',
        'which': 'WCQ',
        'whose': 'WSQ',
        'whom': 'WHQ',
        'who': 'WHQ',
        'when': 'WHQ',
        'where': 'WHQ',
        'why': 'WHQ',
        'how': 'HWQ',
        'do': 'DO',
        'does': 'DOSE',
        'did': 'DID',
        'can': 'MDL',
        'could': 'MDL',
        'will': 'MDL',
        'would': 'MDL',
        'shall': 'MDL',
        'should': 'MDL',
        'may': 'MDL',
        'might': 'MDL',
        'must': 'MDL',
        'is': 'IS',
        'are': 'ARE',
        'was': 'WAS',
        'were': 'WERE',
        'am': 'AM',
        'be': 'B',
        'have': 'HAVE',
        'has': 'HAS',
        '?': "QMK",
        'i': "I"
    }
    new_sent = []
    for word, tag in tokens:
        if word.lower() in question_tags.keys():
            new_tag = question_tags[word.lower()]
        else:
            new_tag = tag
        new_sent.append((word, new_tag))

    grammar = r"""
        NP: {<DT|PRP\$>?<JJ.*>*<NN|NNS>}        
            {<NNP|NNPS>+}                       
            {<PRP>}
            {<NP><IN><NP>}
        BE: {<IS|ARE|WAS|WERE|AM>} # verb to be
        MDL: {<MDL|DO|DOES|DID|>} # modal verbs
        MDP:{<MDL><NP><VB|DO|B>} # Modal verb question phrase

        QSTN: {<WTQ|WCQ|WSQ><NP><VB|VBZ|BE><NP>?}  
              {<WHQ|WTQ><VBZ|VBD|BE><NP>(<VB><NP>?)?}
              {<WHQ|WTQ><BE><VBZ>}
              {<WHQ|WTQ><MDP><NP>}
              {<WHQ><MDL><VB>}
              {<HWQ>(<VBN>|<VBG>|<JJ.*|RB.*>+)<NP>?((<BE><NP>)|<MDP>)?}
              {<MDP><NP>?}
              {<BE><EX>?<NP><VB|VBN|VBG|JJ.*|RB.*>}     
              {((<AM><I>)|(<BE><NP>))<NP>}               
              {<HVP|HVS><NP><VBN><NP>?}
              {<QSTN><QMK>}
    """
    parser = RegexpParser(grammar)
    res = parser.parse(new_sent)
    tags_score = {
        'QSTN': 1,
        'WTQ': 0.5,
        'WCQ': 0.5,
        'WSQ': 0.5,
        'WHQ': 0.5,
        'HWQ': 0.5,
        'QMK': 1
    }
    return score_parse_tree(res, tags_score)


def get_request_score(tokens: list):
    # set custom tags for parsing
    """
        examples:
        * I/we want/wish/hope/need to 
        * I/we would like to
        * would/could/can you/anyone/someone <VB>
        * can/could/may/might/ I <VB>
        * please 
        * if you can/could
    """
    request_tags = {
        "i": "FPS",  # first person subject (singular)
        "we": "FPS",  # first person subject (plural)
        "me": "FPO",  # first person object (singular)
        "us": "FPO",  # first person object (plural)
        "my": "FPP",  # first person possesive (singular)
        "our": "FPP",  # first person possesive (plural)
        "want": "RVB",  # request verb
        "wish": "RVB",  # request verb
        "hope": "RVB",  # request verb
        "need": "RVB",  # request verb
        "like": "LIK",
        "would": "MDV",  # modal verb
        "could": "CAN",  # modal verb
        "can": "CAN",  # modal verb
        "may": "MAY",  # modal verb
        "might": "MAY",  # modal verb
        "please": "PLZ",  # most common request verb
        "you": "SP",  # second person subject/object
        "your": "SPP",  # second person possessive
        "yours": "SPP",  # second person possessive
        "he": "TPS",  # third person subject
        "she": "TPS",  #
        "they": "TPS",
        "his": "TPP",
        "her": "TPP",
        "their": "TPP",
        "him": "TPS",
        "them": "TPO",
        "do": "VB",
        "be": "VB",
        "if": "IF",
        "anyone": "CAL",
        "someone": "CAL",
        "sombody": "CAL",
    }
    # replace POS tags with custom tags
    new_sent = []
    for word, tag in tokens:
        if word.lower() in request_tags.keys():
            new_tag = request_tags[word.lower()]
        else:
            new_tag = tag
        new_sent.append((word, new_tag))

    grammar = r"""
        NP:  {<DT|PRP\$>?<JJ.*>*<NN|NNS>}        
                {<NNP|NNPS>+}                       
                {<PRP>}
                {<NP><IN><NP>}
        REQ:{<FPS>(<RVB>|(<MDV><LIK>))(<NP>|(<TO><VB>))}
                {<IF><SP><CAN>}
        RQU:{<MDV|CAN><SP|CAL><VB>}
        CMD:{<VB><FPO><NP>?}
    """
    parser = RegexpParser(grammar)
    res = parser.parse(new_sent)
    tags_score = {
        'PLZ': 2,
        'RQU': 2,
        'REQ': 2,
        'CMD': 1,
        'TPS': -1,
        'FPS': -1,
        'SPS': 1
    }
    return score_parse_tree(res, tags_score)


def get_suggest_score(tokens: list):
    # set custom tags for parsing
    """
        examples:
        * we/you/he/she/they should/could/may/can <VB>
        * let's <VB>
        * i/we think/suggest/believe that ...
    """
    suggest_tags = {
        "i": "FPS",  # first person subject (singular)
        "we": "FPS",  # first person subject (plural)
        "me": "FPO",  # first person object (singular)
        "us": "FPO",  # first person object (plural)
        "would": "MDV",  # modal verb
        "could": "MDV",  # modal verb
        "can": "MDV",  # modal verb
        "may": "MDV",  # modal verb
        "might": "MDV",  # modal verb
        "shall": "SHD",
        "should": "SHD",
        "suggest": "SJV",
        "think": "SJV",
        "believe": "SJV",
        "see": "SJV",
        "you": "SP",  # second person subject/object
        "he": "TPS",  # third person subject
        "she": "TPS",  #
        "they": "TPS",
        "him": "TPS",
        "them": "TPO",
        "that": "THT",
        "do": "VB",
        "be": "VB",
        "let": "LET",
        "lets": "LTS",
        "must": "MST",
        "please": "PLZ",
        "how": "HOW",
        "about": "ABT",
        "idea": "OPN",
        "opinion": "OPN",
        "suggestion": "OPN",
        "is": "IS",
        "not": "NOT",
        '?': "QMK",
    }
    # replace POS tags with custom tags
    new_sent = []
    for word, tag in tokens:
        if word.lower() in suggest_tags.keys():
            new_tag = suggest_tags[word.lower()]
        else:
            new_tag = tag
        new_sent.append((word, new_tag))

    grammar = r"""
        NP: {<DT|PRP\$>?<JJ.*>*<NN|NNS>}        
            {<NNP|NNPS>+}                       
            {<NP><IN><NP>}

        SUG:{<FPS|SP|TPS|PRP|NP><SHD><NOT>?<RB>?<VB|VBP>}
            {<FPS|SP|TPS|PRP|NP><MDV><RB>?<VB|VBP>}
            {<FPS><SJV>}
            {<LET><FPO><RB>?<VB|VBP>}
            {<LTS><RB>?<VB|VBP>}
            {<HOW><ABT><NP><FPS|SP|TPS|PRP><MDV><RB>?<VB>}
            {<HOW><ABT><FPS|SP|TPS|PRP><RB>?<VB|VBP>}
            {<PRP\$><OPN><IS>}
    """
    ques_parser = RegexpParser(grammar)
    res = ques_parser.parse(new_sent)
    tags_score = {
        'MST': -0.5,
        #'QMK': -1,
        'PLZ': -0.5,
        'SUG': 1
    }
    # print(res)
    return score_parse_tree(res, tags_score)


def get_agreement_score(tokens: list):
    # set custom tags for parsing
    """
        examples:
        * we/I agree/like/
        * nice/good/fine/great/ok/amazing/wonderful/excellent/perfect/true/right/correct/clear
        * bad/incorrect/wrong/unclear/ambiguous/
        * we/I <BE> amazed grateful fine
        * we/I <BE> worried/
        * yes
        * no
    """
    request_tags = {
        "i": "FPS",  # first person subject (singular)
        "we": "FPS",  # first person subject (plural)
        "me": "FPO",  # first person object (singular)
        "us": "FPO",  # first person object (plural)
        "you": "SP",  # second person subject/object
        "that": "THT",
        "this": "THT",
        "do": "MDV",
        "did": "MDV",
        "can": "MDV",
        "could": "MDV",
        "please": "PLZ",
        "is": "IS",
        "are": "IS",
        "am": "IS",
        "not": "NOT",
        "ok": "OK",
        "okay": "OK",
        "fine": "OK",
        "nice": "OK",
        "fair": "OK",
        "perfect": "OK",
        "great": "OK",
        "excellent": "OK",
        "wonderful": "OK",
        "clear": "OK",
        "good": "OK",
        "right": "OK",
        "correct": "OK",
        "true": "OK",
        "bad": "BAD",
        "wrong": "BAD",
        "incorrect": "BAD",
        "unclear": "BAD",
        "false": "BAD",
        "with": "WTH",
        "yes": "YES",
        "no": "NO",
        "problem": "PRM",
        "worries": "PRM",
        "issues": "PRM",
        "issue": "PRM",
        "agree": "PVB",
        "understand": "PVB",
        "disagree": "NVB",
        "like": "PVB",
        "dislike": "NVB",
        "get": "PVB",
        "accept": "PVB",
        "think": "THK",
        "grateful": "PJJ",
        "amazed": "PJJ",
        "delighted": "PJJ",
        "convinced": "PJJ",
        "happy": "PJJ",
        "worried": "NJJ",
        "confused": "NJJ",
        "unhappy": "NJJ",
        "shocked": "NJJ",
        "too": "TOO",
        "also": "TOO",
        '?': "QMK",
    }
    # replace POS tags with custom tags
    new_sent = []
    for word, tag in tokens:
        if word.lower() in request_tags.keys():
            new_tag = request_tags[word.lower()]
        else:
            new_tag = tag
        new_sent.append((word, new_tag))

    grammar = r"""
        NP: {<NO|DT|PRP\$>?<JJ.*>*<NN|NNS>}        
            {<NNP|NNPS>+}                       
            {<NP><IN><NP>}

        NEG:{<THT|NP><IS><RB>?((<NOT><OK>)|<BAD>)}
            {<FPS><IS><RB>?((<NOT><OK|PJJ>)|<BAD|NJJ>)}
            {((<NOT><OK>)|<BAD>)<WTH><FPO>}
            {<FPS><RB>?<NVB>}
            {<FPS><RB>?<MDV><NOT><PVB|THK>}
            {<NOT><FPO>}

        POS:{<THT|NP><IS><RB>?((<NOT><BAD>)|<OK>)}
            {<FPS><IS><RB>?((<NOT><BAD|NJJ>)|<OK|PJJ>)}
            {(<OK>|(<NOT><BAD>))<WTH><FPO>}
            {<YES|OK>|(<NOT><BAD>)}
            {<FPS><RB>?<PVB>}
            {<FPS><RB>?<MDV><NOT><NVB>}
            {<FPO|FPS><TOO>}
            {<NO><PRM>}

        NEG:{<NO>|(<NOT><OK>)|<BAD>}    # lower priority than POS
    """
    ques_parser = RegexpParser(grammar)
    res = ques_parser.parse(new_sent)
    pos_score = {
        'POS': 1,
        'QMK': -0.5,
    }
    neg_score = {
        'NEG': 1,
        'QMK': -0.5,
    }
    return score_parse_tree(res, pos_score), score_parse_tree(res, neg_score)


# common knowledge dictionary
#load_dictionary('data/dictionary_basic.csv')

# dataset-specific dictionary
load_dictionary('data/dictionary.csv')

