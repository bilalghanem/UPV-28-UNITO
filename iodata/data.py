#imports
import warnings, json, ast, os, re
from os.path import join
import pandas as pd
from joblib import Parallel, delayed
from textblob import TextBlob
from tqdm import tqdm
import numpy as np

# config
warnings.filterwarnings("ignore")
np.random.seed(0)


def save_json(path, _object):
    with open(path, 'w') as fp:
        json.dump(_object, fp)

def read_json(path):
    with open(path, 'r') as fp:
        model = json.load(fp)
    return model

def hier_level(rep_id, hier_structure, count, found):
    if type(hier_structure) == str:
        hier_structure = ast.literal_eval(hier_structure)
    if rep_id in hier_structure:
        return count, 1
    else:
        for key, val in hier_structure.items():
            if len(val) > 0:
                count, found = hier_level(rep_id, val, count+1, 0)
                if found == 1:
                    break
        if found == 1:
            return count, found
        else:
            return count-1, found

def load_hier_file(path):
    with open(join(path, 'structure.json')) as file:
        structure = json.load(file)
        file.close()
    return structure

def load_replies(path):
    path = join(path, r'replies/')
    replies = dict()
    all_replies = sorted(os.listdir(path))

    for rep_thread in all_replies:
        text = ""
        content = ""
        with open(join(path, rep_thread)) as reply:
            json_file = json.load(reply)

            if 'text' in json_file:
                text = json_file['text']
                content = json_file
            else:
                try:
                    content = json_file['data']
                    text = json_file['data']['body']
                except:
                    text = 'commentbilal'
                    content = '{}'

            replies.update({re.sub(r'.json', '', rep_thread): (text, content)})
    return replies

def load_source(path):
    path = join(path, r'source-tweet/')
    source = sorted(os.listdir(path))
    text = ''
    content = ''
    for source_tweet in source:
        with open(join(path, source_tweet)) as reply:
            json_file = json.load(reply)
            if 'text' in json_file:
                text = json_file['text']
                content = json_file
            else:
                text = json_file['data']['children'][0]['data']['title']
                content = json_file['data']['children'][0]['data']
    return text, content

def load_labels(path, label_files):
    labels = {}
    for lf in label_files:
        with open(join(path, lf)) as labels_json:
            labels.update(json.load(labels_json)['subtaskaenglish'])
    return labels

class get_parent_childs():
    def __init__(self, tree=dict(), target=""):
        self.parent = 'source'
        self.child = 0
        self.get(tree, target)

    def get(self, struct_dict, target):
        for item in struct_dict:
            childs = struct_dict[item]
            if item == target:
                self.child = len(childs)
                return 1
            if len(childs):
                res = self.get(childs, target)
                if res == 1:
                    self.parent = item

def read_twitter_data(path, dataset_source):
    dataset = []
    listOfEvents = sorted(os.listdir(path))
    for event in listOfEvents:
        list_event_threads = sorted(os.listdir(join(path, event)))
        for thread in list_event_threads:
            struct_file = load_hier_file(join(path, event, thread))
            dict_replies_content = load_replies(join(path, event, thread))
            source_text, source_content = load_source(join(path, event, thread))

            parent_obj = get_parent_childs(struct_file, thread)
            dataset.append([str(event), str('source'), str(thread), str(thread), str(source_text), str(source_content), int(1), str(dataset_source), str(parent_obj.parent), str(parent_obj.child)])

            for rep_id, reply_text_content in dict_replies_content.items():
                count_hier_level, __ = hier_level(rep_id, struct_file, 1, 0)
                parent_obj = get_parent_childs(struct_file, rep_id)
                dataset.append([str(event), str('reply'), str(thread), str(rep_id), str(reply_text_content[0]), str(reply_text_content[1]), int(count_hier_level), str(dataset_source), str(parent_obj.parent), str(parent_obj.child)])
    return dataset

def read_reddit_data(path, dataset_source):
    dataset = []
    list_event_threads = sorted(os.listdir(path))
    for thread in list_event_threads:
        struct_file = load_hier_file(join(path, thread))

        dict_replies_content = load_replies(join(path, thread))
        source_text, source_content = load_source(join(path, thread))

        parent_obj = get_parent_childs(struct_file, thread)
        dataset.append([str('event_'+thread), str('source'), str(thread), str(thread), str(source_text), str(source_content), int(1), str(dataset_source), str(parent_obj.parent), str(parent_obj.child)])

        for rep_id, reply_text_content in dict_replies_content.items():
            count_hier_level, __ = hier_level(rep_id, struct_file, 1, 0)
            parent_obj = get_parent_childs(struct_file, rep_id)
            dataset.append([str('event_'+thread), str('reply'), str(thread), str(rep_id), str(reply_text_content[0]), str(reply_text_content[1]), int(count_hier_level), str(dataset_source), str(parent_obj.parent), str(parent_obj.child)])
    return dataset


def load_data(dir_path, test_data=False, correcting=False):
    dataPath = join(dir_path, 'iodata/rumoureval-2019-{}-data'.format('training' if not test_data else 'test'))
    dataset = []
    if not os.path.isfile('iodata/{}'.format('training' if not test_data else 'test')):
        labels = {}
        data_sources = sorted(os.listdir(dataPath))
        for d_source in data_sources:
            if os.path.isdir(join(dataPath, d_source)):
                if d_source.__contains__('twitter'):
                    dataset.extend(read_twitter_data(join(dataPath, d_source), d_source))
                else:
                    dataset.extend(read_reddit_data(join(dataPath, d_source), d_source))

        dataset = pd.DataFrame(dataset, columns=['event', 'is_source', 'source_id', 'id', 'text', 'content', 'level', 'dataset_source', 'parent', 'childs'])

        if not test_data:
            labels.update(load_labels(dataPath, ['dev-key.json', 'train-key.json']))
            dataset['label'] = dataset['id'].map(lambda x: str(labels[str(x)]) if str(x) in labels else '0')
        else:
            labels.update(load_labels(dataPath, ['final-eval-key.json']))
            dataset['label'] = dataset['id'].map(lambda x: str(labels[str(x)]) if str(x) in labels else '0')

        dataset = dataset.infer_objects()
        dataset.sort_values(['dataset_source', 'event', 'is_source', 'id'], ascending=[True, True, False, True], inplace=True)
        dataset = dataset.reset_index(drop=True)

        if correcting:
            loop = tqdm(dataset['text'].tolist())
            loop.set_description('TextBlob correcting')
            corrected_text = Parallel(n_jobs=-1)(delayed(lambda x: str(TextBlob(x).correct()))(sentence) for sentence in loop)
            dataset['text'] = corrected_text

        ## For the separator
        dataset['text'] = dataset['text'].map(lambda x: re.sub(r'[\^]', '_', x))
        dataset['content'] = dataset['content'].map(lambda x: re.sub(r'[\^]', '_', x))
        dataset.to_pickle('iodata/{}'.format('training' if not test_data else 'test'))
    else:
        dataset = pd.read_pickle('iodata/{}'.format('training' if not test_data else 'test'))

    return dataset
