from tqdm import tqdm
from codecs import open
import shutil


def build_corpus(data_dir: str, make_vocab: object = True) -> object:
    """

    :rtype: object
    """
    word_lists = []
    tag_lists = []
    with open(data_dir, 'r', encoding='UTF-8') as f:
        word_list = []
        tag_list = []
        lines = f.readlines()
        print("开始构建数据集")
        for i in tqdm(range(len(lines))):
            if len(lines[i]) == 1:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []
            else:
                word, tag = lines[i].split()
                word_list.append(word)
                tag_list.append(tag)
        if word_list:
            word_lists.append(word_list)
        if tag_list:
            tag_lists.append(tag_list)
    data = (word_lists, tag_lists)
    print("数据集构建完毕\n")
    print_full_line('*')
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return data, word2id, tag2id
    else:
        return data


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps


def print_full_line(char='-'):
    terminal_width = shutil.get_terminal_size().columns
    line = char * terminal_width
    print(line)
