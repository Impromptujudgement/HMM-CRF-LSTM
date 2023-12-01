import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def word2features(sent, i, language):
    assert language in ["English", "Chinese"], "Input is Illegal."
    global features
    if language == "English":
        word = sent[i]
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word[-1:]': word[-1:],
            'word[0]': word[0],
            'word[:2]': word[:2],
            'word[:3]': word[:3],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }
        # 该字的前一个字
        if i > 0:
            word1 = sent[i - 1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.isdigit()': word1.isdigit(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
            })
        else:
            features['BOS'] = True
        if i > 1:
            word2 = sent[i - 2]
            features.update({
                '-2:word.lower()': word2.lower(),
                '-2:word.isdigit()': word2.isdigit(),
                '-2:word.istitle()': word2.istitle(),
                '-2:word.isupper()': word2.isupper(),
            })
        if i > 2:
            word3 = sent[i - 3]
            features.update({
                '-3:word.lower()': word3.lower(),
                '-3:word.isdigit()': word3.isdigit(),
                '-3:word.istitle()': word3.istitle(),
                '-3:word.isupper()': word3.isupper(),
            })
        if i < len(sent) - 1:
            word1 = sent[i + 1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.isdigit()': word1.isdigit(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
            })
        else:
            features['EOS'] = True
    #     if i < len(sent) - 2:
    #         word2 = sent[i + 2]
    #         features.update({
    #             '+2:word.lower()': word2.lower(),
    #             '+2:word.isdigit()': word2.isdigit(),
    #             '+2:word.istitle()': word2.istitle(),
    #             '+2:word.isupper()': word2.isupper(),
    #         })
    #     #该字的后三个字
    #     if i < len(sent) - 3:
    #         word3 = sent[i + 3]
    #         features.update({
    #             '+3:word.lower()': word3.lower(),
    #             '+3:word.isdigit()': word3.isdigit(),
    #             '+3:word.istitle()': word3.istitle(),
    #             '+3:word.isupper()': word3.isupper(),
    #         })
    elif language == "Chinese":
        word = sent[i]
        features = {
            'bias': 1.0,
            'word': word,
            'word.isdigit()': word.isdigit(),
            'word.is_chinese_punctuation()': is_chinese_punctuation(word),
        }
        # 该字的前一个字
        if i > 0:
            word1 = sent[i - 1]
            words = word1 + word
            features.update({
                '-1:word': word1,
                '-1:words': words,
                '-1:word.isdigit()': word1.isdigit(),
                '-1:word.is_chinese_punctuation()': is_chinese_punctuation(word1),
            })
        else:
            # 添加开头的标识 BOS(begin of sentence)
            features['BOS'] = True
        # 该字的前两个字
        if i > 1:
            word2 = sent[i - 2]
            word1 = sent[i - 1]
            words = word1 + word2 + word
            features.update({
                '-2:word': word2,
                '-2:words': words,
                '-2:word.isdigit()': word2.isdigit(),
                '-2:word.is_chinese_punctuation()': is_chinese_punctuation(word2),
            })
        # 该字的前三个字
        if i > 2:
            word3 = sent[i - 3]
            word2 = sent[i - 2]
            word1 = sent[i - 1]
            words = word1 + word2 + word3 + word
            features.update({
                '-3:word': word3,
                '-3:words': words,
                '-3:word.isdigit()': word3.isdigit(),
                '-3:word.is_chinese_punctuation()': is_chinese_punctuation(word3),
            })
        # 该字的后一个字
        if i < len(sent) - 1:
            word1 = sent[i + 1]
            words = word1 + word
            features.update({
                '+1:word': word1,
                '+1:words': words,
                '+1:word.isdigit()': word1.isdigit(),
                '+1:word.is_chinese_punctuation()': is_chinese_punctuation(word1),
            })
        else:
            # 若改字为句子的结尾添加对应的标识end of sentence
            features['EOS'] = True
        # 该字的后两个字
        if i < len(sent) - 2:
            word2 = sent[i + 2]
            word1 = sent[i + 1]
            words = word + word1 + word2
            features.update({
                '+2:word': word2,
                '+2:words': words,
                '+2:word.isdigit()': word2.isdigit(),
                '+2:word.is_chinese_punctuation()': is_chinese_punctuation(word2),
            })
        # 该字的后三个字
        if i < len(sent) - 3:
            word3 = sent[i + 3]
            word2 = sent[i + 2]
            word1 = sent[i + 1]
            words = word + word1 + word2 + word3
            features.update({
                '+3:word': word3,
                '+3:words': words,
                '+3:word.isdigit()': word3.isdigit(),
                '+3:word.is_chinese_punctuation()': is_chinese_punctuation(word3),
            })
    return features


def sent2features(sent, language):
    return [word2features(sent, i, language) for i in range(len(sent))]


def is_chinese_punctuation(s):
    unicode_val = ord(s)
    if (0x3000 > unicode_val or unicode_val > 0x303F) and (0xFF01 > unicode_val or unicode_val > 0xFFEF):
        return False
    else:
        return True


def prepare_sequence(seq, to_ix):
    # 迭代输入序列中的每个词汇，并根据映射字典将其转换为索引
    idxs = [to_ix.get(w, to_ix['<unk>']) for w in seq]  # 使用get方法处理未知词汇，将未知词映射到特殊标记的索引
    return torch.tensor(idxs, dtype=torch.long).to(device)


def extend_maps(word2id, tag2id):
    word2id['<unk>'] = len(word2id)
    tag2id['<start>'] = len(tag2id)
    tag2id['<end>'] = len(tag2id)
    return word2id, tag2id
