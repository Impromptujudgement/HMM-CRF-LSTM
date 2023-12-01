from models.HMM import HMM
import pickle
from test import check
from models.CRF import CRFModel
from models.util import *
import torch.optim as optim
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
import numpy as np


def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)


def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model


def write2file(word_lists, tags_lists, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        # 逐行写入单词和标签
        for word_list, tag_list in zip(word_lists, tags_lists):
            for word, tag in zip(word_list, tag_list):
                file.write(f'{word} {tag}\n')  # 将单词和标签按照制表符分隔，并换行
            file.write('\n')
    print(f"写入文件 '{file_path}' 完成。")


def hmm_train(train_set, word2id, tag2id, model_save_path):
    model = HMM(word2id, tag2id)
    model.train(train_set)
    save_model(model, model_save_path)


def hmm_eval(validation_set, model_save_path, output_path, language):
    model = load_model(model_save_path)
    validation_word_lists, _ = validation_set
    hmm_pred = model.predict(validation_word_lists)
    write2file(validation_word_lists, hmm_pred, output_path)
    if language == "English":
        check(language=language, gold_path="./English/validation.txt", my_path=output_path)
    else:
        check(language=language, gold_path="./Chinese/validation.txt", my_path=output_path)


def crf_train(train_set, language, model_save_path):
    crf_model = CRFModel()
    crf_model.train(train_set, language)
    save_model(crf_model, model_save_path)


def crf_eval(validation_set, model_save_path, output_path, language):
    model = load_model(model_save_path)
    validation_word_lists, _ = validation_set
    crf_pred = model.predict(validation_word_lists, language)
    write2file(validation_word_lists, crf_pred, output_path)
    if language == "English":
        check(language=language, gold_path="./English/validation.txt", my_path=output_path)
    else:
        check(language=language, gold_path="./Chinese/validation.txt", my_path=output_path)


def lstm_crf_train(model, train_set, val_set, word2id, tag2id, epochs, model_save_path, loss_picture_path):
    best_loss = float('inf')  # 初始化一个最大值作为最佳验证集损失
    best_model = None  # 保存损失最小时的模型
    # optimizer = optim.SGD(model.parameters( ), lr=0.01, weight_decay=1e-4)
    optimizer = optim.Adam(model.parameters())  # 使用Adam优化器
    loss_list1 = []
    loss_list2 = []
    train_word_lists, train_tag_lists = train_set
    loss = 0
    for _ in tqdm(range(epochs)):
        for sentence, tags in zip(train_word_lists, train_tag_lists):
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word2id).to(device)
            targets = torch.tensor([tag2id[t] for t in tags],
                                   dtype=torch.long).to(device)
            loss = model.neg_log_likelihood(sentence_in, targets)
            loss.backward()
            optimizer.step()
        loss_list1.append(loss.item())
        val_loss = lstm_crf_loss(model, val_set)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())  # 保存最佳模型的状态字典
        loss_list2.append(val_loss)
        model.train()
    if best_model is not None:
        model.load_state_dict(best_model)
        torch.save(model.state_dict(), model_save_path)  # 保存最佳模型
    plt.plot(np.arange(0, epochs, 1), loss_list1, 'b')
    plt.plot(np.arange(0, epochs, 1), loss_list2, 'k')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "validation"])
    plt.savefig(loss_picture_path)
    plt.show()
    return model


def lstm_crf_loss(model, val_set):
    model.eval()  # 设置模型为评估模式
    total_loss = 0
    val_word_lists, val_tag_lists = val_set
    with torch.no_grad():
        for sentence, tags in zip(val_word_lists, val_tag_lists):
            # 转换为模型需要的张量格式
            sentence_in = prepare_sequence(sentence, model.word2id)
            targets = torch.tensor([model.tag_to_ix[t] for t in tags],
                                   dtype=torch.long).to(device)

            # 计算模型的损失
            loss = model.neg_log_likelihood(sentence_in, targets)
            total_loss += loss.item()

        avg_loss = total_loss / len(val_word_lists)
    model.train()
    return avg_loss


def lstm_crf_eval(model, model_save_path, val_set, output_path, language):
    model.load_state_dict(torch.load(model_save_path))
    validation_word_lists, _ = val_set
    with torch.no_grad():
        print("开始对验证集分析")
        with open(output_path, 'w') as f:
            for sequence in tqdm(validation_word_lists):
                precheck_sent = prepare_sequence(sequence, model.word2id)
                _, list_data = model(precheck_sent)
                list_data_numpy = np.array(list_data)
                output_str = ''
                for word, tag in zip(sequence, list_data_numpy):
                    output_str += word + ' ' + model.id2tag[tag] + '\n'
                output_str += '\n'
                f.write(output_str)
        print("对验证集分析完毕")
        print(f"写入文件 '{output_path}' 完成。")
    if language == "English":
        check(language=language, gold_path="./English/validation.txt", my_path=output_path)
    else:
        check(language=language, gold_path="./Chinese/validation.txt", my_path=output_path)
