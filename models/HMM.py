import numpy as np
from tqdm import tqdm


class HMM:
    def __init__(self, word2id: dict, tag2id):
        # self.tag2id = {
        #     "O": 0,
        #     "B-PER": 1,
        #     "I-PER": 2,
        #     "B-ORG": 3,
        #     "I-ORG": 4,
        #     "B-LOC": 5,
        #     "I-LOC": 6,
        #     "B-MISC": 7,
        #     "I-MISC": 8
        # }
        self.tag2id = tag2id
        self.word2id = word2id
        self.n_tag = len(self.tag2id)
        self.n_char = len(self.word2id)
        self.epsilon = 1e-100  # 防止出现0
        self.id2tag = dict(zip(self.tag2id.values(), self.tag2id.keys()))
        self.A = np.zeros((self.n_tag, self.n_tag))
        self.B = np.zeros((self.n_tag, self.n_char))
        self.Pi = np.zeros(self.n_tag)

    def train(self, train_set):
        word_lists, tag_lists = train_set
        assert len(tag_lists) == len(word_lists)
        tag2id = self.tag2id
        word2id = self.word2id
        for word_list, tag_list in zip(word_lists, tag_lists):
            assert len(word_list) == len(tag_list)
            seq_len = len(tag_list)
            for i in range(seq_len - 1):
                current_tagid = tag2id[tag_list[i]]
                next_tagid = tag2id[tag_list[i + 1]]
                self.A[current_tagid][next_tagid] += 1
            for word, tag in zip(word_list, tag_list):
                tag_id = tag2id[tag]
                word_id = word2id[word]
                self.B[tag_id][word_id] += 1
            init_tagid = tag2id[tag_list[0]]
            self.Pi[init_tagid] += 1
        self.A[self.A == 0.] = self.epsilon
        self.A = self.A / np.sum(self.A, axis=1, keepdims=True)
        #         self.A = np.log(self.A) - np.log(np.sum(self.A, axis=1, keepdims=True))
        self.B[self.B == 0.] = self.epsilon
        self.B = self.B / np.sum(self.B, axis=1, keepdims=True)
        #         self.B = np.log(self.B) - np.log(np.sum(self.B, axis=1, keepdims=True))
        self.Pi[self.Pi == 0.] = self.epsilon
        self.Pi = self.Pi / np.sum(self.Pi)
        #         self.Pi = np.log(self.Pi) - np.log(np.sum(self.Pi))
        np.savetxt('./output/Pi.csv', self.Pi, delimiter=',')
        np.savetxt('./output/A.csv', self.A, delimiter=',')
        np.savetxt('./output/B.csv', self.B, delimiter=',')
        print('训练完毕！')

    def viterbi(self, obs):
        #         pi = self.pi
        #         A = self.A
        #         B = self.B
        Pi = np.log(self.Pi)
        A = np.log(self.A)
        B = np.log(self.B)
        T = len(obs)
        delta = np.zeros((T, self.n_tag))  # 动态规划表格1，存储最大概率对数
        psi = np.zeros((T, self.n_tag))  # 动态规划表格2，存储路径信息
        UNK_ID = len(self.word2id)  # 定义未知词的ID
        word_id = self.word2id.get(obs[0], UNK_ID)
        random_pi = np.ones(self.n_tag) / self.n_tag
        if word_id == UNK_ID:
            delta[0, :] = Pi + random_pi
        else:
            delta[0, :] = Pi + B[:, word_id]
        psi[0, :] = 0
        for t in range(1, T):
            k = self.word2id.get(obs[t], UNK_ID)
            if k == UNK_ID:
                temp = delta[t - 1].reshape(self.n_tag,
                                            -1) + A  # 这里运用到了矩阵的广播算法
                delta[t] = np.max(temp, axis=0)
                delta[t] = delta[t, :] + random_pi
                psi[t] = np.argmax(temp, axis=0)
            else:
                temp = delta[t - 1].reshape(self.n_tag, -1) + A
                delta[t] = np.max(temp, axis=0)
                delta[t] = delta[t, :] + B[:, k]
                psi[t] = np.argmax(temp, axis=0)
        path = np.zeros(T, dtype=int)
        path[T - 1] = np.argmax(delta[T - 1, :])
        for i in range(T - 2, -1, -1):  # 回溯
            path[i] = int(psi[i + 1][int(path[i + 1])])
        return path

    def predict(self, word_list):
        all_sequences_tags = []  # 存放所有序列的标签
        print("开始对验证集分析")
        for Obs in tqdm(word_list):
            T = len(Obs)
            path = self.viterbi(Obs)
            sequence_tags = []  # 存放当前序列的标签

            for i in range(T):
                sequence_tags.append(self.id2tag[path[i]])

            all_sequences_tags.append(sequence_tags)  # 将当前序列的标签添加到整体列表中
        print("对验证集分析完毕")
        return all_sequences_tags

    # def predict(self, word_list, filename: str):
    #     with open(filename, 'w', encoding='UTF-8') as file:
    #         print("开始对验证集分析")
    #         for Obs in tqdm(word_list):
    #             T = len(Obs)
    #             path = self.viterbi(Obs)
    #             output_str = ''
    #             for i in range(T):
    #                 output_str += Obs[i] + ' ' + self.id2tag[path[i]] + '\n'
    #             output_str += '\n'
    #             file.write(output_str)
    #         print("对验证集分析完毕")
