import torch.nn as nn
import os
from .util import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

START_TAG = "<start>"
STOP_TAG = "<end>"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, word2id, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.word2id = word2id
        self.id2tag = dict(zip(tag_to_ix.values(), tag_to_ix.keys()))
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size).to(device))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        self.hidden = self.init_hidden()

    def init_hidden(self):
        #   双向LSTM，所以要除以2
        return (torch.randn(2, 1, self.hidden_dim // 2).to(device),
                torch.randn(2, 1, self.hidden_dim // 2).to(device))

    #   前向算法计算观测序列的分数
    def _forward_alg(self, feats):
        init_alphas = torch.full([self.tagset_size], -10000.).to(device)
        init_alphas[self.tag_to_ix[START_TAG]] = 0.
        forward_var_list = [init_alphas]

        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            t_r1_k = torch.unsqueeze(feats[feat_index], 0).transpose(0, 1)
            aa = gamar_r_l + t_r1_k + self.transitions
            forward_var_list.append(torch.logsumexp(aa, dim=1))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var = torch.unsqueeze(terminal_var, 0)
        alpha = torch.logsumexp(terminal_var, dim=1)[0]
        return alpha

    #   将输入的句子经过词嵌入层和 LSTM 层，得到每个词对应的特征表示。
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    #    计算给定特征和标签序列的得分。
    def _score_sentence(self, feats, tags):
        score = torch.zeros(1).to(device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(device), tags])

        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    #    使用viterbi算法来进行解码，找出最优的标签序列。
    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tagset_size), -10000.).to(device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        forward_var_list = [init_vvars]

        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1]).to(device)
            gamar_r_l = torch.squeeze(gamar_r_l).to(device)
            next_tag_var = gamar_r_l + self.transitions
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0).to(device)
            forward_var_new = torch.unsqueeze(viterbivars_t, 0).to(device) + t_r1_k

            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())

        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]
        best_path = [best_tag_id]

        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    #     计算负对数似然，用于模型的训练。
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    #     前向传播过程，获取模型预测的标签序列及其得分。
    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
