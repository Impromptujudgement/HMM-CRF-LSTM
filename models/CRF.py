from sklearn_crfsuite import CRF
from .util import sent2features


class CRFModel:
    def __init__(self,
                 algorithm='lbfgs',     # CRF 模型训练时所采用的优化算法，默认为 'lbfgs'（Limited-memory Broyden–Fletcher–Goldfarb–Shanno）
                 c1=0.1,     # L1正则化系数
                 c2=0.1,     # L2正则化系数
                 max_iterations=100,
                 all_possible_transitions=True
                 ):
        self.model = CRF(algorithm=algorithm,
                         c1=c1,
                         c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)

    def train(self, train_set, language):
        sentences, tag_lists = train_set
        features = [sent2features(s, language) for s in sentences]
        self.model.fit(features, tag_lists)

    def predict(self, sentences, language):
        features = [sent2features(s, language) for s in sentences]
        print("开始对验证集分析")
        pred_tag_lists = self.model.predict(features)
        print("对验证集分析完毕")
        return pred_tag_lists
