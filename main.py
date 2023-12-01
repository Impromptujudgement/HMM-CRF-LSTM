from data import build_corpus
from evaluating import *
from models.LSTM_CRF import BiLSTM_CRF

if __name__ == "__main__":
    # 英文
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    train_set, word2id, tag2id = build_corpus(data_dir="./English/train.txt")
    validation_set = build_corpus(data_dir="./English/validation.txt", make_vocab=False)
    # hmm_train(train_set, word2id, tag2id, model_save_path="./ckpts/HMM_English.pkl")
    hmm_eval(validation_set, model_save_path="./ckpts/HMM_English.pkl", output_path="./output/HMM_English_model.txt",
             language="English")
    # crf_train(train_set, language="English", model_save_path="./ckpts/CRF_English.pkl")
    crf_eval(validation_set, model_save_path="./ckpts/CRF_English.pkl", output_path="./output/CRF_English_model.txt",
             language="English")
    word2id, tag2id = extend_maps(word2id, tag2id)
    model = BiLSTM_CRF(len(word2id), tag2id, word2id, EMBEDDING_DIM,
                       HIDDEN_DIM).to(device)
    lstm_crf_train(model, train_set, validation_set, word2id, tag2id, epochs=100,
                   model_save_path='./ckpts/Best_Model_English.pth', loss_picture_path="./imgs/English.png")
    lstm_crf_eval(model, model_save_path='./ckpts/Best_Model_English.pth', val_set=validation_set,
                  output_path="./output/BiLSTM_CRF_English_model.txt", language='English')

    # # 中文
    train_set, word2id, tag2id = build_corpus(data_dir="./Chinese/train.txt")
    validation_set = build_corpus(data_dir="./Chinese/validation.txt", make_vocab=False)
    hmm_train(train_set, word2id, tag2id, model_save_path="./ckpts/HMM_Chinese.pkl")
    hmm_eval(validation_set, model_save_path="./ckpts/HMM_Chinese.pkl", output_path="./output/HMM_Chinese_model.txt",
             language="Chinese")
    # crf_train(train_set, language="Chinese", model_save_path="./ckpts/CRF_Chinese.pkl")
    crf_eval(validation_set, model_save_path="./ckpts/CRF_Chinese.pkl", output_path="./output/CRF_Chinese_model.txt",
             language="Chinese")
    word2id, tag2id = extend_maps(word2id, tag2id)
    model = BiLSTM_CRF(len(word2id), tag2id, word2id, EMBEDDING_DIM,
                       HIDDEN_DIM).to(device)
    # model = lstm_crf_train(train_set, validation_set, word2id, tag2id, epochs=10,
    #                        model_save_path='./ckpts/Best_Model_Chinese.pth')
    lstm_crf_eval(model, model_save_path='./ckpts/Best_Model_Chinese.pth', val_set=validation_set,
                  output_path="./output/BiLSTM_CRF_Chinese_model.txt", language='Chinese')
