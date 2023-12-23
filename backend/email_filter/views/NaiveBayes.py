import os

import numpy as np
import re
import random
from django.conf import settings

# 数据预处理操作（词的切分、词转化为小写）
def text_parse(input_str):
    word_list = re.split(r"\W+", input_str)
    return [word.lower() for word in word_list if len(word_list) > 2 and len(word) > 0]  # 字母都转换成小写，过滤掉长度为0的单词


# 获取数据
def read_data():
    doc_list = []  # 每句话的单词列表
    class_list = []  # 每句话是否是垃圾邮件，1表示是
    with open(os.path.join(settings.BASE_DIR, 'email_filter', 'views', 'data', 'SMS.txt'), "r", encoding="utf-8") as file:
        datas = file.read()
        datas = datas.split("\n")  # 按回车分开
        for data in datas:  # 读取每一行数据
            # label = ham 代表 正常邮件 ， label = spam 代表垃圾邮件
            label, text = data.split("\t")
            doc_list.append(text_parse(text))
            # 0：正常邮件，1：垃圾邮件
            class_list.append(0 if label == "ham" else 1)
    return doc_list, class_list


# 构建语料表，把所有单词整合在一个集合中
def create_vocabulary_list(doc_list):
    vocabulary_set = set([])
    for document in doc_list:
        vocabulary_set = vocabulary_set | set(document)
    return list(vocabulary_set)


# 将一篇邮件转化为 类似 One-Hot 的向量，长度和 vocabulary_list 一样，为 1 的位置代表该单词在该邮件中出现了
def set_of_word2vector(vocabulary_list, document):
    vec = [0 for _ in range(len(vocabulary_list))]  # 先生成一个和vocabulary_list长度一样的，全0的向量
    for word in document:
        index = vocabulary_list.index(word)  # 当前这句话的这个单词在vocabulary_list的第几个位置出现了，在vec的对应位置标记为1
        if index >= 0:
            vec[index] = 1
    return vec


# 用朴素贝叶斯算法进行计算
def naive_bayes(train_matrix, train_class):
    # 训练样本个数
    train_data_size = len(train_class)
    # 语料库大小，总数据个数，即向量长度
    vocabulary_size = len(train_matrix[0])
    # 计算垃圾邮件的概率值
    p_spam = sum(train_class) / train_data_size
    # 初始化分子，做了一个平滑处理（拉普拉斯平滑）
    p_ham_molecule = np.ones(vocabulary_size)
    p_spam_molecule = np.ones(vocabulary_size)

    # 初始化分母（通常初始化为类别个数，在垃圾邮件分类中，只有垃圾和正常两种邮件，所以类别数为2）
    p_ham_denominator = 2
    p_spam_denominator = 2
    # 循环计算分子和分母
    for i in range(train_data_size):
        if train_class[i] == 1:
            p_spam_molecule += train_matrix[i]
            p_spam_denominator += sum(train_matrix[i])
        else:
            p_ham_molecule += train_matrix[i]
            p_ham_denominator += sum(train_matrix[i])
    # 计算概率
    p_ham_vec = p_ham_molecule / p_ham_denominator
    p_spam_vec = p_spam_molecule / p_spam_denominator
    # 返回
    return p_ham_vec, p_spam_vec, p_spam


# 预测，返回预测的类别
def predict(vec, p_ham_vec, p_spam_vec, p_spam):
    # 由于计算出来的概率通常很接近0，所以我们通常取对数将它“放大”，这也基于我们在做贝叶斯的时候，不需要知道他们确切的概率，只需要比较他们概率大小即可
    p_spam = np.log(p_spam) + sum(vec * np.log(p_spam_vec))
    p_ham = np.log(1 - p_spam) + sum(vec * np.log(p_ham_vec))
    return 1 if p_spam >= p_ham else 0


def get_newdata_doclist(datas):
    new_data_doclist = []
    datas = datas.split('\n')
    for data in datas:
        new_data_doclist = new_data_doclist + text_parse(data)
    return new_data_doclist


def main(input_data):
    # 读取数据
    doc_list, class_list = read_data()
    print(f"A total of {len(class_list)} email data were read, including {sum(class_list)} spam")

    # 构建语料表
    vocabulary_list = create_vocabulary_list(doc_list)
    # 划分训练集和测试集
    test_ratio = 0.3
    train_index_set = [i for i in range(len(doc_list))]  # 训练集下标，（实际上是所有元素）
    test_index_set = []  # 测试集下标
    for _ in range(int(len(doc_list) * test_ratio)):
        index = random.randint(0, len(train_index_set) - 1)
        test_index_set.append(train_index_set[index])
        del train_index_set[index]  # 从训练集中剔除
    print(f"test_ratio: {test_ratio} , train_data_size: {len(train_index_set)} , test_data_size: {len(test_index_set)}")

    # 将邮件转化为向量
    train_matrix = []
    train_class = []
    for train_index in train_index_set:  # 遍历训练集下标
        train_matrix.append(set_of_word2vector(vocabulary_list, doc_list[train_index]))
        train_class.append(class_list[train_index])

    # 用朴素贝叶斯算法进行计算
    p_ham_vec, p_spam_vec, p_spam = naive_bayes(np.array(train_matrix), np.array(train_class))

    data_input = input_data

    data_input_doclist = get_newdata_doclist(data_input)

    vec = set_of_word2vector(vocabulary_list, data_input_doclist)
    predict_class = predict(vec, p_ham_vec, p_spam_vec, p_spam)

    print(predict_class)
    return predict_class

