# encoding=utf-8
# author： s0mE
# subject： 人名以及关系提取
# date： 2019-06-26

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import jieba
import jieba.posseg as pseg

from pyhanlp import *

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


name_dict =[] # ["罗辑","程心","汪淼","叶文洁","史强","维德","云天明","希恩斯","雷迪亚兹","丁仪","泰勒","章北海","关一帆","文洁","北海","天明","一帆","伟思","文斯","卫宁","始皇","心说","文王","玉菲","志成","西里","晓明","哲泰","庄颜","墨子","杨晋文","晋文","慈欣","沐霖","张援朝","援朝","艾AA","AA"]

name_dict =[] #["林黛玉","薛宝钗","贾元春","贾迎春","贾探春","贾惜春","李纨","妙玉","史湘云","王熙凤","贾巧姐","秦可卿","晴雯","麝月","袭人","鸳鸯","雪雁","紫鹃","碧痕","平儿","香菱","金钏","司棋","抱琴","赖大","焦大","王善保","周瑞","林之孝","乌进孝","包勇","吴贵","吴新登","邓好时","王柱儿","余信","庆儿","昭儿","兴儿","隆儿","坠儿","喜儿","寿儿","丰儿","住儿","小舍儿","李十儿","玉柱儿","贾敬","贾赦","贾政","贾宝玉","贾琏","贾珍","贾环","贾蓉","贾兰","贾芸","贾蔷","贾芹","琪官","芳官","藕官","蕊官","药官","玉官","宝官","龄官","茄官","艾官","豆官","葵官","妙玉","智能","智通","智善","圆信","大色空","净虚","彩屏","彩儿","彩凤","彩霞","彩鸾","彩明","彩云","贾元春","贾迎春","贾探春","贾惜春","薛蟠","薛蝌","薛宝钗","薛宝琴","王夫人","王熙凤","王子腾","王仁","尤老娘","尤氏","尤二姐","尤三姐","贾蓉","贾兰","贾芸","贾芹","贾珍","贾琏","贾环","贾瑞","贾敬","贾赦","贾政","贾敏","贾代儒","贾代化","贾代修","贾代善","晴雯","金钏","鸳鸯","司棋","詹光","单聘仁","程日兴","王作梅","石呆子","张华","冯渊","张金哥","茗烟","扫红","锄药","伴鹤","小鹊","小红","小蝉","小舍儿","刘姥姥","马道婆","宋嬷嬷","张妈妈","秦锺","蒋玉菡","柳湘莲","东平王","乌进孝","冷子兴","山子野","方椿","载权","夏秉忠","周太监","裘世安","抱琴","司棋","侍画","入画","珍珠","琥珀","玻璃","翡翠","史湘云","翠缕","笑儿","篆儿贾探春","侍画","翠墨","小蝉","贾宝玉","茗烟","袭人","晴雯","林黛玉","紫鹃","雪雁","春纤","贾惜春","入画","彩屏","彩儿","贾迎春","彩凤","彩云","彩霞"] 

class hanlp(object):
    def __init__(self):
        for n in name_dict:
            CustomDictionary.add(n,"nr")
        CustomDictionary.insert("凤霞", "nr")
        ## 数据集目录
        data_dir = "/home/dream/miniconda3/envs/py37/lib/python3.7/site-packages/pyhanlp/static/data/"
        data_path = data_dir + "model/perceptron/large/cws.bin"
        
        ## 构造人名分析器
        # 常规识别
        # self.analyzer = HanLP.newSegment().enableNameRecognize(True)
        self.analyzer = None
        
        
        #感知机识别
        PerceptronLexicalAnalyzer = JClass(
            "com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer")
        self.analyzer = PerceptronLexicalAnalyzer(data_path,
                                                  HanLP.Config.PerceptronPOSModelPath,
                                                  HanLP.Config.PerceptronNERModelPath)
        
        
        # # # crf识别
        # self.analyzer = JClass(
        #     "com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer")()
        

    def analyze(self,words):
        res = []
        
        #感知机和CRF的识别
        terms = str(self.analyzer.analyze(words)).split(" ")
        for term in terms:
            if "/" in term:
                flag = term.split("/")[-1]
                word = term[:len(term)-len(flag)-1]
                res.append((word,flag)) 
        return res
    def cut(self, words):
        res = []
        if self.analyzer is None:
            terms = HanLP.segment(words)
        else:
            terms = self.analyzer.seg(words)
        for term in terms:
            res.append( (str(term.word),str(term.nature)) )

        return res

def count_names(fp,model):
    """
    统计文本中的所有名字，返回统计矩阵
    """
    #逐行提取名字
    name_set = set()
    cut_result = []         
    with open(fp, "r") as f:
        lines = f.readlines()
        for ith, line in enumerate(lines):
            if ith % 200 == 0:  # 显示处理进度
                print("Processing:{:.5f}".format(ith*1.0/len(lines)))
            #每一行做预处理
            line = line.strip().replace(" ","")

            words = model.cut(line)
            line_dict = {}

            for word, flag in words:
                # if word == "。家珍":
                #     print(word,flag,"|||",line)
                
                if flag == "nr" or flag == "nrf":# or flag == "j":
                    # 如果 word 是人名，加入人名的统计中
                    line_dict[word] = line_dict.get(word, 0) + 1
                    name_set.add(word)
            if len(line_dict) != 0:
                cut_result.append(line_dict)

    # 名字关系矩阵计算
    names = list(name_set)  # 所有名字的列表
    name_arr = np.zeros((len(names), len(cut_result)),
                        dtype=np.int32)  # 储存统计结果的数组
    for n, n_dict in enumerate(cut_result):
            for k, v in n_dict.items():
                i = names.index(k)
                name_arr[i, n] += v
    # 计算人名的关系矩阵
    names = np.array(names)
    rel = np.zeros((len(names), len(names)), dtype=np.int32)
    for i in range(len(names)):
        rel[i, :] = np.sum(name_arr[:, name_arr[i, :] > 0], axis=1)

    ########至此，已经初步完成了文章的人物关系统计##############
    ############ 不过这里仍然有很多问题   ###################
    #### 例如明显的错误名字，以及同一人物不同的别称需要进一步处理 ###
    ################需要后续的处理 #######################
    print("==============Processing end!============")
    return rel, names


def filter_names(rel, names, trans={}, err=[], threshold=0):
    """对结果进行精细的调整与过滤

    处理顺序: 转换 ==> 去错 ==> 过滤 ==> 排序

    Args:
        rel:关系矩阵 n x n
        names: 人名向量矩阵 n
        trans: 别称转换字典 将别称转换为统一名字
        err: 错误名称矩阵 要删除的错误名称列表
        threshold: 词频阈值 词频低于此阈值的名字会被过滤，等于0（default）时使用词频均值自动过滤，等于-1不过滤
    
    Returns:
        rel_filter
        names_filter
        过滤好的人名矩阵和名称矩阵
    """

    # 名字的转换与计数的合并
    if len(trans) != 0:
        name_new = list(set(names) - set(trans.keys()))  # 转换后的名字
        indexes = [list(names).index(n) for n in name_new]
        for i, name in enumerate(names):
            if name in trans.keys():
                new_i = list(names).index(trans[names[i]])
                rel[new_i, :] += rel[i, :]
                rel[:, new_i] += rel[:, i]
        names = np.array(name_new)
        rel = rel[indexes, :][:, indexes]

    # 去错
    if len(err) != 0:
        name_new = list(set(names)-set(err))  # 去错后的名字列表
        indexes = [list(names).index(n) for n in name_new]
        names = np.array(name_new)
        rel = rel[indexes, :][:, indexes]

    # 过滤掉低频的名字
    if threshold != -1:
        if threshold == 0:
            rel_threshold = max(rel.diagonal().mean(), threshold)
        else:
            rel_threshold = threshold
        print("threshold:{:.3f}".format(rel_threshold))
        rel_filter = np.diag(rel) > rel_threshold
        names = names[rel_filter]
        rel = rel[rel_filter, :][:, rel_filter]
    

    # 人名排序
    indexes = np.argsort(np.diag(rel))[::-1]  # 从大到小
    names = names[indexes]
    rel = rel[indexes, :][:, indexes]

    return rel, names


def plot_rel(relations, names):

    # 过滤掉孤立的名字
    relations =relations.astype(np.float) + 0.1

    # 画图
    G = nx.Graph()
    sizes = np.diag(relations)
    G.add_nodes_from(names)
    weights = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            G.add_edge(names[i], names[j], weight=relations[i, j])
            weights.append(relations[i, j])

    weights = np.array(weights)
    weights = weights*5.0/np.max(weights)
    nx.draw(G, with_labels=True, node_size=sizes, width=weights)
    plt.show()


if __name__ == "__main__":
    jieba.load_userdict("user_dict.txt")
    fp = "huozhe.txt" #"白夜行.txt"  #"santi.txt"
    model = hanlp()
    rels, ns = count_names(fp, model)
    print(rels)

    trans_dict = {}
    err_list = []

    threshold = 2

    relations, names = filter_names(rels, ns, trans=trans_dict, err=err_list, threshold=threshold)
    print(names,np.diag(relations))
    plt.subplot(111)
    plot_rel(relations,names)
    plt.show()

   
