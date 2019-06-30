# encoding=utf-8
# author： s0mE
# subject： 人名以及关系提取
# date： 2019-06-26
import argparse
import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pyhanlp import *

from tqdm import tqdm

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


class hanlp(object):
    def __init__(self, analyzer = "Perceptron", custom_dict = True ):
        ## 数据集目录
        data_path = "/home/dream/miniconda3/envs/py37/lib/python3.7/site-packages/pyhanlp/static/data/model/perceptron/large/cws.bin"
        
        ## 构造人名分析器
        # 常规识别
        # self.analyzer = HanLP.newSegment().enableNameRecognize(True)

        # # crf识别
        self.CRFLAnalyzer = JClass("com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer")()

        #感知机识别
        _PLAnalyzer = JClass("com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer")
        self.PLAnalyzer = _PLAnalyzer(
            data_path, HanLP.Config.PerceptronPOSModelPath, HanLP.Config.PerceptronNERModelPath)
        
        self.analyzer = self.PLAnalyzer
        if analyzer=="Perceptron":
            self.analyzer = self.PLAnalyzer.enableCustomDictionary(custom_dict)
        elif analyzer=="CRF":
            self.analyzer = self.CRFLAnalyzer.enableCustomDictionary(custom_dict)
        
    def cut(self, words):
        res = []
        if self.analyzer is None:
            terms = HanLP.segment(words)
        else:
            terms = self.analyzer.seg(words)
        for term in terms:
            res.append( (str(term.word),str(term.nature)) )
        return res
    
    @classmethod
    def add(self,names_list):
        for n in names_list:
            if CustomDictionary.get(n) is None:
                CustomDictionary.add(n,"nr 1000 ")
            else:
                attr = "nr 1000 " + str(CustomDictionary.get(n))
                # attr = "nr 1000 "
                CustomDictionary.insert(n,attr)

    @classmethod
    def insert(self, names_list):
        for n in names_list:
            CustomDictionary.insert(n, "nr 1")
            
def count_names(fp,model):
    """
    统计文本中的所有名字，返回统计矩阵
    """
    #逐行提取名字
    name_set = set() # 所有名字的集合
    
    
    nr_nrf_dict = {"nr":{},"nrf":{}}

    cut_result = []         
    with open(fp, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Analyzing"):
            #每一行做预处理
            line = line.strip().replace(" ","")

            words = model.cut(line)
            line_dict = {}

            for word, flag in words:
                # if word == "张":
                #     print(word,flag,"|||",line)
                
                if flag == "nr" or flag == "nrf":# or flag == "j":
                    # 如果 word 是人名，加入人名的统计中
                    line_dict[word] = line_dict.get(word, 0) + 1
                    name_set.add(word)

                    # 分中文名和英文名统计名称
                    nr_nrf_dict[flag][word] = nr_nrf_dict[flag].get(word, 0) + 1
                    
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
    return rel, names, nr_nrf_dict


def filter_nr(nr_nrf_dict, threshold = -1,first=False):
    """
    自动生成可信名称列表 和 名字转换字典
    """
    nr_dict = nr_nrf_dict["nr"]
    nrf_dict = nr_nrf_dict["nrf"]
    
    first_threshold = 5
    if threshold == -1:
        threshold = np.mean( list(nr_dict.values())+list(nrf_dict.values()))
        first_threshold = max(np.sqrt(len(nr_dict)+len(nrf_dict)),5*threshold)
    print("auto_dict threshold:{:.3f}".format(threshold))
    names = []
    trans_dict = {}
    last_names = []
    last_repeat = []

    first_names = []
    first_repeat = []
    for name,value in sorted(nr_dict.items(), key=lambda d: d[1], reverse=True):
        if value > threshold:
            if len(name) == 1 and value < first_threshold:
                continue
            names.append(name)
            last_name = name[1:]
            # 获取三字姓名的名字的部分，如果存在重复的删除
            if len(name)==3 and not last_name in last_repeat:
                if last_name in last_names:
                    last_names.remove(last_name)
                    trans_dict.pop(last_name)
                    last_repeat.append(last_name)
                else:
                    trans_dict[last_name] = name
                    last_names.append(last_name)
            
            # 获取姓名的姓的部分
            first_name = name[:1]
            if first and len(name)==3 and not first_name in first_repeat:
                if first_name in first_names:
                    first_names.remove(first_name)
                    trans_dict.pop(first_name)
                    first_repeat.append(first_name)
                else:
                    trans_dict[first_name] = name
                    first_names.append(first_name)
        
    names = last_names + names
    # print(names)
    for name,value in nrf_dict.items():
        if value > threshold:
            names.append(name)
    return names,trans_dict

def filter_names(rel, names, trans={}, err=[], threshold= -1):
    """对结果进行精细的调整与过滤

    处理顺序: 转换 ==> 去错 ==> 过滤 ==> 排序

    Args:
        rel:关系矩阵 n x n
        names: 人名向量矩阵 n
        trans: 别称转换字典 将别称转换为统一名字
        err: 错误名称矩阵 要删除的错误名称列表
        threshold: 词频阈值 词频低于此阈值的名字会被过滤，等于-1（default）时使用词频均值自动过滤
    
    Returns:
        rel_filter
        names_filter
        过滤好的人名矩阵和名称矩阵
    """
    
    rel = np.copy(rel)
    names = np.copy(names)

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
    if threshold != 0:
        if threshold == -1:
            rel_threshold = max(rel.diagonal().mean(), threshold)
        else:
            rel_threshold = threshold
        print("out threshold:{:.3f}".format(rel_threshold))
        rel_filter = np.diag(rel) > rel_threshold
        names = names[rel_filter]
        rel = rel[rel_filter, :][:, rel_filter]
    

    # 人名排序
    indexes = np.argsort(np.diag(rel))[::-1]  # 从大到小
    names = names[indexes]
    rel = rel[indexes, :][:, indexes]

    return rel, names


def plot_rel(relations, names,balanced=True):

    # 平衡名字关系
    if balanced == True:
        relations =(relations.T+relations)/2
    

    # 画图
    G = nx.Graph()

    # 将每个名字，和名字出现的次数加入图
    nums = np.diag(relations)
    for i,name in enumerate(names):
        G.add_node(name, num = nums[i])

    # 将关系加入图
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            if relations[i, j] != 0:
                G.add_edge(names[i], names[j], weight=relations[i, j])

    # 判断是否联通并切分子图
    max_weight = 0.0
    #### for c in sorted(nx.connected_components(G), key=len, reverse=True):
    #　画出主要子图
    main_c = max(nx.connected_components(G), key=len)
    sub_G = G.subgraph(main_c)
    sub_nums = np.array([n[1] for n in sub_G.nodes(data="num")])
    sub_weight = np.array([e[2] for e in sub_G.edges(data="weight")])
    if len(sub_weight) != 0:  # 权重值为 0 则不需要归一化
        max_weight = max(np.max(sub_weight), max_weight)
        sub_weight = sub_weight*4.5/max_weight

    # nx.draw(sub_G, with_labels=True, node_size=sub_nums, width=sub_weight)
    # plt.show()
    nx.draw_spring(sub_G, with_labels=True, node_size=sub_nums, width=sub_weight)
    plt.show()
    nx.draw_circular(sub_G, with_labels=True, node_size=sub_nums, width=sub_weight)
    plt.show()
    nx.draw_random(sub_G, with_labels=True, node_size=sub_nums, width=sub_weight)
    plt.show()
    nx.draw_spectral(sub_G, with_labels=True, node_size=sub_nums, width=sub_weight)
    plt.show()
    nx.draw_kamada_kawai(sub_G, with_labels=True, node_size=sub_nums, width=sub_weight)
    # plt.show()
    # nx.draw_shell(sub_G, with_labels=True, node_size=sub_nums, width=sub_weight)
    # plt.show()
    #主要子图外其他的图
    other_c = set(G.nodes) - main_c

    info = "<<shown-points>>\n{}\n<<dropout-points>>\n{}".format(
        sub_G.nodes(data="num"), G.subgraph(other_c).nodes(data="num"))
    return info

def trans_list2dict(trans_list):
    """
    把别名转换列表转换为别名转换字典
    """
    trans_dict = {}
    for names in trans_list:
        for i,name in enumerate(names):
            if i==0:
                continue
            trans_dict[name] = names[0]
    return trans_dict



# ["罗辑","程心","汪淼","叶文洁","史强","维德","云天明","希恩斯","雷迪亚兹","丁仪","泰勒","章北海","关一帆","文洁","北海","天明","一帆","伟思","文斯","卫宁","始皇","心说","文王","玉菲","志成","西里","晓明","哲泰","庄颜","墨子","杨晋文","晋文","慈欣","沐霖","张援朝","援朝","艾AA","AA"]
# info = ["林黛玉","薛宝钗","贾元春","贾迎春","贾探春","贾惜春","李纨","妙玉","史湘云","王熙凤","贾巧姐","秦可卿","晴雯","麝月","袭人","鸳鸯","雪雁","紫鹃","碧痕","平儿","香菱","金钏","司棋","抱琴","赖大","焦大","王善保","周瑞","林之孝","乌进孝","包勇","吴贵","吴新登","邓好时","王柱儿","余信","庆儿","昭儿","兴儿","隆儿","坠儿","喜儿","寿儿","丰儿","住儿","小舍儿","李十儿","玉柱儿","贾敬","贾赦","贾政","贾宝玉","贾琏","贾珍","贾环","贾蓉","贾兰","贾芸","贾蔷","贾芹","琪官","芳官","藕官","蕊官","药官","玉官","宝官","龄官","茄官","艾官","豆官","葵官","妙玉","智能","智通","智善","圆信","大色空","净虚","彩屏","彩儿","彩凤","彩霞","彩鸾","彩明","彩云","贾元春","贾迎春","贾探春","贾惜春","薛蟠","薛蝌","薛宝钗","薛宝琴","王夫人","王熙凤","王子腾","王仁","尤老娘","尤氏","尤二姐","尤三姐","贾蓉","贾兰","贾芸","贾芹","贾珍","贾琏","贾环","贾瑞","贾敬","贾赦","贾政","贾敏","贾代儒","贾代化","贾代修","贾代善","晴雯","金钏","鸳鸯","司棋","詹光","单聘仁","程日兴","王作梅","石呆子","张华","冯渊","张金哥","茗烟","扫红","锄药","伴鹤","小鹊","小红","小蝉","小舍儿","刘姥姥","马道婆","宋嬷嬷","张妈妈","秦锺","蒋玉菡","柳湘莲","东平王","乌进孝","冷子兴","山子野","方椿","载权","夏秉忠","周太监","裘世安","抱琴","司棋","侍画","入画","珍珠","琥珀","玻璃","翡翠","史湘云","翠缕","笑儿","篆儿贾探春","侍画","翠墨","小蝉","贾宝玉","茗烟","袭人","晴雯","林黛玉","紫鹃","雪雁","春纤","贾惜春","入画","彩屏","彩儿","贾迎春","彩凤","彩云","彩霞"] 
# hanlp.add(info)
parser = argparse.ArgumentParser(description="指定书的名字")

parser.add_argument("--book", default="weicheng", type=str,
                    help="书的名字，不带后缀")
parser.add_argument("--debug",default=False,type=bool,help="控制中间结果的输出。默认关闭")

if __name__ == "__main__":

    # a = str(CustomDictionary.get("鸿渐"))
    # print(a=="nz 3 ")
    #################################################
    # ############################################# 
    # ############# 手动调整模型 ####################
    # 前期添加的字典
    name_dict = []
    
    # 后期效果优化
    trans_list = [] 
    # 转换列表，格式如下
    # [[name1,name1_,...],[name2,name2_,...],... ]
    # 列表内的每一个列表代表一个人物的一组别名，所有别名会转换为第一个名字
    
    trans_dict = {}
    trans_dict.update(trans_list2dict(trans_list))
    
    err_list = []

    threshold = -1
    # ############################################
    # ############################################
    
    # 获取书名参数
    args = parser.parse_args()
    fp = "book/"+ args.book +".txt"
    assert os.path.exists(fp),"error!: no such book in "+ fp
    print("=====+++=== NER for book: "+fp+" ===+++=====",flush=True)
    ###################################33
    ###############################
    # 插入个性化字典
    # name_dict = []
    hanlp.add(name_dict)
    #################################
    #################################
    
    # 感知机分析器对文本进行分析
    model = hanlp(custom_dict=True)
    rels, ns, nr_nrf_dict = count_names(fp, model)
    if args.debug:
        f = np.diag(rels) >= 40
        print("="*50)
        print("<<粗提取结果>>\n名字总数: {} \n{}{}".format(len(ns),ns[f],np.diag(rels)[f]))
        print("="*50)

    ## 分别生成新的名称字典，以及转换字典
    # print(filter_nr(nr_nrf_dict))
    auto_name_list, auto_trans_dict = filter_nr(nr_nrf_dict,first=True)
    if args.debug:
        print("="*50)
        print("<<自动生成的名称列表和名称转换字典>>")
        print("名称列表:\n", auto_name_list)
        print("名称转换字典\n",auto_trans_dict)
        print("="*50)
    hanlp.add(auto_name_list)
    
          
    ############################################
    # 手动调整的转换字典
    auto_trans_dict.update(trans_dict)
    trans_dict = auto_trans_dict
    ###############################################
    
    
    ### 重新进行统计和计数
    model = hanlp(custom_dict=True)#,analyzer="CRF")
    rels,ns,_ = count_names(fp,model)
  
    #####根据手工调整以不同效果展示
    relations, names = filter_names(
            rels, ns, trans=trans_dict, err=err_list, threshold=threshold)
    # print(names, np.diag(relations))
    plt.subplot(111)
    info = plot_rel(relations,names)
    print("="*50)
    print("+++++++ 最终分析结果: +++++++")
    print(info)
    print("="*50)
    plt.show()

   
