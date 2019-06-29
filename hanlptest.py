from pyhanlp import *


txt1 = "所以，外太空宇航中的负面心理因素大多是以外部环境的超开放性为基础的，而在这种环境下，关一帆竟然产生了幽闭恐惧，这在韦斯特丰富的专业经历中十分罕见。但眼前还有一件更奇怪的事：韦斯特明显看出，关一帆进入广场后，暴露于广阔太空并没有使他产生舒适的解脱感，他身上那种因幽闭产生的躁动不安似乎一点都没有减轻。这也许证明了他说过的话，他的幽闭恐惧可能真的与那狭窄的观测站无关，这使得韦斯特对他产生了更大的兴趣。"

txt2 = "程心和关一帆再次拥抱在一起，他们都为艾AA和云天明流下了欣慰的泪水，幸福地感受着那两个人在十八万个世纪前的幸福，在这种幸福中，他们绝望的心灵变得无比宁静了。在这仅有三个人的孤寂世界中，即使短暂的分别也是一件让人激动的事，AA与程心和关一帆拥抱道别，祝他们平安。在登上穿梭机前，程心回头看，AA站在如水的星光中向他们挥手，大片的蓝草从她周围涌过，寒风吹起她的短发，也在移动的草地上激起道道波纹。"

txt3 = "桐原洋介曾在二战时入伍。笹垣以为他是谈这件事，文代却摇摇头。“是国外的战争。桐原先生说，这次石油一定会再涨。”"

txt4 = "香港特别行政区的张朝阳说商品和服务是三原县鲁桥食品厂的主营业务"

txt5 = "“宇宙规律。”程心说。"  # 这种句式的识别貌似存在问题
txt6 = "“元首，是这样——”科学执政官连忙解释道，“我们选择质子而不是中子进行二维展开，目的就是为了避免这种危险。万一零维展开真的出现，质子带有的电荷也会转移到展开后形成的黑洞中，我们就能用电磁力捕捉和控制住它。”"

txt7 = "这时我女儿凤霞推门进来，又摇摇晃晃地把门关上。凤霞尖声细气地对我说："

class hanlp(object):
    def __init__(self):
        # CustomDictionary.add("程心","nr 1800")
        # CustomDictionary.add("AA", "nr 1800")
        ## 数据集目录
        data_path = "/home/dream/miniconda3/envs/py37/lib/python3.7/site-packages/pyhanlp/static/data/model/perceptron/large/cws.bin"

        ## 构造人名分析器
        # 常规识别
        # self.model = HanLP.newSegment().enableJapaneseNameRecognize(True)

        #感知机识别
        # PerceptronLexicalAnalyzer = JClass(
        #     'com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer')
        # self.analyzer = PerceptronLexicalAnalyzer(data_path,
        #                                           HanLP.Config.PerceptronPOSModelPath,
        #                                           HanLP.Config.PerceptronNERModelPath)

        # crf识别
        self.analyzer = JClass(
            'com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer')()

    def cut(self, words):
        res = []

        #感知机和CRF的识别
        terms = str(self.analyzer.analyze(words)).split(" ")

        for term in terms:
            if "/" in term:
                flag = term.split("/")[-1]
                word = term[:len(term)-len(flag)-1]
                res.append((word, flag))
        return res


txt = txt3
# 默认分词
print(HanLP.segment(txt))  

#关键词提取
print(HanLP.extractKeyword(txt, 5))

print("="*50)
# 日语分词
segmenter = HanLP.newSegment().enableJapaneseNameRecognize(True)#
print(segmenter.seg(txt))

print("="*50)
# 中文名分词
segmenter = HanLP.newSegment().enableNameRecognize(True)
print(segmenter.seg(txt))

#感知机词法分析器
print("="*50)
#################
## 同时进行中文分词、词性标注与命名实体识别3个任务的子系统称为“词法分析器”。
data_path = "/home/dream/miniconda3/envs/py37/lib/python3.7/site-packages/pyhanlp/static/data/model/perceptron/large/cws.bin"
PerceptronLexicalAnalyzer = JClass('com.hankcs.hanlp.model.perceptron.PerceptronLexicalAnalyzer')
analyzer = PerceptronLexicalAnalyzer(data_path,
                                    HanLP.Config.PerceptronPOSModelPath,
                                     HanLP.Config.PerceptronNERModelPath)
print(analyzer.seg(txt))
print(analyzer.analyze(txt))


print("="*50)
analyzer = PerceptronLexicalAnalyzer()
# CustomDictionary.add("凤霞","nr")
percep_seg = analyzer.seg(txt)
print(percep_seg)
# print(analyzer.analyze(txt))

# CRF词法分析器
print("="*50)
analyzer = JClass('com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer')(
).enableJapaneseNameRecognize(True)
crf_seg = analyzer.seg(txt)
print(crf_seg)
# print(analyzer.analyze(txt))


# 结果合并与优化
print("="*50)
for i in range(2):
    crf_seg = analyzer.seg(txt)
    for item in crf_seg:
        word , flag = str(item.word),str(item.nature)
        print(word,flag,end="\t")
    print("="*50)
    # CustomDictionary.add("艾AA", "nr")
    CustomDictionary.add("AA", "nr")
    CustomDictionary.add("天明", "nr")
    # CustomDictionary.add("关一帆", "nr")
    CustomDictionary.add("一帆", "nr")

