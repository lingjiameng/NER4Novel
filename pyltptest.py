

from pyltp import Postagger
from pyltp import Segmentor
import os

LTP_DATA_DIR = "/media/Study/project/nlp/ltp_data_v3.4.0/"  # ltp模型目录的路径

txt1 = "所以，外太空宇航中的负面心理因素大多是以外部环境的超开放性为基础的，而在这种环境下，关一帆竟然产生了幽闭恐惧，这在韦斯特丰富的专业经历中十分罕见。但眼前还有一件更奇怪的事：韦斯特明显看出，关一帆进入广场后，暴露于广阔太空并没有使他产生舒适的解脱感，他身上那种因幽闭产生的躁动不安似乎一点都没有减轻。这也许证明了他说过的话，他的幽闭恐惧可能真的与那狭窄的观测站无关，这使得韦斯特对他产生了更大的兴趣。"

txt2 = "程心和关一帆再次拥抱在一起，他们都为艾AA和云天明流下了欣慰的泪水，幸福地感受着那两个人在十八万个世纪前的幸福，在这种幸福中，他们绝望的心灵变得无比宁静了。在这仅有三个人的孤寂世界中，即使短暂的分别也是一件让人激动的事，AA与程心和关一帆拥抱道别，祝他们平安。在登上穿梭机前，程心回头看，AA站在如水的星光中向他们挥手，大片的蓝草从她周围涌过，寒风吹起她的短发，也在移动的草地上激起道道波纹。"

txt3 = "桐原洋介曾在二战时入伍。笹垣以为他是谈这件事，文代却摇摇头。“是国外的战争。桐原先生说，这次石油一定会再涨。”"

txt4 = "香港特别行政区的张朝阳说商品和服务是三原县鲁桥食品厂的主营业务"

txt = txt1


# 分词模型路径，模型名称为`cws.model`
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  
# 词性标注模型路径，模型名称为`pos.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')




segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型

words = segmentor.segment(txt)  # 分词
print('\t'.join(words))


postagger = Postagger()  # 初始化实例
postagger.load(pos_model_path)  # 加载模型

# words = ['元芳', '你', '怎么', '看']  # 分词结果
postags = postagger.postag(words)  # 词性标注

print ('\t'.join(postags))


postagger.release()  # 释放模型
segmentor.release()  # 释放模型
