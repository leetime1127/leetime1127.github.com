基于Transformer的NLP智能对话机器人实战课程
One Architecture， One Course，One World

本课程以Transformer架构为基石、萃取NLP中最具有使用价值的内容、围绕手动实现工业级智能业务对话机器人所需要的全生命周期知识点展开，学习完成后不仅能够从算法、源码、实战等方面融汇贯通NLP领域NLU、NLI、NLG等所有核心环节，同时会具备独自开发业界领先智能业务对话机器人的知识体系、工具方法、及参考源码，成为具备NLP硬实力的业界Top 1%人才。
课程特色：
101章围绕Transformer而诞生的NLP实用课程
5137个围绕Transformers的NLP细分知识点、
大小近1200个代码案例落地所有课程内容、
10000+行纯手工实现工业级智能业务对话机器人在具体架构场景和项目案例中习得AI相关数学知识
NLP大赛全生命周期讲解并包含比赛的完整代码实现直
接加Gavin大咖微信咨询

第1章: 贝叶斯理论下的Transformer揭秘
1，基于Bayesian Theory，融Hard Attention、Soft Attention、Self-Attention、Multi-head Attention于一身的Transformer架构
2，为什么说抛弃了传统模型（例如RNN、 LSTM、CNN等）的Transformer拉开了非序列化模型时代的序幕？
3，为什么说Transformer是预训练领域底层通用引擎？
4，Transformer的Input-Encoder-Decoder-Output模型组建逐一剖析
5，Transformer中Encoder-Decoder模型进行Training时候处理Data的全生命周期七大步骤揭秘
6，Transformer中Encoder-Decoder模型进行Inference时候处理Data的全生命周期六大步骤详解
7，Teacher Forcing数学原理及在Transformer中的应用
8，穷根溯源：为何Scaled Dot-Product Attention是有效的？
9，透视Scaled Dot-Product Attention数据流全生命周期
10，穷根溯源：Queries、Keys、Values背后的Trainable矩阵揭秘
11，当Transformer架构遇到Bayesian理论：Multi-head attention
12，End-to-end Multi-head attention的三种不同实现方式分析	
13，透视Multi-head attention全生命周期数据流
14，Transformer的Feed-Forward Networks的两种实现方式：Linear Transformations和Convolutions
15，Embeddings和Softmax参数共享剖析
16，Positional Encoding及Positional Embedding解析
17，Sequence Masking和Padding Masking解析
18，Normal distribution、Layer Normalization和Batch Normalization解析
19，Transformer的Optimization Algorithms数学原理、运行流程和最佳实践
20，Learning rate剖析及最佳实践
21，从Bayesian视角剖析Transformer中的Dropout及最佳实践
22，Label Smoothing数学原理和工程实践解析
23，Transformer背后的驱动力探讨

第2章: 通过30+个细分模块完整实现Transformer论文源码及项目调试
1，Transformer源码训练及预测整体效果展示
2，模型训练model_training.py代码完整实现
3，数据预处理data_preprocess.py代码完整实现
4，Input端Embeddings源码完整实现
5，Attention机制attention.py代码完整实现
6，Multi-head Attention机制multi_head_attention.py代码完整实现
7，Position-wise Feed-forward源码完整实现
8，Masking 在Encoder和Decoder端的源码完整实现0
9，SublayerConnection源码完整实现
10，Encoder Layer源码完整实现
11，LayerNormalization源码完整实现
12，DecoderLayer源码完整实现
13，Encoder Stack源码完整实现
14，Decoder Stack源码完整实现
15，由Memory链接起来的EncoderDecoder Module源码完整实现
16，Batch操作完整源码实现
16，Optimization源码完整实现
17，Loss计算数学原理及完整源码实现
18，Output端Generator源码完整实现
19，Transformer模型初始化源码及内幕揭秘
20， Label Smoothing源码完整实现
21，Training源码完整实现
22，Greedy Decoding源码及内幕解析
23，Tokenizer源码及调试
24，Multi-GPU训练完整源码
27，使用自己实现的Transformer完成分类任务及调试
28，Transformer翻译任务代码完整实现及调试
29，BPE解析及源码实现
30，Shared Embeddings解析及源码实现
31，Beam Search解析及源码实现
32，可视化Attention源码实现及剖析

第3章: 细说Language Model内幕及Transformer XL源码实现
1，人工智能中最重要的公式之一MLE数学本质剖析及代码实战
2，Language Model的数学原理、Chain Rule剖析及Sparsity问题
3，Markov Assumption：first order、second order、third order剖析
4，Language Model：unigram及其问题剖析、bigram及依赖顺序、n-gram
5，使用Unigram训练一个Language Model剖析及实践
6，使用Bigram训练一个Language Model剖析及实践
7，使用N-gram训练一个Language Model剖析及实践
8，拼写纠错案例实战：基于简化后的Naive Bayes的纠错算法详解及源码实现
9，使用基于Average Log Likelihood的PPL(Perplexity)来评估Language Model
10，Laplace Smoothing剖析及基于PPL挑选最优化K的具体方法分析
11，Interpolation Smoothing实现解析：加权平均不同的N-gram概率
12，Good-Turning Smoothing算法解析
13，Vallina Transformer language model处理长文本架构解析
14， Vallina Transformer Training Losses：Multiple Postions Loss、Intermediate Layer Losses、Multiple Targets Losses
15，Vallina Transformer的三大核心问题：Segment上下文断裂、位置难以区分、预测效率低下
16，Transformer XL：Attentive Language Models Beyond a Fixed-Length Context
17，Segment-level Recurrence with State Reuse数学原理及实现分析
18，Relative Positional Encoding算法解析
19，Transformer XL 中降低矩阵运算复杂度的Trick解析
20，缓存机制在语言模型中的使用思考
21，Transformer XL之数据预处理完整源码实现及调试
22，Transformer XL之MemoryTransformerLM完整源码实现及调试
23，Transformer XL之PartialLearnableMultiHeadAttention源码实现及调试
24，Transformer XL之PartialLearnableDecoderLayer源码实现及调试
25，Transformer XL之AdaptiveEmbedding源码实现及调试
26，Transformer XL之相对位置编码PositionalEncoding源码实现及调试
27，Transformer XL之Adaptive Softmax解析及源码完整实现
28，Transformer XL之Training完整源码实现及调试
29，Transformer XL之Memory更新、读取、维护揭秘
30，Transformer XL之Unit单元测试
31，Transformer XL案例调试及可视化
![webwxgetmsgimg (4)](https://user-images.githubusercontent.com/30858035/141042661-3eca3df4-b961-4b9c-a28b-2beeffb234e7.jpg)
