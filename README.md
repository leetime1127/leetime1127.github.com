基于Transformer的NLP智能对话机器人实战课程

NLP on Transformers 101

One Architecture， One Course，One Universe

本课程以Transformer架构为基石、萃取NLP中最具有使用价值的内容、围绕手动实现工业级智能业务对话机器人所需要的全生命周期知识点展开，学习完成后不仅能够从算法、源码、实战等方面融汇贯通NLP领域NLU、NLI、NLG等所有核心环节，同时会具备独自开发业界领先智能业务对话机器人的知识体系、工具方法、及参考源码，成为具备NLP硬实力的业界Top 1%人才。

课程特色：
	101章围绕Transformer而诞生的NLP实用课程
	5137个围绕Transformers的NLP细分知识点
	大小近1200个代码案例落地所有课程内容
	10000+行纯手工实现工业级智能业务对话机器人
	在具体架构场景和项目案例中习得AI相关数学知识
	以贝叶斯深度学习下Attention机制为基石架构整个课程
	五大NLP大赛全生命周期讲解并包含比赛的完整代码实现


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

第4章: Autoregressive Language Models之GPT-1、2、3解析及GPT源码实现
	1，Task-aware的人工智能Language model + Pre-training + Fine-tuning时代
	2，Decoder-Only Stack数学原理及架构解析
	3，训练材料标注：neutral、contradiction、entailment、multi-label、QA等
	4，NLP(Natural Language Understanding)：Semantic similarity、document classification、textual entailment等
	5，大规模Unsupervised pre-training贝叶斯数学原理及架构剖析
	6，Task-specific Supervised fine-tuning的Softmax及Loss详解
	7，针对Classification、Entailment、Similarity、Mutiple Choice特定任务的Input数据预处理解析及矩阵纬度变化处理
	8，GPT2架构解析：Language Models for unsupervised multitask learners
	9，GPT 2把Layer Norm前置的数据原理剖析
	10，GPT 2 Self-Attention剖析
	11，GPT 2 Training数据流动全生命周期解析
	12，GPT 2 Inference数据流动全生命周期解析
	13，GPT 3 架构剖析：Language Models are Few-Shot Learners
	14，由GPT 3引发的NLP12大规律总结
	15，GPT数据预处理源码完整实现及调试
	16，GPT的BPE实现源码及调试
	17，GPT的TextEncoder源码实现及调试
	18，GPT的Attention完整源码实现及调试
	19，GPT的Layer Normalization完整源码实现及调试
	20，GPT的Feed Foward神经网络通过Convolutions源码实现
21，GPT的Block源码完整实现及调试
	22，GPT的TransformerModel源码完整实现及调试
23，GPT的输入LMHead源码完整实现及调试
	24，GPT的MultipleChoiceHead源码完整实现及调试
	25，GPT的语言模型及特定Task的DoubleHeadModel源码完整实现
	26，GPT的OpenAIAdam优化器源码及调试
27，GPT的LanguageModel loss源码及调试
	28，GPT的MultipleChoiceLoss源码及调试
	29，OpenAI GPT的Pretrained Model的加载使用
	30，GPT模型Task-specific训练完整源码及调试
	31，GPT进行Inference完整源码实现及代码调试
	
第5章: Autoencoding Language Models数学原理及模型架构解析
1，Auto-encoding Language Models通用数学原理详解
2，为何要放弃采用Feature-Based语言模型ELMo而使用Fine-tuning模型？
3，双向语言模型：both left-to-right and right-to-left不同实现及数学原理解析
4，深度双向语言模型背后的数学原理及物理机制
5，Unsupervised Fine-tuning训练模型架构及数学原理解析
6，Transfer Learning数学原理及工程实现详解
7，MLM(Masked Language Models)数学原理及工程架构解析
8，MLM问题解析及解决方案分析
9，Pre-training + Fine-tuning的BERT分层架构体系及组件解析
10，BERT的三层复合Embeddings解析
11，BERT不同模块的参数复杂度分析
12，BERT在进行Masking操作中采用10%随机选取词库的内容进行替换masked位置的内容的数学原理剖析
13，BERT在进行Masking操作中采用10%的内容维持不变的数学原理揭秘
14，BERT的Masking机制五大缺陷及其解决方案分析
15，BERT的Masking机制在Data Enchancement方面的妙用
16，BERT的Masking机制在处理智能对话系统中不规范用语甚至是错误语法及用词的妙用
17，BERT的NSP(Next Sentence Prediction)机制及其实现
18，BERT的NSP三大问题及解决方案剖析
19，BERT的CLS剖析及工程实现
20，BERT的CLS三个核心问题及解决方案
21，Knowledge Distillation for BERT数学原理贝叶斯及KL散度解析及案例实战
22，使用BERT进行Classification架构及案例实战
23，使用BERT进行NER(Named Entity Recognition)架构及案例实战
24，使用BERT实现文本Similarity任务的架构及案例实战
25，使用BERT实现Question-Answering任务的架构及案例实战
26，ALBERT模型架构解析
27，RoBERTa模型架构解析
28，SpanBERT模型架构解析
29，TinyBERT模型架构解析
30，Sentence-BERT模型架构解析
31，FiBERT模型架构解析
32，K-BERT模型架构解析
33，KG-BERT模型架构解析

	
第6章: BERT Pre-training模型源码完整实现、测试、调试及可视化分析
	1，词典Vocabulary库构建多层级源码实现及测试
	2，Dataset加载及数据处理源码完整实现及测试和调试
3，Next Sentence Prediction机制源码完整实现及测试
4，Masked Language Model机制中80%词汇Masking源码实现
	5，Masked Language Model机制中10%词汇随机替换和10%词汇保持不变源码实现
	6，Masked Language Model机制下的Output Label操作源码实现
	7，加入CLS、SEP 等Tokens
	8，Segment Embeddings源码实现
	9，Padding源码实现及测试
	10，使用DataLoader实现Batch加载
	11，BERT的初始化init及forward方法源码实现
	12，PositionalEmbeddings源码实现详解
	13，TokenEmbeddings源码
	14，SegmentEmbeddings源码
	15，BERTEmbeddings层源码实现及调试
	16，基于Embeddings之多Linear Transformation操作
	17，Queries、Keys、Values操作源码
	18，Attention机制源码实现
	19，Multi-head Attention源码实现
	20，Layer Normalization数学原理及源码实现
21，Sublayer Connection源码实现
	22，Position-wise Feedforward层源码实现
	23，Dropout数学机制及源码实现
	24，基于Embeddings之上的Linear Transformation及其不同源码实现方式
	25，TransformerBlock源码完整实现及测试
	26，BERT模型训练时候多二分类和多分类别任务数学原理和实现机制
	26，BERT Training Task之MLM源码完整实现及测试
	27，BERT Training Task之NSP源码完整实现及测试
	28，Negative Sampling数学原理及实现源码
	29，MLM和NSP的Loss计算源码实现
	30，BERT模型的训练源码实现及测试
	31，使用小文本训练BERT模型源码、测试和调试
	32，使用特定领域的(例如医疗、金融等)来对BERT进行Pre-training最佳实践
	33，BERT加速训练技巧：动态调整Attention的Token能够Attending的长度
	34，BERT可视化分析

第7章: BERT Fine-tuning源码完整实现、调试及案例实战
	1，数据预处理训练集、测试集源码
	2，文本中的Token、Mask、Padding的预处理源码
	3，数据的Batch处理实现源码及测试
	4，加载Pre-training模型的BertModel及BertTokenizer
	5，模型Config配置
	6，Model源码实现、测试、调试
	7，BERT Model微调的数学原理及工程实践
8，BERT Model参数Frozen数学原理及工程实践
	9，BertAdam数学原理及源码剖析
	10，训练train方法源码详解
	11，fully-connected neural network层源码详解及调试
	12，采用Cross-Entropy Loss Function数学原理及代码实现
	13，Evaluation 指标解析及源码实现
	14，Classification任务下的Token设置及计算技巧
	15，适配特定任务的Tokenization解析
	16，BERT + ESIM(Enhanced Sequential Inference Model)强化BERT模型
	17，使用BERT + LSTM整合强化BERT 模型
	18，基于Movie数据的BERT Fine-tuning案例完整代码实现、测试及调试

第8章: 轻量级ALBERT模型剖析及BERT变种中常见模型优化方式详解
	1，从数学原理和工程实践的角度阐述BERT中应该设置Hidden Layer的维度高于(甚至是高几个数量级)Word Embeddings的维度背后的原因
2，从数学的角度剖析Neural Networks参数共享的内幕机制及物理意义
	3，从数学的角度剖析Neural Networks进行Factorization的机制及物理意义
	4，使用Inter-sentence coherence任务进行模型训练的的数学原理剖析
	5，上下文相关的Hidden Layer Embeddings
	6，上下午无关或不完全相关的Word Embeddings
	7，ALBERT中的Factorized embedding parameterization剖析
	8，ALBERT中的Cross-Layer parameter sharing机制：只共享Attention参数
	9，ALBERT中的Cross-Layer parameter sharing机制：只共享FFN参数
	10，ALBERT中的Cross-Layer parameter sharing机制：共享所有的参数
	11，ALBERT不同Layers的Input和Output相似度分析
	12，训练Task的复杂度：分离主题预测和连贯性预测的数学原因及工程实践
	13，ALBERT中的不同于BERT的 Sentence Negative Sampling
	14，句子关系预测的有效行分析及问题的底层根源
	15，ALBERT的SOP(Sentence Order Prediction)实现分析及工程实践
	16，ALBERT采用比BERT更长的注意力长度进行实际的训练
	17，N-gram Masking LM数学原理和ALERT对其实现分析
	18，采用Quantization优化技术的Q8BERT模型架构解析
	19，采用Truncation优化技术的“Are Sixteen Heads Really Better than One?”模型架构解析
	20，采用Knowledge Distillation优化技术的distillBERT模型架构解析
	21，采用多层Loss计算+知识蒸馏技术的TinyBERT模型架构解析
	22，由轻量级BERT带来的关于Transformer网络架构及实现的7点启示

第9章: ALBERT Pre-training模型及Fine-tuning源码完整实现、案例及调试
	1，Corpus数据分析
	2，Pre-training参数设置分析
	3，BasicTokenizer源码实现
	4，WordpieceTokenizer源码实现
	5，ALBERT的Tokenization完整实现源码
	6，加入特殊Tokens CLS和SEP
	7，采用N-gram的Masking机制源码完整实现及测试
	8，Padding操作源码
	9，Sentence-Pair数据预处理源码实现
	10，动态Token Length实现源码
	11，SOP正负样本源码实现
	12，采用了Factorization的Embeddings源码实现
	13，共享参数Attention源码实现
	14，共享参数Multi-head Attention源码实现
	15，LayerNorm源码实现
	16，共享参数Position-wise FFN源码实现
	17，采用GELU作为激活函数分析
	18，Transformer源码完整实现
	19，Output端Classification和N-gram Masking机制的Loss计算源码
	20，使用Adam进行优化源码实现
	21，训练器Trainer完整源码实现及调试
	22，Fine-tuning参数设置、模型加载
	23，基于IMDB影视数据的预处理源码
	24，Fine-tuning阶段Input Embeddings实现源码
	25，ALBERT Sequence Classification参数结构总结
	26，Fine-tuning 训练代码完整实现及调试
	27，Evaluation代码实现
	28，对Movie数据的分类测试及调试



第10章: 明星级轻量级高效Transformer模型ELECTRA: 采用Generator-Discriminator的Text Encoders解析及ELECTRA模型源码完整实现
	1，GAN：Generative Model和Discriminative Model架构解析
2，为什么说ELECTRA是NLP领域轻量级训练模型明星级别的Model？
	3，使用replaced token detection机制规避BERT中的MLM的众多问题解析
	4，以Generator-Discriminator实现的ELECTRA预训练架构解析
	5，ELECTRTA和GAN的在数据处理、梯度传播等五大区别
	6，ELECTRA数据训练全生命周期数据流
	7，以Discriminator实现Fine-tuning架构解析
	8，ELECTRA的Generator数学机制及内部实现详解
	9，Generator的Loss数学机制及实现详解
	10，Discriminator的Loss数学机制及实现详解
	11，Generator和Discriminator共享Embeddings数据原理解析
	12，Discriminator网络要大于Generator网络数学原理及工程架构
	13，Two-Stage Training和GAN-style Training实验及效果比较
	14，ELECTRA数据预处理源码实现及测试
	15，Tokenization源码完整实现及测试
	16，Embeddings源码实现
	17，Attention源码实现
	18，借助Bert Model实现Transformer通用部分源码完整实现
	19，ELECTRA Generator源码实现
	20，ELECTRA Discriminator源码实现
	21，Generator和Discriminator相结合源码实现及测试
	22，pre-training训练过程源码完整实现
	23，pre-training数据全流程调试分析
	24，聚集于Discriminator的ELECTRA的fine-tuning源码完整实现
	25，fine-tuning数据流调试解析
	26，ELECTRA引发Streaming Computations在Transformer中的应用思考

第11章: 挑战BERT地位的Autoregressive语言模型XLNet剖析及源码完整实现
	1，作为Autoregressive语言模型的XLNet何以能够在发布时在20个语言任务上都能够正面挑战作为Autoencoding与训练领域霸主地位的BERT？
	2，XLNet背后Permutation LM及Two-stream self-attention数学原理解析
3，Autoregressive LM和Autoencoding LM数学原理及架构对比
	4，Denoising autoencoding机制的数学原理及架构设计
	5，对Permutation进行Sampling来高性价比的提供双向信息数学原理
	6，XLNet的Permutation实现架构和运行流程：content stream、query stream
	7，XLNet中的缓存Memory记录前面Segment的信息
	8，XLNet中content stream attention计算
	9，XLNet中query stream attention计算
	10，使用Mask Matrices来实现Two-stream Self-attention
	11，借助Transformer-XL 来编码relative positional 信息
	12，XLNet源码实现之数据分析及预处理
	13，XLNet源码实现之参数设定
	14，Embeddings源码实现
	15，使用Mask实现causal attention
	16，Relative shift数学原理剖析及源码实现
	17，XLNet Relative attention源码完整实现
	18，content stream源码完整实现
	19，queery stream源码完整实现
	20，Masked Two-stream attention源码完整实现
21，处理长文件的Fixed Segment with No Grad和New Segment
	22，使用einsum进行矩阵操作
	23，XLNetLayer源码实现
	24，Cached Memory设置
	25，Head masking源码
	26，Relative-position encoding源码实现
	27，Permutation实现完整源码
	28，XLNet FFN源码完整实现
	29，XLNet源码实现之Loss操作详解
	30，XLNet源码实现之training过程详解
	31，从特定的checkpoint对XLNet进行re-training操作
	32，Fine-tuning源码完整实现
	33，Training Evaluation分析
	34，使用XLNet进行Movies情感分类案例源码、测试及调试

第12章：NLP比赛的明星模型RoBERTa架构剖析及完整源码实现
	1，为什么说BERT模型本身的训练是不充分甚至是不科学的？
2，RoBERTa去掉NSP任务的数学原理分析
3，抛弃了token_type_ids的RoBERTa
	4，更大的mini-batches在面对海量的数据训练时是有效的数学原理解析
	5，为何更大的Learning rates在大规模数据上会更有效？
	6，由RoBERTa对hyperparameters调优的数学依据
	7，RoBERTa下的byte-level BPE数学原理及工程实践
	6，RobertaTokenizer源码完整实现详解
	7，RoBERTa的Embeddings源码完整实现
	8，RoBERTa的Attention源码完整实现
	9，RoBERTa的Self-Attention源码完整实现
	10，RoBERTa的Intermediate源码完整实现
	11，RobertLayer源码完整实现
	12，RobertEncoder源码完整实现
	13，RoBERTa的Pooling机制源码完整实现
	14，RoBERTa的Output层源码完整实现
	15，RoBERTa Pre-trained model源码完整实现
	16，RobertaModel源码完整实现详解
	17，实现Causal LM完整源码讲解
	18，RoBERTa中实现Masked LM完整源码详解
	19，RobertLMHead源码完整实现
	20，RoBERTa实现Sequence Classification完整源码详解
	21，RoBERTa实现Token Classification完整源码详解
	22，RoBERTa实现Multiple Choice完整源码详解
	23，RoBERTa实现Question Answering完整源码详解


第13章：DistilBERT：smaller, faster, cheaper and lighter的轻量级BERT架构剖析及完整源码实现
	1，基于pretraining阶段的Knowledge distillation
	2，Distillation loss数学原理详解
	3，综合使用MLM loss、distillation loss、cosine embedding loss
	4，BERT Student architecture解析及工程实践
	5，抛弃了BERT的token_type_ids的DistilBERT
	6，Embeddings源码完整实现
	7，Multi-head Self Attention源码完整实现
	8，Feedforward Networks源码完整实现
	9，TransformerBlock源码完整实现
	10，Transformer源码完整实现
	11，继承PreTrainedModel的DistilBertPreTrainedModel源码完整实现
	13，DistilBERT Model源码完整实现
	14，DistilBertForMaskedLM源码完整实现
	15，DistilBert对Sequence Classification源码完整实现

第14章: Transformers动手案例系列
	1，动手案例之使用Transformers实现情感分析案例代码、测试及调试
	2，动手案例之使用Transformers实现NER代码、测试及调试
	3，动手案例之使用Transformers实现闲聊系统代码、测试及调试
	4，动手案例之使用Transformers实现Summarization代码、测试及调试
	5，动手案例之使用Transformers实现Answer Span Extraction代码、测试及调试
	6，动手案例之使用Transformers实现Toxic Language Detection Multi-label Classification代码、测试及调试
	7，动手案例之使用Transformers实现Zero-shot learning代码、测试及调试
	8，动手案例之使用Transformers实现Text Clustering代码、测试及调试
	9，动手案例之使用Transformers实现semantics search代码、测试及调试
	10，动手案例之使用Transformers实现IMDB分析代码、测试及调试
	11，动手案例之使用Transformers实现cross-lingual text similarity代码、测试及调试

第15章: Question Generation综合案例源码、测试及调试
	1，从Text到Multiple choice question数学原理、使用的Transformer知识、架构设计
	1，自动生成错误的问题选项
	2，使用GPT2自动生成对错二分类的问题
	3，使用Transformer生成多选题目
	4，使用Transformer自动生成完形填空题目
	5，使用Transformer基于特定内容生成问题
6，完整案例调试分析
	7，使用fastAPI部署、测试Transformer案例
	8，使用TFX部署、测试Transformer案例

第16章：Kaggle BERT比赛CommonLit Readability Prize赛题解析、Baseline代码解析、及比赛常见问题
	1，以问题为导向的Kaggle Data Competition分析
	2，为何Kaggle上的NLP 80%以上都是文本分类比赛，并必须使用Neural Networks？
	3，文本复杂度衡量不同类型实现技术分析
	4，比赛的Training数据集分析：id、url_legal、license、excerpt、target、standard_error
	5，比赛的评价指标分析
	6，Readability：NLP Classification or Regression based on neural networks
	7，Kaggle比赛通用步骤：Data - Cleaning - Store - GridSearch - Model - Prediction
	8，比赛外部数据集分析
	9，比赛使用的硬件条件分析
	10，Training Set、Validation Set、Test Set
	11，比赛的双层Pretraining技术解析
	12，Pretraining的三大类型解析：ITPT、IDPT、CDPT
13，传统的Statistics Method建模 + 树模型
	14，Statistical features构建源码分析
	15，融合统计信息并使用Regression模型解析
	16，使用RoBERTa模型解析
	17，使用AutoModelForMaskedLM
	18，TrainConfig解析
	19，模型的Tokenizer解析
	20，模型加载
	21，对RoBERTa进行pretrain源码解析解决原声BERT和比赛数据领域Discrepancy的问题
	22，Model weights保存时的json和bin解析
	23，使用Kaggle Notebook加载第一次pretrain后的模型
	24，验证集：K-Fold、Sampling等分析
	25，Early stoping分析
	26，把Examples转为Features
	27，DatasetRetriever源码实现详解
	28，Input IDs、Attention Mask、Token type IDs
	28，CommonLitModel源码之regressor解析
	30，CommonLitModel源码之Loss计算
	31，CommonLitModel源码之train方法源码解析
	32，finetuning中的AutoModel
	33，fineturning完整源码解析
	34，Local CV解析
	35，RoBERTa Base + RoBERT Large结合
	36，对不同子模型结果的处理
	37，Classification实现解析
	38，通过Kaggle Kernel对GPU的使用
	39，Submission过程解析
	40，为何比赛时不要私下共享数据？
	41，kernel赛能够在本地进行训练和微调，然后在上传到Kaggle上吗？
	42，如何在kaggle kernel加载外部模型？
	43，RobertaModel提示not initialized的情况下该如何处理？
	44，kernel无法提交应该如何处理？
	45，提交后报错该如何处理？
	46，CV和公开榜单应该更加注重哪一个？
	47，使用BERT比赛的时候最重要的Hyper Parameter是什么？
	48，如何选择GPU训练平台？
	49，在Kaggle上运行Notebook的时候一直是等待状态该怎么处理？
	50，在kernel中如何运行脚本文件？
	51，如何解决BERT训练效果反复波动的情况？
	52，为何看到的效果并不是最终的结果？

第17章： BERT CommonLit Readability Prize比赛技术进阶详解
	1，Data Label based on pairwise comparisions between excerpts
	2，Target中数字为0的原因解析
	3，文本对比中的技巧
	4，target和std构成联合信息
	5，Coarse Validation Loop
	6，private test set
	7，Hold-out validation、K-fold CV validation、Bootstrap resampling
	11，Diversity of models：RoBERTa、BERT、DistilRoBERTa等联合使用
	12，模型参数多样化：不同来源、不同层次的参数及Hyper parameters
	13，多模型结合的training和inference时间复杂度分析
	14，验证集pretraining的意义分析
	15，对embeddings的size的处理
	16，FFN代码分析
	17，warmup数学原理及实现剖析
	18，learning rate scheduler剖析
	19，RoBERTa模型参数结构详解
	20，Data enhancement解析和实现
	21，外部数据集应该用在two-phase pretraining的具体什么阶段？
	22，多样性模型背后的数学原理机制深度剖析
	23，多样性数据来源背后的数学原理剖析
	24，多层次数据编码数学原理分析
	25，One-hot编码和Dense embeddings的巧妙结合
	26，对抗网络的使用分析
	27，长文本处理技巧：head+tail
	28，模型训练不收敛的解决技巧：动态learning rate
	29，联合使用不同类别的预训练模型作为输入的Embedding层来提高收敛速度及避免过拟合背后的数学原理剖析
	30，为何concatenation的embedding很适合Classification任务？
	31，Trainable Parameters开启与停止
	32，Sentence vector：TFIDF、监督任务、及SIF
	33，Adversarial training：FGSM产生Adversary examples揭秘
	34，为何Adversarial training应用到NLP文本到时候一般都是对Embedding层进行Adversary操作？背后的贝叶斯数学原理及神经网络工作机制
	35，Adversarial training的五步骤详解
	36，Adversarial training能够极大的提升NLP效果的数学原理剖析
	37，Adversarial training及Adversarial example在Readability Prize比赛的应用
	38，对每个Batch进行Adversarial training源码解析
	39，Data augmentation方法Easy Data Augmentation解析及实现
	40，基于BERT模型生成高质量的增强数据
	41，孪生网络的使用
	42，Dynamic Padding解析及源码实现
	43，Uniform Length Batching解析及源码实现
	44，Gradient Accumulation解析及源码实现
	45，Freeze Embedding解析及源码实现
	46，Numeric Precision Reduction解析及源码实现
	47，Gradient Checkpoining解析及源码实现
	48，使用memory-profiler来对进程及Python内存使用逐行分析
	49，使用subprocess监视GPU使用
	50，Debiasiing Omission in BertADAM
	51，Re-Initializing Transformer Layers
	52，Utilizing Intermediate Layers
	53，LLRD（Layer-wise Learning Rate Decay）
	54，Mixout Regularization
	55，Pre-trained Weight Decay
	56，Stochastic Weight Averaging
	57，将code存储为dataset存储来更好的使用Kaggle的内存和计算资源
	

第18章：BERT CommonLit Readability Prize比赛中的高分思路及源码解析
	1，Ensemble methods解析
	2，ML中的机器学习：Bagging、Boosting、GBDT等
	3，Kaggle比赛中的Ensemble methods：Vote、Blend、Stacking等
	4，为何Kaggle竞赛中的Ensemble methods会获得更好的精度？
	5，Bagging ensemble method：row-based sampling、column-based sampling等
	6，Bagging ensemble method中的并行训练及预测
	7，Boosting串行训练多个模型：多错误样本权重调整、拟合误差
	8，Blend和Average：对结果基于精度进行加权求和
	9，Stacking：out of fold及交叉验证
	10，模型和特征多样性
	11，比赛对Bagging的使用
	12，比赛对Boosting的使用
	13，深度学习中的模型集成方法：Dropout
	14，训练阶段调整句子顺序Flipping操作
	15，对Ensemble进行Snapshot
	16，Stochstic Weight Averaging操作
	17，Pseudo Label解析：基于方差的标签构建
	18，Kernel赛Pseudo Label和非Kernel赛的Pseudo Lable
	19，Pseudo Lable实现四步骤详解
	20，Knowlede distillation soft label
	21，用于分类的network distillation：embedding layer、transformer layer、prediction layer
	22，public LB及private LB
	23，借助Roberta-large+和训练集相同books的外部数据集
	24，使用Pooling解析及代码实现解析
	25，混合使用不同来源的特征工程结果进行集成
	26，高分作品1完整源码剖析
	27，高分作品2完整源码剖析
	28，高分作品3完整源码剖析
	29，高分作品4完整源码剖析
	30，高分作品5完整源码剖析
	

第19章：NLP阅读理解MRC(Machine Reading Comprehension)数学原理、技术本质及常见算法
	1，以一篇119个Words的GRE(Graduate Record Examinations)文章及2个相应的阅读理解题目为例来剖析阅读理解的过程及其背后的机制
2，MRC在智能客服、机器问答、搜索引擎等等广泛应用背后的原因：规模化价值复制
3，信息的本质及信息理解的本质数学机制剖析
	4，MRC三元素：Question-Context-Answer数学模型及技术本质剖析
	5，MRC的核心：Attention Computations
	6，MRC对信息理解三大层次解析及背后对应的数学模型
	7，MRC实现方法之传统特征工程解析
	8，MRC实现方法之深层语意图匹配解析
	9，MRC实现方式之神经网络及Attention机制解析
	10，MRC数据之Single-Document和Multiple-Document解析
	11，MRC的四大核心任务之Cloze Tests数据集、数学原理和技术本质剖析
	12，MRC的四大核心任务之Multiple Choice数据集、数学原理和技术本质剖析
	13，MRC的四大核心任务之Span Extraction数据集、数学原理和技术本质剖析
	14，MRC的四大核心任务之Free Anwering数据集、数学原理和技术本质剖析
	15，Cloze Tests数据集分析：CNN&Daily Mail、CBT等
	16，Multiple Choice数据集分析：MC Test、RACE等
	17，Span Extraction数据集分析：SQuAD、NewsQA等
	18，Free Answering数据集分析：MS MARCO、DuReader等
	19，MRC的测试集解析：In-domain、Over-sensitivity、Over-stability、Generalization等
	20，MRC的可回答问题及无答案问题数学原理剖析及BERT实现
	21，MRC的Feature extraction数学原理及算法分析
	22，传统Machine Learning Algorithms对MRC 算法解析
	23，BiDAF (Bi-Directional Attention Flow)下的MRC算法解析
	24，QANet下的MRC算法解析
	25，Transformer架构下的BERT及ALBERT下的MRC 解析
	26，Transformer架构下的XLNET下的MRC 解析

第20章：MRC通用架构双线模型内核机制、数学原理、及组件内幕
	1，双线模型架构解析：Multiple Encoders、Interaction
	2，双线模型中为何Interaction环节是实现信息理解的关键？
	3，双线模型底层数学原理剖析
	4，Embeddings下的One-hot Representation及多层One-hot机制解析
	5，Embeddings下的Word2vec的CBOW模型解析及源码实现
	6，Embeddings下的Word2vec的Skipgram模型解析及源码实现
	7，MRC下GloVe: Global Vectors for Word Representation架构解析及源码实现
	8，MRC 下解决一次多义Elmo架构解析及源码实现
	9，使用BERT进行Embeddings架构及最佳实践解析
	10，Feature Extraction下的CNN模型解析及源码实现
	11，Feature Extraction下的RNN模型解析及源码实现
	12，Feature Extraction下的Transformer Encoder或者Decoder的架构解析及源码实现
	13，MRC灵魂：Context-Question Interaction及Question-Context Interaction
	14，Answer Prediction之Word Predictor数学原理及源码实现剖析
	15，Answer Prediction之Option Predictor数学原理及源码实现剖析
	16，Answer Prediction之Span Extractor数学原理及源码实现剖析
	17，Answer Prediction之Answer Generator数学原理及源码实现剖析
	18，MRC中的Negative Sampling数学机制及具体实现
	19，BERT对MRC中无答案问题处理剖析及实现
20，MRC on Knowledge Graph解析
21，对MRC进行Evaluation Metrics之Accuracy、Precision、Recall、F1解析
22，对MRC进行Evaluation Metrices之Rouge-L解析
23，对MRC进行Evaluation Metrics之BLEU解析
24，提升MRC能力的7大方法详解


第21章：基于Bayesian Theory的MRC文本理解基础经典模型算法详解
	1，Bayesian prior在模型训练时候对Weight控制、训练速度影响等功能详解
	2，Bayesian prior能够提供模型训练速度和质量的数学原理剖析
3，从Word2vec走向GloVe：从Local 信息走向Global+Local信息表示模式
	4，GloVe 中的Vector相关性算法
	5，GloVe的Co-occurrence matrix解析
	6，GloVe的Loss计算
	7，神经网络表达信息的三大局限剖析
	7，使用Convolutions取代神经网络传统的matrix multiplication操作
	8，文本序列的Vector表示及Convolutions的天然契合点分析
	9，Parameter sharing背后的数学原理和工程的广泛应用
	10，Vector中的参数真的能够很好的表达信息吗？数学原理及工程实践
	11，TextCNN架构设计解析
	12，CNN-rand数学原理及工程实现
	13，CNN-static数学原理及工程实现
	14，CNN-non-static数学原理及工程实现
	15，CNN-multiple channel数学原理及工程实现
	16，处理长短不一的Sentence
	17，Kernel设置的数学原理及最佳实践
	18，传统模型Attention实现本质：权重分配
	19，通过Soft-Search的方式来构建Attention机制及数学原理剖析
	20，KQV：Attention-based model based on weight allocation
	21，Local-Attention、Global-Attention、Self-Attention对比及最佳实践
	22，基于一维匹配的Attentive Reader架构及数学原理剖析
	23，基于二维匹配的Impatient Reader架构及数学原理剖析
	24，Multi-Hop机制多次提取更充足信息的Attention实现剖析
	25，Multi-Hop机制多次提取更充足信息的TimeStep状态推进剖析
	26，Pointer network和Attention机制的对比
	27，R-NET：借助pointer network和使用gateway机制的attention实现
	28，R-NET的Encoding Layer解析
	29，R-NET的Question-Passage Matching解析
	30，R-NET的Passage Self-Matching解析
	31，R-NET的Answer Prediction解析
	32，Fully-Aware Fusion Network提出的MRC的Fusion层次划分解析
	33，Fully-Aware Fusion Network中的History-of-word机制来更好的理解语意
	34，Fully-Aware Fusion Network的Attention机制解析
	35，Fully-Aware Fusion Network的Encoding Layer：GloVe、CoVe、POS、NER等
	36，Fully-Aware Fusion Network的Multi-level Fusion解析
	37，Fully-Aware Fusion Network的Fully-Aware Self-Boosted Fusion解析
	38，Fully-Aware Fusion Network的Output layer解析
	39，QA-Net的架构之Embedding Encoder Layer解析
	40，QA-Net的架构之Context-Query Attention Layer解析
	41，QA-Net的架构之Model Encoder Layer解析
	42，QA-Net的架构之Output Layer解析

第22章：揭秘针对Cloze Tests基于Attention机制的的MRC领域开山之作：Teaching Machines to Read and Comprehend架构设计及完整源码实现
	1，对Text提供精细化的语言理解能力和推理能力的MRC为何需要Neural Networks和Attention机制的支持？
	2，基于大规模训练数据集的集特征工程和分类于一体的深度学习MRC
	3，数据集结构分析
	4，Two-layer Deep LSTM Reader的Input和Output分析
	5，Two-layer Deep LSTM Reader中article和question的Concatenation操作
	6，Two-layer Deep LSTM Reader中的Embedding Layer解析
	7，具有Attention功能的Two-layer Deep LSTM Reader架构解析
	8，Two-layer Deep LSTM Reader的classification解析
	9，Attentive Reader的Input时候对Document和Question分别作LSTM建模
	10，Attentive Reader使用加法操作实现Attention机制进行Classification操作
	11，Impatient Reader的Output中的Attention数学原理和操作解析
	12，对模型复杂度及数据量的最佳实践思考
	13，为何Attention机制在阅读理解中是有效的？数学原理和工程实践
	14，CNN Daily Mail数据Padding、Batch等预处理操作
	15，QADataset完整源码解析
	16，QAIterator完整源码解析
	17，Context和Question进行Concatenation操作完整源码解析
	18，Deep LSTM中的Word Embedding Layer实现
	19，Deep LSTM中的Contextual Embedding Layer实现
	20，Deep LSTM中的Output Layer实现
	21，Deep LSTM中的Dropout
	22，Deep LSTM中的Modeling Layer源码实现
	23，AttentiveReader中的Word Embedding Layer实现
	24，AttentiveReader中的Contextual Embedding Layer实现
	25，AttentiveReader中的Modeling Layer实现
	26，AttentiveReader中的Attention机制实现
	27，ImpatientReader中的Embedding Layers实现
	28，ImpatientReader中的Mdoeling Layer实现
	29，ImpatientReader中的Attention源码完整实现
	30，training方法的源码完整实现
	31，对整个整个算法完整源码实现的调试及分析

第23章：MRC经典的Span Extraction模型Bi-DAF 算法架构、运行机制及数学原理
	1，双向Attention Flow：Query2Context、Context2Query数学原理及工程实现
	2，Bi-DAF能够正式确立编码-交互-输出三层架构阅读理解模型背后的原因分析
	3，Bi-DAF模型本身的五层架构及其背后数学原理解析
	4，不同粒度的多阶段Embeddings层的架构设计和数学原理
	5，Bonus：多阶段Embeddings在智能对话信息表示中的应用剖析
	6，Character Embedding数学原理及Char-CNN实现解析
	7，Word Embedding数学原理及GloVe实现解析
8，双向LSTM架构机制及数学原理剖析
9，使用Highway Network解决梯度问题的数学原理及实现解析
	10，组合Char embedding和word embedding
	11，Contextual Embedding数学原理及实现解析
	12，Bi-DAF中的Context2Query实现解析
	13，Bi-DAF中的Query2Context实现解析
	14，Trainable Matrix for attention mechanism
	15，Modeling层架构和数学原理剖析
	15，输出层的Start index计算数学原理解析
	16，输出层的End index计算数学原理解析
	17，Training Loss计算解析
	18，参数设置
	18，Bi-DAF在信息抽取时候的Assumption存在的问题分析
	19，为何Bi-DAF不擅长回答Why类型的问题？如何改进？
	
第24章：基于SQuAD对Bi-DAF进行MRC源码完整实现、测试和调试
	1，SQuAD训练集和验证集数据分析及answer的Index机制分析
	2，从JSON文件中获取样本信息
	3，Tokenization代码实现
	4，遍历处理data中的paragraphs下context及qas
	5，对data中answer的start index、text、end index的处理及word的处理
	6，构建基于Batch的Iterator
	7，Padding源码实现及测试
	8，Character Embedding Layer对Char进行Vector编码实现和测试
	9，Word Embedding Layer对word进行Vector编码实现及测试
	10，dropout操作
	11，Convolutions操作实现
	12，Transformer数据流源代码剖析
	13，Concatenate Context和Question完整源码实现
	14，通过基于ReLU的highway network来整合信息完整源码实现及测试
	15，highway network中的门控机制数学原理及其在AI中的广泛应用
	16，通过LSTM对Contextual Embedding Layer进行编码完整实现及测试
	17，Context Matrix和Question Matrix可视化分析
	18，attention flow layer中相似矩阵S的源码实现
	19，Context2Query完整源码实现及测试
	20，Query2Context完整源码实现及测试
	21，attention flow layer中信息前向和增强信息表示的G的融合源码实现
	22，Modeling Layer完整源码实现调试分析
	23，output layer中p1的计算实现
	24，output layer中p2的计算实现
	25，Cross Entropy Loss p1的细节说明
	26，在验证集上进行Test源码完整实现
	27，Mask机制的具体作用剖析及调试分析
	28，对Answer进行Normalization操作
	29，EM (Exact Match) 数学公式解析及源码实现
	30，F1对MRC的应用数学公式解析及源码实现
	31，Evaluation完整源码实现及调试
	32，Soft Evaluation的重大意义思考
	33，Bi-DAF全流程调试及深度分析

