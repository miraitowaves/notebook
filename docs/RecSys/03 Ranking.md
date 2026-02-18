在完成候选的快速筛选之后，精排需要对千级候选进行精细化的偏好预测。这个阶段要在可接受的延迟内平衡准确性、泛化能力与稳定性

精排模型的发展有一条比较清晰的脉络。Wide & Deep 模型将线性模型的记忆能力和深度网络的泛化能力结合起来，成为了一个实用的基础框架。随着对特征交互重要性的认识加深，从 FM 开始，到 DeepFM、xDeepFM，再到基于注意力的自动化交互建模，这些方法让模型能够更好地处理特征间的复杂关系

考虑到用户兴趣的多样性和时序特点，序列建模技术也被引入到精排中。DIN 关注用户兴趣的多样性，DIEN 进一步建模兴趣演化，DSIN 则处理会话序列，这些发展帮助模型更好地理解用户的动态偏好

在实际应用中，往往需要同时优化多个目标，并且要适应不同的业务场景。多目标优化和多场景建模通过合理的架构设计、任务关系建模及动态权重策略，让精排模型能够在复杂的业务环境中取得更好的业务效果

下面让我们进入推荐系统的核心环节——精排模型的世界

---

## 01 记忆与泛化

在构建推荐模型时，我们常常追求两个看似矛盾的目标：**记忆（Memorization）** 与**泛化（Generalization）**

- **记忆能力**，指的是模型能够学习并记住那些在历史数据中频繁共同出现的特征组合。例如，模型记住“买了 A 的用户，通常也会买 B”。这种能力可以精准地**捕捉显性、高频的关联，为用户提供与他们历史行为高度相关的推荐**
- **泛化能力**，就是模型能学到特征间的深层关系，处理训练时很少见到的特征组合。举个例子，模型发现“物品 A 和物品 C 都是同一类的，用户喜欢这类东西”，那就可以给喜欢 A 的用户推荐 C，哪怕用户以前没见过 C。这能让推荐更丰富一些

怎么让一个模型同时做好这两件事呢？这确实不容易。2016 年 Google 提出的 Wide & Deep 模型给了一个不错的思路。这个模型的想法很直接：既然需要两种能力，那就设计两个部分，然后让它们一起训练，通过 **联合训练（Joint Training）** 的方式配合工作

模型的设计思路是把结构分成两块，各自负责不同的事情：

![[public/Pasted image 20260114153738.png|350]]

**记忆的捷径：Wide 部分**

Wide 部分本质上是一个广义线性模型，比如逻辑回归。它的优势在于结构简单、可解释更强，并且能高效地“记忆”那些显而易见的关联规则。其数学表达形式如下：

![[public/Pasted image 20260114153804.png|450]]

其中，y 是预测值，$\omega$ 是模型权重，$x$ 是特征向量， $b$ 是偏置项。

Wide 部分的关键在于其输入的特征向量。它不仅包含原始特征，更重要的是包含了大量**人工设计的交叉特征（Cross-product Features）**。交叉特征可以将多个独立的特征组合成一个新的特征，用于捕捉特定的共现模式

例如，在应用商店的推荐场景中，我们可以创建一个交叉特征 `AND(installed_app=photo_editor, impression_app=filter_pack)`，它代表用户已经安装了“照片编辑器”应用，并且现在看到了“滤镜包”应用的推荐

通过这种方式，Wide 部分能够直接、快速地学习到“照片编辑器用户对滤镜包应用有更高的安装意愿”这类强关联规则，正是“记忆能力”的直接体现

**核心代码**

Wide 部分的关键在于处理交叉特征。对于每一对需要交叉的特征，模型会创建一个专门的权重来记住它们的共现模式：

```python
# 遍历所有需要交叉的特征对
for i in range(len(cross_feature_columns)):
    for j in range(i + 1, len(cross_feature_columns)):
        fc_i = cross_feature_columns[i]
        fc_j = cross_feature_columns[j]

        # 获取两个特征的输入
        feat_i = input_layer_dict[fc_i.name]  # [B, 1]
        feat_j = input_layer_dict[fc_j.name]  # [B, 1]

        # 为每个特征对创建独立的权重表
        cross_vocab_size = fc_i.vocab_size * fc_j.vocab_size
        cross_embedding = Embedding(
            input_dim=cross_vocab_size,
            output_dim=1,  # 标量权重，直接记住这对特征的影响
            name=f"cross_{fc_i.name}_{fc_j.name}"
        )

        # 将特征对组合成单一索引并查找权重
        combined_index = feat_i * fc_j.vocab_size + feat_j
        cross_weight = cross_embedding(combined_index)  # 查表得到这对特征的权重
        cross_weights.append(cross_weight)

# 所有交叉特征权重相加
cross_logits = tf.add_n(cross_weights)
```

这段代码的设计体现了 Wide 部分的本质：为每个特征组合分配一个独立的权重，通过查表操作直接“记住”历史数据中的共现模式

**学习复杂关系：Deep 部分**

Deep 部分是一个标准的前馈神经网络（DNN），它负责模型的“泛化能力”。与 Wide 部分依赖人工特征工程不同，Deep 部分可以自动学习特征之间的高阶、非线性关系

它的工作流程如下：首先，对于那些高维稀疏的类别特征（如用户 ID、物品 ID），通过一个**嵌入层（Embedding Layer）** 将它们映射为低维、稠密的向量。**这些嵌入向量能够捕捉到特征的潜在语义信息，是实现泛化的基础**

例如，《流浪地球》和《三体》的电影 ID 在嵌入空间中的距离，可能会比《流浪地球》和《熊出没》更近

随后，这些嵌入向量与其他数值特征拼接在一起，被送入多层神经网络中进行前向传播：

![[public/Pasted image 20260114154657.png|450]]

其中，$a^{(l)}$ 是第 $l$ 层的激活值，$W^{(l)}$ 和 $b^{(l)}$ 是该层的权重和偏置，$f$ 是激活函数（如 ReLU）。通过逐层抽象，DNN 能够发掘出数据中隐藏的复杂模式，从而对未曾见过的特征组合也能做出合理的预测

**核心代码**

Deep 部分的实现分为两个关键步骤：首先将类别特征映射为稠密向量，然后通过多层神经网络学习高阶特征交互：

- 在推荐系统的特征工程（FE）中，通常会将数以百计的特征按照**属性来源**划分为不同的“组”

```python
# 1. 特征嵌入：将稀疏的类别特征转换为稠密向量
group_feature_dict = {}
for group_name, _ in group_embedding_feature_dict.items():
    group_feature_dict[group_name] = concat_group_embedding(
        group_embedding_feature_dict, group_name, axis=1, flatten=True
    )  # B x (N * D) - 拼接所有特征的嵌入向量

# 2. 深度神经网络：逐层学习特征的非线性组合
deep_logits = []
for group_name, group_feature in group_feature_dict.items():
    # 构建多层神经网络
    deep_out = DNNs(
        units=dnn_units,  # 例如 [64, 32]
        activation="relu",  # ReLU激活函数
        dropout_rate=dnn_dropout_rate
    )(group_feature)

    # 输出层：将深度特征映射为预测分数
    deep_logit = tf.keras.layers.Dense(1, activation=None)(deep_out)
    deep_logits.append(deep_logit)
```

这种设计使得模型能够自动学习特征的语义表示，例如将“物品 A”相关的特征映射到向量空间的相近位置，从而实现对未见过的特征组合的泛化预测

**两者结合**

Wide & Deep 模型通过联合训练，将两部分的输出结合起来进行最终的预测。其预测概率如下：

![[public/Pasted image 20260114155757.png|500]]

在这里， $\sigma$ 是 Sigmoid 函数，$[x, \phi(x)]$ 代表 Wide 部分的输入（包含原始特征和交叉特征），$a^{(lf)}$ 是 Deep 部分最后一层的输出向量，$w_{wide} w_{deep}$ 和 b 是最终预测层的权重和偏置。模型的梯度在反向传播时会同时更新 Wide 和 Deep 两部分的所有参数

一个值得注意的工程细节是，由于两部分处理的特征类型不同，它们通常会采用不同的优化器

- **Wide 部分**的输入特征非常稀疏，常使用带 L1 正则化的 FTRL ([Ferreira and Soares, 2025](https://datawhalechina.github.io/fun-rec/chapter_references/references.html#id46 "Ferreira, R. N., & Soares, C. (2025). Follow-the-regularized-leader with adversarial constraints. arXiv preprint arXiv: 2503.13366.")) 等优化器。L1 正则化可以产生稀疏的权重，相当于自动进行特征选择，让模型只“记住”重要的规则
- **Deep 部分**的参数是稠密的，更适合使用像 AdaGrad ([Duchi _et al._, 2011](https://datawhalechina.github.io/fun-rec/chapter_references/references.html#id69 "Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 12 (7).")) 或 Adam ([Kingma and Ba, 2014](https://datawhalechina.github.io/fun-rec/chapter_references/references.html#id70 "Kingma, D. P., & Ba, J. (2014). Adam: a method for stochastic optimization. arXiv preprint arXiv: 1412.6980.")) 这样的优化器

**核心代码**

联合训练的核心是将 Wide 和 Deep 两部分的输出进行融合：

```python
# Wide部分：线性特征 + 交叉特征
linear_logit = get_linear_logits(input_layer_dict, feature_columns)
cross_logit = get_cross_logits(input_layer_dict, feature_columns)

# Deep部分：多个特征组的深度网络输出
deep_logits = []
for group_name, group_feature in group_feature_dict.items():
    deep_out = DNNs(units=dnn_units, activation="relu", dropout_rate=dnn_dropout_rate)(
        group_feature
    )
    deep_logit = tf.keras.layers.Dense(1, activation=None)(deep_out)
    deep_logits.append(deep_logit)

# 联合训练：将Wide和Deep的输出相加
wide_deep_logits = add_tensor_func(deep_logits + [linear_logit, cross_logit])

# 最终预测：通过sigmoid函数输出点击概率
output = tf.keras.layers.Dense(1, activation="sigmoid")(wide_deep_logits)
```

Wide & Deep 模型的意义不只是提供了一个新的网络结构，更重要的是给出了一个思路：怎么把记忆能力和泛化能力结合起来。该模型不仅成为了许多推荐业务的基线模型，更为后续精排模型的发展提供了重要的参考

---

## 02 特征交叉

前面我们讲了 Wide & Deep 模型，它把记忆能力和泛化能力结合起来。不过 Wide 部分有个问题：需要人工设计交叉特征，比如“用户年龄×商品类别”这样的组合。这种手工设计的方式不仅费时费力，还很难覆盖所有有用的特征组合

既然手工设计这么麻烦，那能不能让模型自己学会做特征交叉呢？这就是本节要讨论的核心问题。我们会按照两条技术路线来看：先从简单的二阶交叉开始，然后到更复杂的高阶交叉，最后看看怎么让交叉变得更个性化和自适应

### 2.1 二阶特征交叉

Wide & Deep 模型虽然不错，但手工设计特征交叉实在太麻烦了。能不能让机器自己学会特征之间的关系呢？这就是我们要解决的问题。最直接的想法是让模型自动捕捉所有特征对之间的交互关系。但这里有个大问题：推荐系统的特征动辄成千上万，如果每两个特征都要学一个参数，参数量会爆炸。而且推荐数据本身就很稀疏，大部分特征组合根本没有足够的样本来训练

所以关键是要找到一种巧妙的方法，既能自动学习特征交叉，又不会让参数太多。解决了这个问题后，还得考虑怎么把这些学到的交叉特征和深度网络结合起来

#### FM

还记得我们在召回章节 [2.2.2.1节](https://datawhalechina.github.io/fun-rec/chapter_1_retrieval/2.embedding/2.u2i.html#fm-matching-model) 遇到的 FM 吗？当时我们看到它如何巧妙地将用户和物品分解成向量，通过内积实现高效的双塔召回。现在到了精排阶段，FM 又要展现它的另一面了

在召回时，FM 主要解决的是“如何快速从海量物品中找到候选集”的问题。但在精排阶段，我们面临的挑战完全不同：**如何自动学习特征之间的交叉关系，而不用手工一个个去设计**

这时候 FM 的核心思想就派上用场了——**给每个特征学一个向量表示，然后用向量内积来捕捉特征间的关系**。听起来很简单对吧？但这个简单的想法解决了一个大问题：不管你有多少特征，不管特征组合有多复杂，都能用同一套方法来处理。最关键的是，参数数量不会爆炸式增长，这对于推荐系统这种特征超多的场景来说太重要了

![[public/Pasted image 20260114163533.png|400]]

$$
\text{图3.2.1 FM 模型结构}
$$

为了捕捉特征间的交互关系，一个直接的想法是在线性模型的基础上增加所有特征的二阶组合项，即多项式模型：

$$y=w_0+\sum_{i=1}^nw_ix_i+\sum_{i=1}^{n-1}\sum_{j=i+1}^nw_{ij}x_ix_j$$

其中，$w_0$ 是全局偏置项，$w_i$ 是特征 $x_i$ 的权重，$w_{ij}$ 是特征 $x_i$ 和 $x_j$ 交互的权重，$n$ 是特征数量。这个模型存在两个致命缺陷：第一，参数数量会达到 $O(n^2)$ 的级别，在特征数量庞大的推荐场景下难以承受；第二，在数据高度稀疏的环境中，绝大多数的交叉特征 $x_ix_j$ 因为在训练集中从未共同出现过，导致其对应的权重 $w_{ij}$ 无法得到有效学习

FM 模型巧妙地解决了这个问题。它将交互权重 $w_{ij}$ 分解为两个低维隐向量的内积，即 $w_{ij}=\langle\mathbf{v}_i,\mathbf{v}_j\rangle$。这样，模型的预测公式就演变为：

$$y=w_0+\sum_{i=1}^nw_ix_i+\sum_{i=1}^{n-1}\sum_{j=i+1}^n\langle\mathbf{v}_i,\mathbf{v}_j\rangle x_ix_j$$

这种参数共享的设计是 FM 的精髓所在。原本需要学习 $O(n^2)$ 个独立的交叉权重 $w_{ij}$, 现在只需要为每个特征学习一个 $k$ 维的隐向量 $v_i$, 总参数量就从 $O(n^2)$ 降低到了 $O(nk)$。更重要的是，它极大地缓解了数据稀疏问题。即使特征 $i$ 和 $j$ 在训练样本中从未同时出现过，模型依然可以通过它们各自与其他特征 (如 $k$) 的共现数据，分别学到有效的隐向量 $v_i$ 和 $v_j$。只要隐向量学习得足够好，模型就能够泛化并预测 $x_i$ 和 $x_j$ 的交叉效果。此外，通过巧妙的数学变换，FM 的二阶交叉项计算复杂度可以从 $O(kn^2)$ 优化到线性的 $O(kn)$, 使其在工业界得到了广泛应用

**核心代码**

FM 的核心在于将 $O(n^2)$ 的二阶交叉项优化为线性复杂度。通过简单的代数变换，我们可以高效计算所有特征对的交互：

```python
# FM层的核心计算：0.5 * ((sum(v))^2 - sum(v^2))
# inputs: [batch_size, field_num, embedding_size]

# 先求和再平方：(∑v_i)^2
square_of_sum = tf.square(
    tf.reduce_sum(inputs, axis=1, keepdims=True)
)  # [B, 1, D]

# 先平方再求和：∑(v_i^2)
sum_of_square = tf.reduce_sum(
    inputs * inputs, axis=1, keepdims=True
)  # [B, 1, D]

# FM二阶交互项
cross_term = 0.5 * tf.reduce_sum(
    square_of_sum - sum_of_square, axis=2
)  # [B, 1]
```

这个实现的巧妙之处在于，无论有多少特征，计算复杂度始终保持线性，使得 FM 能够处理推荐系统中常见的高维稀疏特征

#### AFM

FM 对所有特征交叉给予了相同的权重，但实际上不同交叉组合的重要性是不同的。AFM ([Xiao _et al._, 2017](https://datawhalechina.github.io/fun-rec/chapter_references/references.html#id48 "Xiao, J., Ye, H., He, X., Zhang, H., Wu, F., & Chua, T.-S. (2017). Attentional factorization machines: learning the weight of feature interactions via attention networks. arXiv preprint arXiv: 1708.04617.")) 在此基础上引入注意力机制，为不同的特征交叉分配权重，使模型能关注到更重要的交互。例如，在预测一位用户是否会点击一条体育新闻时，“用户年龄=18-24 岁”与“新闻类别=体育”的交叉，其重要性显然要高于“用户年龄=18-24 岁”与“新闻发布时间=周三”的交叉

![[public/Pasted image 20260114165710.png|500]]

AFM 的模型结构在 FM 的基础上进行了扩展。它首先将所有成对特征的隐向量进行**元素积（Hadamard Product, 记为 : $\odot$ ）**，而不是像 FM 那样直接求内积。这样做保留了交叉特征的向量信息，为后续的注意力计算提供了输入。这个步骤被称为成对交互层（Pair-wise Interaction Layer）

$$f_{PI}(\mathcal{E})=\left\{(v_i\odot v_j)x_ix_j\right\}_{(i,j)\in\mathcal{R}_x}$$

其中，$\mathcal{E}$ 表示输入样本中所有非零特征的 Embedding 向量集合，$\mathcal{R}_x$ 表示输入样本中所有非零特征的索引对集合

随后，模型引入一个注意力机制，来学习每个交叉特征 $(v_i\odot v_j)$ 的重要性得分 $a_{ij}$

$$a'_{ij}=\mathbf{h}^T\text{ReLU}(\mathbf{W}(v_i\odot v_j)x_ix_j+\mathbf{b})$$

$$a_{ij}=\frac{\exp(a'_{ij})}{\sum_{(i,k)\in\mathcal{R}_x}\exp(a'_{ik})}$$

其中，$\mathbf{W}$ 是注意力网络的权重矩阵，$\mathbf{b}$ 是偏置向量，$\mathbf{h}$ 是输出层向量。这个得分 $a_{ij}$ 经过 Softmax 归一化后，被用作加权求和的权重，与原始的交叉特征向量相乘，最终汇总成一个向量。这个过程被称为注意力池化层（Attention-based Pooling）

$$f_{Att}=\sum_{(i,j)\in\mathcal{R}_x}a_{ij}(v_i\odot v_j)x_ix_j$$

最后，AFM 的完整预测公式由一阶线性部分和经过注意力加权的二阶交叉部分组成：

$$\hat{y}_{afm}(x)=w_0+\sum_{i=1}^{n}w_ix_i+\mathbf{P}^Tf_{Att}$$

其中  $P$ 是一个投影向量，用于将最终的交叉结果映射为标量。通过引入注意力机制，**AFM 不仅提升了模型的表达能力，还通过可视化注意力权重 $a_{ij}$ 赋予了模型更好的可解释性**，让我们可以洞察哪些特征交叉对预测结果的贡献最大

**核心代码**

AFM 的关键在于注意力池化层，它为每个特征交叉对分配不同的权重：

```python
# 1. 计算所有特征对的元素积交互
# group_pairwise: [batch_size, num_pairs, embedding_dim]
group_pairwise = pairwise_feature_interactions(
    group_feature, drop_rate=dropout_rate
)

# 2. 注意力权重计算：h^T · ReLU(W · (v_i ⊙ v_j) + b)
weighted_inputs = tf.matmul(
    group_pairwise, attention_weight
) + attention_bias  # [B, num_pairs, attention_factor]

activation = tf.nn.relu(weighted_inputs)
projected = tf.matmul(activation, attention_projection)  # [B, num_pairs, 1]

# 3. Softmax归一化得到注意力权重
attention_weights = tf.nn.softmax(projected, axis=1)

# 4. 加权求和：∑ a_ij · (v_i ⊙ v_j)
attention_output = tf.reduce_sum(
    tf.multiply(group_pairwise, attention_weights), axis=1
)  # [B, D]
```

相比 FM 对所有特征交叉一视同仁，AFM 通过注意力机制自动识别重要的交互模式，提升了模型的表达能力和可解释性

#### NFM

NFM ([He and Chua, 2017](https://datawhalechina.github.io/fun-rec/chapter_references/references.html#id47 "He, X., & Chua, T.-S. (2017). Neural factorization machines for sparse predictive analytics. Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval (pp. 355–364).")) 探索了如何更深入地利用交叉信息。它将 FM 的二阶交叉结果（用哈达玛积表示的向量）作为输入，送入一个深度神经网络（DNN），从而在 FM 的基础上学习更高阶、更复杂的非线性关系。NFM 的核心思想是，**FM 所捕获的二阶交叉信息本身就是一种非常有价值的特征，可以作为“原料”输入给强大的 DNN，由 DNN 来自动学习这些交叉特征之间的高阶组合关系**

---

## 03 序列建模

在上一节中，我们探讨了如何通过各类特征交叉模型，让机器自动学习特征之间复杂的组合关系。无论是二阶交叉的 FM、AFM，还是高阶交叉的 DCN、xDeepFM，它们的核心目标都是从一个静态的特征集合中挖掘出有价值的信息。然而，这些模型普遍存在一个共同的局限：它们大多将用户的历史行为看作一个无序的“物品袋”（a bag of items），如同用户的兴趣是一个静态的表示

但用户的兴趣不是静止的，而是具有明显的**时序性**和**动态演化**特点。一个用户先浏览“鼠标”再浏览“显示器”，与先浏览“小说”再浏览“显示器”，这两个行为序列背后指向的购买意图截然不同。前者可能是一位正在组装电脑的数码爱好者，而后者可能只是在工作之余的随性浏览。传统的特征交叉模型难以捕捉这种蕴含在行为顺序中的、随时间变化的意图

因此，本节我们将转换视角，不再将用户历史看作一堆静态特征的集合，而是将其视为一个动态的序列。我们将聚焦于如何对用户的行为序列进行建模，从这个序列中挖掘出用户动态、演化的兴趣。接下来，我们将介绍工业界在序列建模方向上的三个代表性模型：DIN、DIEN 和 DSIN，看看它们是如何解决这个核心挑战的

### 3.1 局部激活的注意力机制

在大型电商平台中，用户的兴趣是**多样**的。一个用户可能在一段时间内，既关注数码产品，又浏览运动装备，还会购买生活用品。在传统的深度学习模型（即 Embedding&MLP 范式）中，通常的做法是将用户所有的历史行为（如点击过的商品 ID）对应的 Embedding 向量通过池化（Pooling）操作，压缩成一个**固定长度的向量**来代表该用户

这个固定长度的用户向量，很快就成为了表达用户多样兴趣的瓶颈。想象一下，无论系统准备向这个用户推荐“跑鞋”还是“手机”，代表他的都是同一个向量。这个向量试图“一视同仁”地蕴含该用户所有的兴趣点，这不仅非常困难，而且在面对具体推荐任务时显得不够聚焦。为了增强表达能力而粗暴地增加向量维度，又会带来参数量爆炸和过拟合的风险

**DIN 的核心思想：局部激活 (Local Activation)**

深度兴趣网络（Deep Interest Network, DIN） ([Zhou _et al._, 2018](https://datawhalechina.github.io/fun-rec/chapter_references/references.html#id52 "Zhou, G., Zhu, X., Song, C., Fan, Y., Zhu, H., Ma, X., … Gai, K. (2018). Deep interest network for click-through rate prediction. Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 1059–1068).")) 的提出者们发现，用户的某一次具体点击行为，通常只由其历史兴趣中的**一部分**所“激活”。当向一位数码爱好者推荐“机械键盘”时，真正起决定性作用的，很可能是他最近浏览“游戏鼠标”和“显卡”的行为，而不是他上个月购买的“跑鞋”

基于此，DIN 提出了一个观点：**用户的兴趣表示不应该是固定的，而应是根据当前的候选广告（Candidate Ad）不同而动态变化的**

![[public/Pasted image 20260114190800.png|650]]

**技术实现：注意力机制**

为了实现“局部激活”这一思想，DIN 在模型中引入了一个关键模块——**局部激活单元（Local Activation Unit）**，其本质就是**注意力机制**。如上图右侧所示，DIN 不再像基准模型 ([图3.3.1](https://datawhalechina.github.io/fun-rec/chapter_2_ranking/3.sequence.html#din-architecture) 左) 那样对所有历史行为的 Embedding 进行简单的池化，而是进行了一次“加权求和”

这个权重 (即注意力分数) 的计算，体现了 DIN 的核心思想。具体来说，对于一个给定的用户 U 和候选广告 A，用户的兴趣表示向量 $\boldsymbol{v}_U(A)$ 是这样计算的：

$$\boldsymbol{v}_U(A)=f(\boldsymbol{v}_A,\boldsymbol{e}_1,\boldsymbol{e}_2,\ldots,\boldsymbol{e}_H)=\sum_{j=1}^Ha(\boldsymbol{e}_j,\boldsymbol{v}_A)\boldsymbol{e}_j=\sum_{j=1}^Hw_j\boldsymbol{e}_j$$

其中：

$\bullet e_1,e_2,\ldots,e_H$ 是用户 U 的历史行为 Embedding 向量列表

$\bullet v_A$ 是候选广告 A 的 Embedding 向量

$\bullet a(\boldsymbol{e}_j,\boldsymbol{v}_A)$ 是一个激活单元 (通常是一个小型前馈神经网络), 它接收历史行为 $\boldsymbol{e}_j$ 和候选广告 $\boldsymbol{v}_A$ 作为输入

输出一个权重 $\boldsymbol{w}_j$。这个权重就代表了历史行为 $\boldsymbol{e}_j$ 在面对广告 $\boldsymbol{v}_A$ 时的"相关性"或"注意力得分"

一个值得注意的细节是，DIN 计算出的注意力权重没有经过 Softmax 归一化。这意味着不一定等于 1。这样设计的目的是为了保留用户兴趣的绝对强度。例如，如果一个用户的历史行为大部分都与某个广告高度相关，那么加权和之后的向量模长就会比较大，反之则较小。这种设计使得模型不仅能捕捉兴趣的“方向”，还能感知兴趣的“强度”

**核心代码**

DIN 的注意力机制通过将候选广告与历史行为进行多角度交互来计算权重：

```python
# DIN注意力层的核心计算
# query: 候选广告 [batch_size, 1, embedding_dim]
# keys: 历史行为序列 [batch_size, seq_len, embedding_dim]

query = tf.squeeze(query, axis=1)  # [B, H]
length = tf.shape(keys)[-2]
query = tf.expand_dims(query, axis=1)  # [B, 1, H]

# 构建多角度交互特征：query, keys, query-keys, query*keys
att_inputs = tf.concat([
    tf.tile(query, [1, length, 1]),  # 重复query以匹配序列长度
    keys,                             # 历史行为
    query - keys,                     # 差异特征
    query * keys                      # 元素积特征
], axis=-1)  # [B, L, 4*H]

# 通过前馈网络计算注意力分数
hidden_layer = ffn_layer(att_inputs)  # [B, L, hidden_units]
scores = tf.keras.layers.Dense(1)(hidden_layer)  # [B, L, 1]

# 应用mask并进行加权求和（注意：不使用softmax归一化）
attention_weights = scores * mask  # [B, L, 1]
user_interest = tf.reduce_sum(keys * attention_weights, axis=1)  # [B, H]
```

这种设计的关键在于：通过 `[query, keys, query-keys, query*keys]` 四种交互方式，模型能够从多个角度衡量历史行为与候选广告的相关性，同时不使用 softmax 归一化以保留兴趣强度信息

### 3.2 兴趣的演化建模

DIN 成功地捕捉了用户兴趣的“多样性”和“局部激活”特性，但它仍然存在一个局限：它将用户的历史行为看作是一个无序的集合，忽略了行为之间的**时序依赖关系**。用户的兴趣不仅是多样的，更是在持续**演化**的

怎么解决这个问题呢？深度兴趣演化网络（Deep Interest Evolution Network, DIEN） ([Zhou _et al._, 2019](https://datawhalechina.github.io/fun-rec/chapter_references/references.html#id53 "Zhou, G., Mou, N., Fan, Y., Pi, Q., Bian, W., Zhou, C., … Gai, K. (2019). Deep interest evolution network for click-through rate prediction. Proceedings of the AAAI conference on artificial intelligence (pp. 5941–5948).")) 给出了答案。它的想法很简单：光知道用户过去喜欢什么还不够，还得搞清楚这些兴趣是怎么变化的，这样才能更好地预测用户接下来会喜欢什么

![[public/Pasted image 20260114191150.png|550]]

DIEN 的核心想法其实很有意思：用户的点击、购买这些行为只是表面现象，真正重要的是藏在背后的 **“兴趣”状态**。比如你今天买了本编程书，明天又买了个键盘，这些行为背后可能反映的是你对编程越来越感兴趣。DIEN 就是要抓住这种兴趣变化的规律，所以它设计了一个两阶段的结构来实现这个目标

**第一阶段：兴趣提取层 (Interest Extractor Layer)**

这一层要做的事情就是从用户的行为序列中，找出真正能反映兴趣状态的信息。DIEN 用 GRU 来按时间顺序处理用户的行为 Embedding 序列 $e_1, e_2, \ldots, e_T$。按理说，GRU 在 $t$ 时刻的隐状态 $h_t$ 应该包含了到那个时刻为止的所有信息。但问题是，这个隐状态真的能准确表示用户的“兴趣”吗？

DIEN 的做法很巧妙：既然我们说 $t$ 时刻的兴趣状态能反映用户的真实想法，那它应该能预测用户接下来会做什么，对吧？所以 DIEN 加了一个辅助损失 (Auxiliary Loss)，让 $t$ 时刻的兴趣状态 $h_t$ 去预测用户在 $t+1$ 时刻的真实行为 $e_{t+1}$。这样一来，模型就被“逼着”学出更有意义的兴趣表示。具体地，辅助损失 $L_{aux}$ 定义如下：

$$L_{aux} = -\frac{1}{N}\left(\sum_{i=1}^{N}\sum_{t=1}^{T}\log\sigma(h_t^i, e_{b[t+1]}^i) + \log(1-\sigma(h_t^i, \hat{e}_{b[t+1]}^i))\right)$$

其中：

- $h_t^i$ 是用户 $i$ 在 $t$ 时刻的兴趣状态（即 GRU 的隐状态）
- $e_{b[t+1]}^i$ 是用户 $i$ 在 $t+1$ 时刻真实点击的物品 Embedding（正样本）
- $\hat{e}_{b[t+1]}^i$ 是从物品池中负采样得到的物品 Embedding（负样本）
- $\sigma(\cdot)$ 是 Sigmoid 函数，这里用于计算两个向量的点积并映射到 $(0, 1)$ 区间

这个辅助损失会与模型最终的 CTR 预测损失 $L_{target}$ 加在一起共同优化：$L = L_{target} + \alpha L_{aux}$。这个额外的监督信号，在每个时间步都对 GRU 的学习进行指导，使其产出的隐状态 $h_t$ 能够更精准地表达用户的潜在兴趣

**核心代码**

辅助损失的计算通过预测下一个行为来监督兴趣状态的学习：

```python
# 兴趣提取层的辅助损失计算
# interest_states: [batch_size, seq_len, hidden_units]
# pos_behaviors: [batch_size, seq_len, embedding_dim] 正样本行为
# neg_behaviors: [batch_size, seq_len, embedding_dim] 负样本行为

# 用t时刻的兴趣预测t+1时刻的行为
current_interests = interest_states[:, :-1, :]      # [B, T-1, H]
next_pos_behaviors = pos_behaviors[:, 1:, :]        # [B, T-1, D]
next_neg_behaviors = neg_behaviors[:, 1:, :]        # [B, T-1, D]

# 拼接兴趣和行为，送入MLP预测
pos_input = tf.concat([current_interests, next_pos_behaviors], axis=-1)
neg_input = tf.concat([current_interests, next_neg_behaviors], axis=-1)

# 预测正负样本的概率
pos_probs = auxiliary_mlp(pos_input)  # [B, T-1, 1]
neg_probs = auxiliary_mlp(neg_input)  # [B, T-1, 1]

# 二元交叉熵损失
aux_loss = -tf.reduce_mean(
    tf.math.log(pos_probs + 1e-8) + tf.math.log(1 - neg_probs + 1e-8)
)
```

这种设计确保 GRU 的隐状态不仅能记录历史信息，还能有效预测未来行为，从而学到更有意义的兴趣表示。

**第二阶段：兴趣演化层 (Interest Evolving Layer)**

经过第一阶段，我们得到了一个更能代表用户内在兴趣的兴趣状态序列 $h_1,h_2,\ldots,h_T$。第二阶段的目标，就是对

这个兴趣序列的演化过程进行建模

然而，兴趣的演化并不总是平滑的，常常会伴随着兴趣漂移 (Interest Drifting) 现象，即用户可能在不同的兴趣点之间快速切换。如果用一个标准的 GRU 来建模这个兴趣序列，不相关的历史兴趣 (漂移) 可能会干扰对当前主要兴趣演化的判断

为了解决这个问题，DIEN 再次借鉴了 DIN 的思想，并将其与序列模型融合，设计了带注意力更新门的 GRU

(AUGRU) 。AUGRU 的核心是在 GRU 的更新门 (Update Gate) 上融入了注意力机制。注意力得分 $a_{t}$ 由 $t$ 时刻的兴趣

状态 $\boldsymbol{h}_t$ 和候选广告 $e_a$ 共同决定：

$$a_t=\frac{\exp(\boldsymbol{h}_tW\boldsymbol{e}_a)}{\sum_{j=1}^T\exp(\boldsymbol{h}_jW\boldsymbol{e}_a)}$$

然后，这个注意力得分 $a_t$ 会去调整 (scale) GRU 的原始更新门 $u_t^{\prime}{:}$

$$\tilde{\boldsymbol{u}}_t^{\prime}=a_t\cdot\boldsymbol{u}_t^{\prime}$$

最后，使用这个被注意力调整过的更新门 $\tilde{\boldsymbol{u}}_t^{\prime}$ 来更新隐状态：

$$\boldsymbol{h}_t^{\prime}=(1-\tilde{\boldsymbol{u}}_t^{\prime})\circ\boldsymbol{h}_{t-1}^{\prime}+\tilde{\boldsymbol{u}}_t^{\prime}\circ\tilde{\boldsymbol{h}}_t^{\prime}$$

其中 o 表示元素级乘积 (element-wise product) 

通过这种方式，AUGRU 在兴趣演化的每一步，都会参考当前的候选广告，来判断历史兴趣的相关性。与候选广告越相关的兴趣，其对应的越大，其信息在更新门中的权重也越大，从而能更顺畅地在序列中传递；反之，不相关的兴趣（漂移）其影响力就会被削弱。这使得模型能够聚焦于与当前推荐任务最相关的兴趣演化路径

**核心代码**

AUGRU 的核心在于用注意力分数调整 GRU 的更新门：

```python
# AUGRU的前向传播
# interest_states: [batch_size, seq_len, hidden_units]
# target_item_embedding: [batch_size, embedding_dim]

# 1. 计算双线性注意力分数
# h_t * W * e_a
h_W = tf.tensordot(interest_states, bilinear_weight, axes=[[2], [0]])
target_expanded = tf.expand_dims(target_item_embedding, axis=1)
attention_scores = tf.reduce_sum(h_W * target_expanded, axis=2)  # [B, T]
attention_scores = tf.nn.softmax(attention_scores, axis=1)  # [B, T]

# 2. 逐步处理序列
hidden_state = tf.zeros([batch_size, hidden_units])
for t in range(seq_len):
    current_input = interest_states[:, t, :]     # [B, H]
    current_attention = attention_scores[:, t]   # [B]

    # 标准GRU计算
    update_gate = tf.nn.sigmoid(
        dense_input_update(current_input) + dense_hidden_update(hidden_state)
    )  # [B, H]

    reset_gate = tf.nn.sigmoid(
        dense_input_reset(current_input) + dense_hidden_reset(hidden_state)
    )  # [B, H]

    candidate_state = tf.nn.tanh(
        dense_input_candidate(current_input) +
        dense_hidden_candidate(reset_gate * hidden_state)
    )  # [B, H]

    # 关键：用注意力分数缩放更新门
    attention_expanded = tf.expand_dims(current_attention, axis=1)
    attention_expanded = tf.tile(attention_expanded, [1, hidden_units])
    attention_update_gate = attention_expanded * update_gate  # [B, H]

    # 更新隐藏状态
    hidden_state = (1 - attention_update_gate) * hidden_state + \
                   attention_update_gate * candidate_state
```

AUGRU 通过注意力分数动态调整更新门，使得与目标广告相关的兴趣能够顺利传递，而不相关的兴趣（漂移）被抑制，从而更精准地捕捉兴趣演化路径

### 3.3 从行为序列到会话序列

从 DIN 到 DIEN，我们看到了模型对用户兴趣的理解从“静态相关”走向了“动态演化”。然而，它们都将用户的行为看作一条连续的序列。但现实中，用户的行为模式更多是间断性的。用户通常在**一个会话（Session）** 内拥有一个明确且集中的意图，而在**不同会话**之间，兴趣点可能发生巨大转变

![[public/Pasted image 20260114191533.png|325]]

如上图所示，一个用户可能在一个会话里集中浏览各种裤子，而在下一个会话则专注于戒指。这种**会话内同质、会话间异质**的现象非常普遍。如果直接用一个 RNN 模型处理这种“断层”明显的长序列，模型需要花费很大力气去学习这种兴趣的突变，效果并不理想。深度会话兴趣网络（Deep Session Interest Network, DSIN） ([Feng _et al._, 2019](https://datawhalechina.github.io/fun-rec/chapter_references/references.html#id54 "Feng, Y., Lv, F., Shen, W., Wang, M., Sun, F., Zhu, Y., & Yang, K. (2019). Deep session interest network for click-through rate prediction. arXiv preprint arXiv: 1905.06482.")) 将“会话”作为分析用户行为的基本单元，并采用一种**分层**的思想来建模

![[public/Pasted image 20260114191554.png|425]]

---

## 04 多目标建模

多目标建模（Multi-Task Learning, MTL）通过联合优化多个相关任务，在推荐系统中实现用户体验与商业目标的协同提升。相比独立建模，多目标方法能够降低参数量、提升系统效率，并通过知识迁移缓解数据稀疏问题

在实际应用中，电商场景联合优化 CTR、CVR、GMV 避免单一指标导致的低质商品推荐；视频平台同时优化播放完成率、评分预测、用户留存率提升长期用户价值。然而，多目标建模面临任务冲突、跷跷板效应和负迁移等核心挑战

- CVR 相对 CTR 更进一步，指向购买、收藏等具备某些效益的行为

针对这些挑战，业界发展出三大解决方向：

1. 模型架构从 Shared-Bottom 到 MMoE 再到 PLE 的演进，解决任务冲突与负迁移
2. ESMM 和 ESM2 等依赖关系建模方法，处理用户行为链路的样本偏差
3. 以及从手工加权到自适应优化的多损失融合策略，解决量级失衡与收敛异步问题

本章将详细介绍这些核心技术的原理与实践

### 4.1 基础结构演进

#### Shared-Bottom

Shared-Bottom 模型作为多目标建模的奠基性架构, 采用"共享地基+独立塔楼"的设计范式。其核心结构包含两个关键组件:

- **共享底层**(Shared Bottom): 所有任务共用同一组特征转换层, 负责学习跨任务的通用特征表示
- **任务特定塔**(Task-Specific Towers): 每个任务拥有独立的顶层网络, 基于共享表示学习任务特定决策边界

![[public/Pasted image 20260217180813.png|475]]

这种架构的数学表达可描述为:

$$\hat{y}_t = f_t(W_t \cdot g(W_s \mathbf{x}))$$

其中 $\mathbf{W}_s$ 为共享层参数, $g(\cdot)$ 为共享特征提取函数, $f_t(\cdot)$ 为任务 $t$ 的预测函数。其设计哲学建立在任务同质性假设上: 不同任务共享相同的底层特征空间, 仅需在顶层进行任务适配

Shared-Bottom 模型在效率与泛化之间实现了良好的平衡, 其核心优势主要体现在以下几点

1. 在参数效率方面, 共享层占据了模型大部分的参数量这显著降低了模型的总参数量
2. 共享层具有正则化效应, 它如同一个天然的正则化器, 通过强制任务共用特征表示, 有效防止了单个任务出现过拟合的情况
   Regularization：模型在当前已知的训练数据上"过拟合" 👉 确保在未见过的数据上表现稳定
3. 在知识迁移方面, 当任务之间存在潜在的相关性时, 例如视频的点击率与完播率, 共享层能够学习到通用的模式, 从而提升小样本任务的泛化能力

然而, Shared-Bottom 模型也存在一个致命缺陷, 即**负迁移现象**。当任务之间存在本质上的冲突时, 该模型的硬共享机制会引发负迁移问题。从机制本质上来看, 共享层的梯度更新方向是由所有任务共同决定的, 一旦任务目标之间出现冲突, 参数优化就会陷入方向性的矛盾

在一些典型场景中, 例如电商平台同时优化"点击率"与"客单价"时, 低价商品可能会推动点击率的提升, 但同时却抑制了客单价的增长; 又如内容平台在平衡"内容消费深度"与"广告曝光量"时, 深度阅读行为往往与广告点击行为呈负相关

从数学角度来解释, 假设任务 $i$ 与任务 $j$ 的损失梯度分别为 $\nabla L_i$ 与 $\nabla L_j$, 当 $\nabla L_i \cdot \nabla L_j < 0$ 时, 共享层参数更新就会产生内在的冲突。这种冲突使得模型在处理矛盾任务时呈现出"零和博弈"的特性, 即提升某一目标的性能往往需要以牺牲另一目标为代价, 我们一般也称这类问题为**跷跷板问题**

**Shared-Bottom 代码实践**

shared-bottom 模型构建代码如下, 先组装输入到 shared-bottom 网络中的特征 dnn_inputs, 经过一个 shared-bottom DNN 网络, 遍历创建各个任务独立的 DNN 塔, 最后输出多个塔的预估值用于计算 Loss

```python
def build_shared_bottom_model(
        feature_columns,
        task_name_list,
        share_dnn_units=[128, 64],
        task_tower_dnn_units=[128, 64],
        ):
    # 输入层:将原始特征映射为 Keras 输入
    input_layer_dict = build_input_layer(feature_columns)

    # 嵌入层:为各特征组创建嵌入表,得到组内嵌入向量
    embedding_table_dict = build_group_feature_embedding_table_dict(
        feature_columns, input_layer_dict, prefix="embedding/"
    )

    # 合并嵌入:将多组嵌入拼接为共享 DNN 的输入
    dnn_inputs = concat_group_embedding(embedding_table_dict, 'dnn')

    # 共享底座:所有任务共享的特征抽取网络(Shared Bottom)
    shared_feature = DNNs(share_dnn_units)(dnn_inputs)

    # 任务塔:在共享特征上为每个任务构建独立塔并输出概率
    task_outputs = []
    for task_name in task_name_list:
        task_logit = DNNs(task_tower_dnn_units + [1])(shared_feature)
        task_prob = PredictLayer(name=f"task_{task_name}")(task_logit)
        task_outputs.append(task_prob)

    # 构建多任务模型:共享底座 + 多任务塔输出
    model = tf.keras.Model(inputs=list(input_layer_dict.values()), outputs=task_outputs)
    return model
```

**训练和评估**

```python
from funrec import run_experiment

run_experiment('shared_bottom')
```

```
+----------------+---------------+-----------------+-------------+-----------------+----------------+------------------+--------------+---------------------+--------------------+----------------------+
|   auc_is_click |   auc_is_like |   auc_long_view |   auc_macro |   gauc_is_click |   gauc_is_like |   gauc_long_view |   gauc_macro |   val_user_is_click |   val_user_is_like |   val_user_long_view |
+================+===============+=================+=============+=================+================+==================+==============+=====================+====================+======================+
|           0.61 |         0.431 |          0.4364 |      0.4925 |          0.5727 |         0.4433 |           0.4438 |       0.4866 |                 928 |                530 |                  925 |
+----------------+---------------+-----------------+-------------+-----------------+----------------+------------------+--------------+---------------------+--------------------+----------------------+
```

#### MMoE

![[public/Pasted image 20260217181630.png]]

针对 Shared-Bottom 对相关性低的多个任务出现负迁移的现象, OMoE (OneGate Mixture-of-Experts) 将底层共享的一个 Shared-Bottom 模块拆分成了多个 Expert, 最终 OMoE 的输出为多个 Expert 的加权和, 本质可以看成是专家网络和全局门控的双层结构

虽然 OMoE 通过底层多专家融合的方式提升了特征表征的多样性, 从最终的实验结果看, 确实可以一定程度上缓解低相关性任务的负迁移问题, 但没有彻底解决多任务冲突的问题。因为不同任务反向传播的梯度还是会直接影响底层专家网络的学习

为了进一步缓解多任务冲突, MMoE (Multi-gate Mixture-of-Experts) 为每个任务配备专属门控网络, 实现了门控从"全局共享"升级为"任务自适应"的方式。MMoE 的数学表达式可以表示为:

$$
\begin{aligned} 
\mathbf{e}_k &= f_k(\mathbf{x}) \\ 
g_t(\mathbf{x}) &= \text{softmax}(\mathbf{W}_t \mathbf{x}) \\ 
\mathbf{h}_t &= \sum_{k=1}^K g_{t,k} \cdot \mathbf{e}_k \\ 
\hat{y}_t &= f_t(\mathbf{h}_t)
\end{aligned}
$$

其中 $\mathbf{x}$ 表示底层的特征输入, $\mathbf{e}_k$ 表示第 k 个专家网络的输出, $g_t(\mathbf{x})$ 表示第 $t$ 个任务融合专家网络的门控向量, $\mathbf{h}_t$ 表示第 $t$ 个任务融合专家网络的输出, $\hat{y}_t$ 表示第 $t$ 个任务的预测结果

差异化特征融合门控网络 $g_t$ 根据任务特性选择专家组合, 例如在电商场景, CTR 任务门控加权"即时兴趣""价格敏感"专家, CVR 任务门控: 侧重"消费能力""品牌忠诚"专家

当任务 $i$ 与 $j$ 冲突时, MMoE 的门控机制会让两个任务学习到不同专家的权重分布。例如, 某个专家 $e_m$ 可能在任务 $i$ 的门控网络中获得很高的权重 $g_{i,m}$, 而在任务 $j$ 的门控网络中获得很低的权重 $g_{j,m}$。这样一来, 专家 $e_m$ 的参数更新主要由任务 $i$ 的梯度决定, 而任务 $j$ 的梯度影响很小, 从而实现了梯度隔离。不同任务通过选择不同的专家组合, 可以各自学习到适合自己的特征表示, 缓解任务冲突

**MMoE 代码实践**

MMoE 模型构建代码如下, 先组装输入到 MoE 网络中的特征 dnn_inputs, 然后为每个任务创建一个门控网络输出最终融合专家网络的门控向量。最后为每个任务都创建一个任务塔, 并且不同任务塔的输入都是对应任务的门控向量和多个专家网络融合后的向量

```python
def build_mmoe_model(
        feature_columns,
        task_name_list,
        expert_nums=4,
        expert_dnn_units=[128, 64],
        gate_dnn_units=[128, 64],
        task_tower_dnn_units=[128, 64],
        ):
    # 输入层:原始特征 → Keras 输入
    input_layer_dict = build_input_layer(feature_columns)

    # 嵌入层:为各特征组创建嵌入表,得到组内嵌入向量
    embedding_table_dict = build_group_feature_embedding_table_dict(
        feature_columns, input_layer_dict, prefix="embedding/"
    )

    # 合并嵌入:拼接为专家与门控的共同输入
    dnn_inputs = concat_group_embedding(embedding_table_dict, 'dnn')

    # 共享专家:多个并行 DNN(专家)供所有任务共享
    expert_outputs = [DNNs(expert_dnn_units, name=f"expert_{i}")(dnn_inputs)
                      for i in range(expert_nums)]
    # 形状 [B, E, D]:按专家维度堆叠,便于后续加权求和
    experts = tf.keras.layers.Lambda(lambda xs: tf.stack(xs, axis=1))(expert_outputs)

    # 任务门控:每个任务产生 softmax 权重,对专家加权融合
    task_features = []
    for idx, task_name in enumerate(task_name_list):
        gate_hidden = DNNs(gate_dnn_units, name=f"task_{idx}_gate_mlp")(dnn_inputs)
        gate_weights = tf.keras.layers.Dense(expert_nums, use_bias=False,
                                             activation='softmax',
                                             name=f"task_{idx}_gate_softmax")(gate_hidden)  # [B, E]
        # 加权融合:einsum('be,bed->bd') == sum_e w_e * expert_e
        task_mix = tf.keras.layers.Lambda(lambda x: tf.einsum('be,bed->bd', x[0], x[1]))([gate_weights, experts])
        task_features.append(task_mix)

    # 任务塔:基于融合特征为每个任务建塔并输出概率
    task_outputs = []
    for task_name, task_feat in zip(task_name_list, task_features):
        task_logit = DNNs(task_tower_dnn_units + [1])(task_feat)
        task_prob = PredictLayer(name=f"task_{task_name}")(task_logit)
        task_outputs.append(task_prob)

    # 构建模型:共享专家 + 任务门控 + 任务塔
    model = tf.keras.Model(inputs=list(input_layer_dict.values()), outputs=task_outputs)
    return model
```

**训练和评估**

```python
run_experiment('mmoe')
```

```
+----------------+---------------+-----------------+-------------+-----------------+----------------+------------------+--------------+---------------------+--------------------+----------------------+
|   auc_is_click |   auc_is_like |   auc_long_view |   auc_macro |   gauc_is_click |   gauc_is_like |   gauc_long_view |   gauc_macro |   val_user_is_click |   val_user_is_like |   val_user_long_view |
+================+===============+=================+=============+=================+================+==================+==============+=====================+====================+======================+
|         0.6036 |        0.4203 |          0.4586 |      0.4942 |          0.5728 |         0.4362 |           0.4759 |        0.495 |                 928 |                530 |                  925 |
+----------------+---------------+-----------------+-------------+-----------------+----------------+------------------+--------------+---------------------+--------------------+----------------------+
```

#### PLE

MMoE 通过为每个任务配备专属门控网络, 在一定程度上缓解了多任务冲突问题。专属门控网络能够根据任务特性选择专家组合, 从而使得不同任务可以关注不同的特征表示。但其架构仍存在一个根本性局限:**所有专家对所有任务门控可见**。这种"软隔离"设计在实践中仍面临两大挑战:

##### 负迁移未根除

- **干扰路径未切断**: 即使某个专家 (如 $e_m$) 被任务 $i$ 的门控高度加权而被任务 $j$ 的门控忽略, 任务 $j$ 的梯度在反向传播时仍会流经 $e_m$ (因为 $e_m$ 是任务 $j$ 门控的可选项)。当任务冲突强烈时, 这种"潜在通路"仍可能导致共享表征被污染
- **专家角色模糊**: MMoE 缺乏机制强制专家明确分工。一个专家可能同时承载共享信息和多个任务的特定信息, 成为冲突的"重灾区"。尤其在任务相关性低时, 这种耦合会加剧负迁移

##### 门控决策负担重

- 每个任务的门控需要在所有 $K$ 个专家上进行权重分配。当专家数量增加 (通常需扩大 $K$ 以提升模型能力) 时, 门控网络面临高维决策问题, 易导致训练不稳定或陷入次优解
- 门控需要"费力"地从包含混杂信息 (共享+所有任务特定) 的专家池中筛选有用信息, 增加了学习难度

为解决上述问题, 提出了**CGC (Customized Gate Control)** 结构, 其核心思想是通过硬性结构约束, 显式分离共享知识与任务特定知识:

![[public/Pasted image 20260217182612.png|400]]

**专家职责强制分离**:

- **共享专家 (C-Experts)**: 一组专家仅负责学习所有任务的共性知识。设其数量为 $M$, 输出为 $\{\mathbf{c}_1, \mathbf{c}_2, ..., \mathbf{c}_M\}$。

- **任务专家 (T-Experts)**: 每个任务 $t$ 拥有自己专属的专家组, 仅负责学习该任务特有的知识或模式。设任务 $t$ 的专属专家数量为 $N_t$, 输出为 $\{\mathbf{t}_t^1, \mathbf{t}_t^2, ..., \mathbf{t}_t^{N_t}\}$。

**任务专属门控的输入限制**:

- 任务 $t$ 的门控 $g_t$ 的输入被严格限制为: 共享专家输出 ($\{\mathbf{c}_k\}_{k=1}^M$) + 本任务专属专家输出 ($\{\mathbf{t}_t^j\}_{j=1}^{N_t}$)。

- **物理切断干扰路径**: 任务 $t$ 的门控完全无法访问其他任务 $s (s \neq t)$ 的专属专家 $\{\mathbf{t}_s^j\}$。同样, 任务 $s$ 的梯度绝不会更新任务 $t$ 的专属专家参数。

CGC 门控的计算如下:

$$g_t(\mathbf{x}) = \text{softmax}\Big(\mathbf{W}_t \cdot \mathbf{x} + \mathbf{b}_t\Big)$$

$$\mathbf{h}_t = \sum_{k=1}^{M} g_{t,k} \cdot \mathbf{c}_k + \sum_{j=1}^{N_t} g_{t, M+j} \cdot \mathbf{t}_t^j$$

$$\hat{y}_t = f_t(\mathbf{h}_t)$$

其中:

- $\mathbf{W}_t, \mathbf{b}_t$: 任务 $t$ 门控的参数
- $g_{t, k}$: 分配给第 $k$ 个共享专家的权重
- $g_{t, M+j}$: 分配给任务 $t$ 第 $j$ 个专属专家的权重

**CGC 核心代码为:**

```python
def cgc_net(
        input_list,
        task_num,
        task_expert_num,
        shared_expert_num,
        task_expert_dnn_units,
        shared_expert_dnn_units,
        task_gate_dnn_units,
        shared_gate_dnn_units,
        leval_name=None,
        is_last=False):
    """CGC(共享专家 + 任务门控)核心结构(简化版)
    - 每个任务:拥有若干 Task-Experts;
    - 全局:拥有若干 Shared-Experts;
    - 每个任务 Gate 产生 softmax 权重,对其 Task-Experts 与 Shared-Experts 加权融合;
    - 若非最后一层:再用 Shared-Gate 融合所有任务的 Task-Experts 与 Shared-Experts,供下一层共享使用。
    input_list:为方便处理,给每个任务复制一份相同输入,最后一个为共享输入。
    """

    # 任务专家:每个任务创建 task_expert_num 个专家
    task_expert_list = []
    for i in range(task_num):
        task_expert_list.append([
            DNNs(task_expert_dnn_units, name=f"{leval_name}_task_{i}_expert_{j}")(input_list[i])
            for j in range(task_expert_num)
        ])

    # 共享专家:创建 shared_expert_num 个专家(共享输入使用 input_list[-1])
    shared_expert_list = [
        DNNs(shared_expert_dnn_units, name=f"{leval_name}_shared_expert_{i}")(input_list[-1])
        for i in range(shared_expert_num)
    ]

    # 任务门控与融合:对当前任务的(Task + Shared)专家集合进行 softmax 加权求和
    cgc_outputs = []
    fusion_expert_num = task_expert_num + shared_expert_num
    for i in range(task_num):
        cur_experts = task_expert_list[i] + shared_expert_list
        experts = tf.keras.layers.Lambda(lambda xs: tf.stack(xs, axis=1))(cur_experts)  # [B, E, D]

        gate_hidden = DNNs(task_gate_dnn_units, name=f"{leval_name}_task_{i}_gate")(input_list[i])
        gate_weights = tf.keras.layers.Dense(fusion_expert_num, use_bias=False, activation='softmax')(gate_hidden)  # [B, E]

        # 加权融合:einsum('be,bed->bd') == sum_e w_e * expert_e
        fused = tf.keras.layers.Lambda(lambda x: tf.einsum('be,bed->bd', x[0], x[1]))([gate_weights, experts])
        cgc_outputs.append(fused)

    # 若非最后一层:共享门控融合所有任务专家与共享专家,作为下一层共享输入
    if not is_last:
        # 展平所有任务的专家 + 共享专家
        all_task_experts = [e for task in task_expert_list for e in task]
        cur_experts = all_task_experts + shared_expert_list
        experts_all = tf.keras.layers.Lambda(lambda xs: tf.stack(xs, axis=1))(cur_experts)  # [B, E_all, D]
        cur_expert_num = len(cur_experts)

        shared_gate_hidden = DNNs(shared_gate_dnn_units, name=f"{leval_name}_shared_gate")(input_list[-1])
        shared_gate_weights = tf.keras.layers.Dense(cur_expert_num, use_bias=False, activation='softmax')(shared_gate_hidden)  # [B, E_all]
        shared_fused = tf.keras.layers.Lambda(lambda x: tf.einsum('be,bed->bd', x[0], x[1]))([shared_gate_weights, experts_all])
        cgc_outputs.append(shared_fused)

    return cgc_outputs
```

CGC 解决了知识分离的核心问题, 但其本质是单层结构, 表征学习深度有限。受深度神经网络逐层抽象特征的启发,**PLE (Progressive Layered Extraction)** 将多个 CGC 单元纵向堆叠, 形成深层架构, 实现渐进式知识提取与融合

- **第 1 层 (输入层 CGC)**:
  - 输入: 原始特征 $\mathbf{x}$
  - 结构: 一个标准的 CGC 模块 (包含 $M^{(1)}$ 个 C-Experts, 每个任务 $t$ 有 $N_t^{(1)}$ 个 T-Experts, 以及对应的门控 $g_t^{(1)}$)
  - 输出: 每个任务获得初步融合表示 $\mathbf{h}_t^{(1)}$ (或更常见的是, 该层所有专家 (C+T) 的输出被拼接/收集起来作为下一层输入)
- **第 $l$ 层** ($l \geq 2$) **CGC**:
  - **输入关键点**: 第 $l$ 层的输入是第 $l-1$ 层所有专家 (包括所有 C-Experts 和所有任务的 T-Experts) 的输出。设第 $l-1$ 层总专家数为 $E^{(l-1)}$, 则输入向量维度相应增加
  - 结构: 一个新的 CGC 模块 (包含 $M^{(l)}$ 个 C-Experts, 每个任务 $t$ 有 $N_t^{(l)}$ 个 T-Experts, 以及门控 $g_t^{(l)}$)
  - 处理: 在本层输入特征 (即上一层更丰富的表征) 上, 再次进行显式的知识分离 (新的 C-Experts 学习更深层共享模式, 新的 T-Experts 学习更深层任务特定模式) 和融合 (通过新的门控)
  - 输出: 任务 $t$ 的当前层表示 $\mathbf{h}_t^{(l)}$ 或收集所有专家输出
- **输出层**(第 $L$ 层):
  - 最后一层 ($L$) 各任务的门控输出 $\mathbf{h}_t^{(L)}$ 送入各自的任务专属塔网络 (Tower) $f_t$, 得到最终预测 $\hat{y}_t = f_t (\mathbf{h}_t^{(L)})$

**PLE 模型的核心代码:**

```python
def build_ple_model(
        feature_columns,
        task_name_list,
        ple_level_nums=1,
        task_expert_num=4,
        shared_expert_num=2,
        task_expert_dnn_units=[128, 64],
        shared_expert_dnn_units=[128, 64],
        task_gate_dnn_units=[128, 64],
        shared_gate_dnn_units=[128, 64],
        task_tower_dnn_units=[128, 64],
        ):
    # 1) 输入与嵌入:构建输入层/分组嵌入,拼接为 PLE 的共享输入
    input_layer_dict = build_input_layer(feature_columns)
    group_embedding_feature_dict = build_group_feature_embedding_table_dict(
        feature_columns, input_layer_dict, prefix="embedding/"
    )
    dnn_inputs = concat_group_embedding(group_embedding_feature_dict, 'dnn')

    # 2) 级联 PLE(CGC)层:每层包含"任务专家 + 共享专家 + 门控",最后一层仅输出任务特征
    task_num = len(task_name_list)
    ple_input_list = [dnn_inputs] * (task_num + 1)  # 前 task_num 为各任务输入,末尾为共享输入
    for i in range(ple_level_nums):
        is_last = (i == ple_level_nums - 1)
        ple_input_list = cgc_net(
            ple_input_list,
            task_num,
            task_expert_num,
            shared_expert_num,
            task_expert_dnn_units,
            shared_expert_dnn_units,
            task_gate_dnn_units,
            shared_gate_dnn_units,
            leval_name=f"cgc_level_{i}",
            is_last=is_last
        )

    # 3) 任务塔与输出:将各任务特征送入塔 DNN,得到每个任务的概率输出
    task_output_list = []
    for i in range(task_num):
        task_logit = DNNs(task_tower_dnn_units + [1])(ple_input_list[i])
        task_prob = PredictLayer(name="task_" + task_name_list[i])(task_logit)
        task_output_list.append(task_prob)

    # 4) 构建模型:输入为所有原始输入层,输出为各任务概率
    model = tf.keras.Model(inputs=list(input_layer_dict.values()), outputs=task_output_list)
    return model
```

**训练和评估**

```python
run_experiment('ple')
```

```
+----------------+---------------+-----------------+-------------+-----------------+----------------+------------------+--------------+---------------------+--------------------+----------------------+
|   auc_is_click |   auc_is_like |   auc_long_view |   auc_macro |   gauc_is_click |   gauc_is_like |   gauc_long_view |   gauc_macro |   val_user_is_click |   val_user_is_like |   val_user_long_view |
+================+===============+=================+=============+=================+================+==================+==============+=====================+====================+======================+
|         0.5975 |        0.4191 |          0.4474 |       0.488 |           0.574 |         0.4434 |           0.4561 |       0.4912 |                 928 |                530 |                  925 |
+----------------+---------------+-----------------+-------------+-----------------+----------------+------------------+--------------+---------------------+--------------------+----------------------+
```

### 4.2 任务依赖建模

### 4.3 多目标损失融合

---

## 05 多场景建模

### 5.1 多塔结构

### 5.2 动态权重建模