#LLM4Rec 

概念辨析：

- 流行度偏差 Popularity Bias：倾向于推荐那些**大众熟知、热度很高**的物品，而不是真正符合用户独特个性的小众物品 👈 预训练数据中热门物品频率高
- 对地理区域的敏感性 Sensitivity to Geographical Regions：特定区域和文化的偏斜 👈 训练数据不平衡
- 数据稀疏：User 仅与海量 Item 中的少部分进行过 Interaction
- 跨域推荐：迁移学习 👉 目标域的 Data 不足，通过源域的 Data，学习并迁移至目标域的 Data

![[public/Pasted image 20260121132434.png|500]]

## Framework

### Feature-Based Paradigm

#### FM Tokens

这一范式的核心任务是基于物品和用户的特征生成令牌（Tokens），而不是依赖任意分配的数字编号。这种建模范式基于输入物品和用户的特征生成令牌这些生成的令牌旨在通过语义挖掘来捕捉潜在的用户偏好，从而可以将其**直接整合到推荐系统的决策过程**中

- Token 和 Embedding 的差异？
  离散 Token 本身可以直接整合到传统的推荐系统中

传统推荐系统主要面临的挑战是它们依赖于毫无语义的数字 ID（例如 "item 57"）来表示用户和物品。这种 ID 缺乏语义信息，无法利用基础模型（FMs）中蕴含的丰富知识。此外，由于这些 ID 没有内在含义，模型通常需要大量的交互数据才能微调出有效的表示，这限制了其在冷启动、跨域推荐等场景中的泛化能力。这导致了基础模型的语言能力与推荐系统所需的协同信号之间存在“语义鸿沟”。LC-Rec 模型致力于解决大型语言模型与推荐系统之间的这一语义鸿沟

主要的洞察是利用 **语义 ID (Semantic IDs)** 来弥合基础模型与推荐系统之间的差距。语义 ID 是从物品的实际内容（如标题、描述）中衍生出的标识符，而非随机分配的数字

通过这种方式，推荐任务被重构为 **生成式检索 (Generative Retrieval)** 问题。在这个框架下，模型不再仅仅是对固定的候选集进行排序，而是以自回归的方式解码出目标物品的标识符，从而使其能够直接从整个物品集中生成物品，而不依赖预定义的候选集

这一类别的方法通常采用 **语义 ID (Semantic IDs)** 或 **矢量量化 (Vector Quantization)** 来创建基础模型可以处理的有意义令牌：

*   **生成式检索 (如 TIGER):** 该框架提出了一种生成式检索方法，自回归地解码物品标识符。在该框架中，每个物品被分配一个“语义 ID”——这是一组源自物品内容特征的码字（codewords）。 <alphaxiv-paper-citation title="语义 ID" page="9" first="framework proposes a" last="item identifiers." />
*   **矢量量化 (如 LC-Rec):** 该方法使用基于学习的矢量量化方法来分配有意义的物品索引，通过整合语言和协同语义来解决语义鸿沟问题。 <alphaxiv-paper-citation title="矢量量化" page="9" first="employs a learning-based" last="item indices," />
*   **统一生成框架 (如 ColaRec, EAGER):** ColaRec 等框架在一个统一的序列到序列（Seq2Seq）生成框架中结合了内容信息和协同信号。 <alphaxiv-paper-citation title="统一框架" page="9" first="content information and" last="generative framework," />
*   **级联方法 (如 COBRA):** COBRA 框架采用级联方法，在稀疏的语义 ID 和密集的向量之间交替，以同时捕捉语义洞察和来自用户-物品交互的协同信号。 <alphaxiv-paper-citation title="级联方法" page="9" first="framework adopts a" last="collaborative signals" />

| 论文                                                                                                                | 描述                                 |
| :---------------------------------------------------------------------------------------------------------------- | :--------------------------------- |
| [TIGER: Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)                          | 使用“语义 ID”（源自内容的码字）来实现物品推荐的自回归解码    |
| [LC-Rec: Adapting Large Language Models by Integrating Collaborative Semantics](https://arxiv.org/abs/2311.09049) | 使用矢量量化分配有意义的索引，弥合 LLM 和推荐系统之间的语义鸿沟 |
| [ColaRec: Content-Based Collaborative Generation for Recommender Systems](https://arxiv.org/abs/2403.18480)       | 在 Seq2Seq 框架中统一了内容知识和协同信号          |
| [EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration](https://arxiv.org/abs/2406.14017) | 通过双流生成架构整合行为信息和语义信息                |
| [COBRA: Sparse Meets Dense: Unified Generative Recommendations](https://arxiv.org/abs/2503.02453)                 | 在稀疏语义 ID 和密集向量之间交替，以捕捉语义和交互信号      |

#### FM Embeddings

这一范式的核心任务是将基础模型（FMs）视为**特征提取器（Feature Extractor）**

具体而言，它将物品和用户的特征（如文本描述、元数据）输入到大型语言模型（LLMs）中，并利用 LLM 输出的高质量嵌入向量（Embeddings）作为推荐模型的输入特征。此建模范式将语言模型视为特征提取器，将物品和用户的特征输入到 LLM 中并输出相应的嵌入

传统的推荐系统主要面临两大挑战：

1.  **隐式反馈稀疏性**：用户与物品的交互（点击、购买）数据往往非常稀疏，难以捕捉准确的用户偏好。这项工作解决了隐式反馈信号稀疏的问题
2.  **缺乏语义深度**：传统的 ID 嵌入（如 One-hot）或浅层语义嵌入无法充分理解内容中的复杂语义，也难以利用物品之间潜在的关联知识

使用 FMs 生成嵌入的主要动机在于利用基础模型强大的**文本理解和推理能力**。

*   **语义理解**：FMs 可以捕捉用户行为和偏好中细微的语义特征，而不仅仅是基于 ID 的共现关系。利用 LLM 先进的文本理解能力来捕捉用户行为和偏好的细微语义方面
*   **显式推理**：LLM 能够显式地推理用户与物品之间的交互模式，从而增强对协同信息的建模。通过让 LLM 显式推理用户-物品交互模式来解决隐式反馈信号稀疏的问题 
*   **知识增强**：FMs 拥有广泛的世界知识，可以为物品生成更丰富的描述或补充缺失的元数据

该范式通常通过将 FMs 的嵌入与传统推荐模型（如协同过滤、图神经网络）相结合。典型的方法包括：

| 方法                                                       | 描述                                                        |
| :------------------------------------------------------- | :-------------------------------------------------------- |
| **RLMRec** ([arXiv](https://arxiv.org/abs/2310.15950))   | 将 LLM 视为文本编码器，将物品/用户映射到语义空间，并将语义空间与协同关系建模对齐               |
| **AlphaRec** ([arXiv](https://arxiv.org/abs/2407.05441)) | 采用线性映射将物品标题的语言表示投影到推荐行为空间中                                |
| **LLMRec** ([arXiv](https://arxiv.org/abs/2311.00423))   | 利用 LLM 进行图增强（Graph Augmentation），包括增强用户-物品交互边、物品节点属性和用户画像 |
| **BinLLM** ([arXiv](https://arxiv.org/abs/2311.09049))   | 将协同嵌入转换为二进制序列（文本格式），使 LLM 能够直接以文本形式利用协同信息                 |
| **iDreamRec**                                            | 将 LLM 嵌入与扩散模型（Diffusion Models）结合，利用 GPT 生成详细的文本描述以模拟物品分布 |

### Generative Paradigm

#### Fine-tuning FM4RS

这一范式的核心任务是**将通用的大型基础模型（FMs）适配为特定领域的推荐模型**

具体而言，它不仅利用 FMs 作为特征提取器或零样本推理机，而是使用特定领域的推荐数据（如用户交互历史、点击日志、评分数据）对模型的参数进行**微调（Fine-tuning）**，使其学习推荐任务的特定模式。该范式使用特定任务的数据微调强大的 FMs

直接使用预训练的 FMs 进行推荐面临两个主要挑战：

1.  **领域适配不足**：尽管 FMs 拥有丰富的世界知识，但它们缺乏对特定推荐领域（如电商点击率预测）的细粒度理解，导致在特定任务上的表现不如专门训练的小模型。FMs 可能无法胜过针对任务相关数据专门训练的推荐模型
2.  **输入输出不匹配**：通用 FMs 通常设计为处理自然语言任务，而推荐系统通常依赖于稀疏的 ID 特征和协同信号，直接使用 FMs 难以有效整合这些异构数据

微调 FMs 的动机在于结合**通用知识**与**领域专长**

*   **知识迁移**：通过微调，可以将 FMs 中蕴含的广泛世界知识（如物品的背景知识、常识推理能力）迁移到推荐任务中，特别是在冷启动场景下表现优异。TallRec 展示了对冷启动推荐的有效性
*   **指令遵循**：通过指令微调（Instruction Tuning），可以让 FMs 理解各种推荐任务的指令（如“根据历史推荐电影”、“解释推荐理由”），从而实现多任务统一建模。LLMs 可以理解并遵循不同的推荐指令

该范式通常涉及两种微调策略：

1.  **全参数微调 (Full Fine-tuning)**：调整模型的所有参数，但计算成本极高
2.  **高效参数微调 (PEFT)**：如 LoRA (Low-Rank Adaptation)，只调整少量参数，大大降低了计算资源需求

典型的方法包括：

| 方法 | 描述 |
| :--- | :--- |
| **InstructRec** ([arXiv](https://arxiv.org/abs/2305.07001)) | 设计了大量指令模板（如偏好、意图、任务形式），通过指令微调使 LLM 能够适应不同的推荐任务。 <alphaxiv-paper-citation title="InstructRec" page="10" first="designs abundant instructions" last="context of a user." /> |
| **TallRec** ([arXiv](https://arxiv.org/abs/2305.00447)) | 使用 **LoRA** 进行两阶段微调：先在通用指令数据（Alpaca）上微调，再在推荐数据上微调，有效提升了少样本和冷启动性能。 <alphaxiv-paper-citation title="TallRec" page="10" first="uses LoRA, a" last="information of users." /> |
| **BIGRec** ([arXiv](https://arxiv.org/abs/2305.00447)) | 针对 LLM 难以整合统计数据（如流行度、协同过滤信号）的问题，通过微调将这些信号注入到生成过程中。 <alphaxiv-paper-citation title="BIGRec" page="10" first="BIGRec fine-tunes" last="collaborative filtering" /> |
| **TCF** | 微调 LLM 以创建通用的物品表示，尽管其仍然面临新数据适应性的挑战。 <alphaxiv-paper-citation title="TCF" page="10" first="fine-tunes LLMs to" last="recommendation tasks," /> |

#### Non-tuning FM4RS

这一范式的核心任务是**在不更新模型参数的情况下，通过提示工程（Prompt Engineering）直接激活基础模型（FMs）的推荐能力**

具体来说，它通过精心设计包含用户画像、历史行为、任务指令等信息的提示（Prompts），引导预训练的大语言模型（LLMs）执行推荐任务（如排序、评分预测、解释生成）。Non-tuning FM4RecSys 专注于设计适当的提示，以激发 LLM 的推荐能力

尽管 FMs 具有强大的通用能力，但直接将其用于推荐面临以下挑战：

1.  **位置偏差与顺序感知不足**：LLMs 对输入序列中的顺序和位置非常敏感，往往倾向于推荐出现在提示末尾或特定位置的物品，且难以准确捕捉长序列中的时序依赖。LLMs 往往难以感知历史交互序列的顺序
2.  **性能差距**：在序列推荐等任务上，仅依赖上下文学习（In-context Learning）的 LLMs 通常不如专门训练的监督学习模型（如 SASRec），尤其是随着序列长度增加，性能会下降。LLMs 的表现仍落后于 SASRec 等传统监督方法

该范式的核心动机在于利用 FMs 强大的 **零样本（Zero-shot）** 和 **少样本（Few-shot）** 学习能力

*   **无需训练**：FMs 已经在大规模语料上预训练，蕴含了丰富的世界知识和逻辑推理能力。研究者假设 FMs 本身具备推荐潜力，只需通过合适的提示即可“唤醒”这些能力，从而避免昂贵的微调成本。FMs 固有地拥有推荐能力，旨在通过定制提示来激活这些能力
*   **上下文学习**：通过在提示中提供示例（Few-shot）或特定的推理策略，模型可以根据上下文快速适应新任务

该范式主要依靠**提示工程（Prompt Engineering）**和**上下文学习（In-context Learning）**策略：

1.  **提示设计框架**：构建包含任务描述、用户行为、候选物品的结构化提示。例如，Liu et al. 提出了一个评估 ChatGPT 推荐能力的提示构建框架
2.  **增强策略**：
    *   **近因聚焦提示 (Recency-focused Prompting)**：强调最近的交互，以缓解模型对顺序感知的不足
    *   **上下文学习 (In-context Learning)**：在提示中加入少量的推荐示例（Few-shot），帮助模型理解任务模式
    *   **启发式提示**：结合协同过滤信息、知识图谱推理路径等辅助信息来增强提示的丰富度

*   **ChatGPT Evaluation Framework**: Liu et al. [arXiv: 2305.02182]
*   **Large Language Models are Zero-Shot Rankers**: Hou et al. [arXiv: 2305.08845] (提出了 Recency-focused prompting)
*   **Prompting Large Language Models for Recommender Systems**: Zhang et al. [arXiv: 2401.04997]

#### Pre-trained FM4RS

这一范式的核心任务是**从头开始（或基于现有架构）在大规模推荐数据集上预训练一个基础模型**，使其能够直接处理各种下游推荐任务

与仅使用通用语料训练的 LLMs 不同，这些模型专门针对推荐数据（如用户行为序列、商品元数据）进行预训练，旨在学习通用的用户偏好表示和物品语义。该范式旨在预训练一个通用的推荐引擎

传统的推荐模型通常是**针对特定任务或特定领域训练的**（Task-specific & Domain-specific）

1.  **缺乏泛化性**：在一个数据集上训练的模型很难迁移到另一个领域（如从电影推荐迁移到书籍推荐）
2.  **多任务割裂**：不同的推荐任务（如评分预测、序列推荐、解释生成）通常需要不同的模型架构，难以统一建模
3.  **冷启动问题**：依赖 ID 的模型在面对新用户或新物品时表现不佳

该范式的核心动机在于构建一个**统一的、多任务的推荐基础模型**

*   **统一范式**：通过将所有推荐任务转化为统一的格式（如自然语言序列生成），可以使用同一个模型处理排序、检索、解释等多种任务。P5 将所有任务统一为自然语言序列生成
*   **跨域迁移**：利用大规模多领域数据的预训练，模型可以学习到跨领域的通用知识和用户行为模式，从而实现零样本或少样本迁移
*   **通用表示**：通过学习通用的序列表示，模型不再受限于特定的 Item ID，而是可以基于内容（如文本描述）进行推荐

该范式通常采用 **Transformer** 架构，并利用 **自监督学习（Self-Supervised Learning）** 任务进行预训练。常见的方法包括：

| 方法 | 描述 |
| :--- | :--- |
| **P5 (Pretrain, Personalized Prompt, Predict Paradigm)** ([arXiv](https://arxiv.org/abs/2203.13366)) | 将各类推荐任务（如评分、解释、序列推荐）统一为自然语言输入输出，通过个性化提示进行多任务预训练。 <alphaxiv-paper-citation title="P5 模型" page="10" first="P5 unifies various" last="personalized prompts." /> |
| **M6-Rec** ([arXiv](https://arxiv.org/abs/2205.08084)) | 基于 M6 模型，将推荐任务（如检索、排序、解释）统一为文本生成任务，支持少样本和跨域推荐。 <alphaxiv-paper-citation title="M6-Rec" page="10" first="M6-Rec unifies" last="text generation tasks." /> |
| **UniSRec** ([arXiv](https://arxiv.org/abs/2206.05941)) | 学习通用的序列表示，通过对比学习对齐物品的文本描述和协同信号，从而摆脱对特定 ID 的依赖。 <alphaxiv-paper-citation title="UniSRec" page="10" first="learns universal sequence" last="ID dependence." /> |
| **RecFormer** ([arXiv](https://arxiv.org/abs/2306.05817)) | 将物品视为句子，利用双向 Transformer 学习语言表示，用于序列推荐。 <alphaxiv-paper-citation title="RecFormer" page="10" first="treats items as" last="sequential recommendation." /> |
| **PTUM** ([arXiv](https://arxiv.org/abs/2010.01494)) | 通过掩码行为预测和下一 K 行为预测等自监督任务，从无标签的用户行为中预训练用户模型。 <alphaxiv-paper-citation title="PTUM" page="10" first="pre-trains user models" last="unlabeled user behaviors." /> |

### Agentic Paradigm

#### Agent as RS

#### Agent as User Simulator