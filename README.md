# PicoGPT / 60 行 NumPy 代码实现 GPT-2
# 一、项目

本项目从 [jaymody/picoGPT](https://github.com/jaymody/picoGPT) Fork 而来，文章主体内容翻译和整理自 [GPT in 60 Lines of Numpy](https://jaykmody.com/blog/gpt-from-scratch/)，并对部分内容做了额外补充，内容也将在 [PicoGPT](https://github.com/LLLLLayer/PicoGPT) 持续迭代完善。文中引用了 [Jay Alammar](https://jalammar.github.io/) 的书籍 [Hands-On Large Language Models](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/)、博客等，及其他与文章主题相关的文章、论文的部分内容，这部分内容均在引用位置进行了标注。

PicoGPT 是一个基于 NumPy 的 GPT-2 极简实现，其核心逻辑代码仅 60 行，其中前向传播代码 40 行。通过加载 OpenAI 发布的预训练 GPT-2 模型权重，本项目可以实现文本生成功能。

本项目侧重于 GPT-2 基础概念的介绍和代码实现，旨在通过简洁的代码帮助读者理解其核心架构。因此，项目不深入探讨 LLM 的复杂理论(如训练算法、优化策略、分布式训练等)。

阅读本文档，我们假定读者：

1. 熟悉 Python、Numpy，还有一些训练神经网络的基础；
2. 理解此实现以教学为主要目的，为保持简洁性，代码有意省略了部分非核心功能，而在整体架构上力求完整。

## 1.1 依赖项

```bash
pip install -r requirements.txt
```
本项目已在 Python 3.9.10 环境下测试通过。

## 1.2 使用

```bash
python gpt2.py "Alan Turing theorized that computers would one day become"
```

>  生成内容：
>
> ```
> the most powerful machines on the planet.
> The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.
> ```

# 二、GPT 是什么?

**GPT**(Generative Pre-trained Transformer)，是一类基于 Transformer 的神经网络架构：

- **生成式**(Generative)：GPT 可以生成文本；
- **预训练**(Pre-trained)：GPT 基于来自于书本、互联网等的海量文本进行训练；
- **Transformer**：GPT 采用的是仅包含解码器(decoder-only)部分的 Transformer 神经网络结构。

Transformer 最早在 2017 年发表的著名论文 [Attention Is All You Need](https://huggingface.co/papers/1706.03762) 中得到探讨，该论文首次提出了 Transformer 架构，为后续的 GPT 等模型奠定了基础。它完全基于注意力机制，在 Transformer 中，编码和解码组件堆叠在一起。编码器负责理解输入，解码器负责生成输出。GPT 只保留了解码器部分，所以叫“decoder-only”。

| ![Hands-On Large Language Models Figure 1-16. The Transformer is a combination of stacked encoder and decoder blocks.](./README.assets/the_transformer.png) | ![Hands-On Large Language Models Figure 1-1. A peek into the history of Language AI.](./README.assets/the_architecture_of_a_gpt_1.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

像 OpenAI 的 GPT-3 这样的大型语言模型 (Large Language Models，LLM)底层均采用 GPT 架构。其特点在于经过大量数据训练，模型规模极大，例如：

1. [OpenAI GPT-3](https://huggingface.co/papers/2005.14165)：参数量约 1750 亿(175B)，训练数据约 3000 亿个 token，数据量 45TB 左右；
2. [Google LaMDA](https://huggingface.co/papers/2201.08239)：LaMDA 1 代约 1370 亿(137B)，在预训练阶段收集并创建了一个具有 1.56T 单词的数据集。

| ![Hands-On Large Language Models Figure 1-25. GPT models quickly grew in size with each iteration.](./README.assets/gpt_models_quickly_grew_in_size_with_each_iteration.png) | ![Hands-On Large Language Models Figure 1-28. A comprehensive view into the Year of Generative AI.](./README.assets/the_year_of_generative_ai.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

## 2.1 输入和输出

LLM 接收 Prompt 并响应，GPT 的函数签名类似这样：

![Hands-On Large Language Models Figure 2-2. High-level view of a language model and its input prompt.](./README.assets/high_level_view_of_a_language_model_and_its_input_prompt.png)

```python
def gpt(inputs: list[int]) -> list[list[float]]:
    # 参数：
    # inputs: 长度为 n_seq 的一维数组
    # 表示一个输入的 token ID 序列，即一句话中的各个 token 对应的 ID 序列
    # 返回：
    # output: 形状为 [n_seq, n_vocab] 的二维数组
    # 每行表示对应位置 token 的下一个 token 的概率分布
    # 其中 n_vocab 是词表大小，每个元素表示对应词的预测概率
    output = # 神经网络的前向传播计算
    return output
```

输入的文本被表示成一串整数序列，每个整数都与文本对应：

![Hands-On Large Language Models Figure 2-4. A tokenizer processes the input prompt and prepares the actual input into the language model: a list of token IDs. The specific token IDs in the figure are just demonstrative.](./README.assets/a_tokenizer_processes_the_input_prompt.png)

```python
# 整数表示文本中的 token ID，例如：
# text  = "not all heroes wear capes"
# token = "not" "all" "heroes" "wear" "capes"
inputs =    [1,    0,       2,     4,      6]
```

token 是文本的基本语言单元，它们由某种**分词器**(Tokenizer)产生。我们可以通过一个**词表**(Vocabulary)将 token 映射为整数：

```python
# 词表中每个 token 的 index 就是该 token 的整数 ID
# 例如 "heroes" 的 index 是 2，因为 vocab[2] = "heroes"
vocab = ["all", "not", "heroes", "the", "wear", ".", "capes"]

# 一个假想的分词器，按空格进行分词
tokenizer = WhitespaceTokenizer(vocab)

# encode() 方法将字符串转换为整数列表
ids = tokenizer.encode("not all heroes wear") # ids = [1, 0, 2, 4]

# 可以通过词表映射查看实际的 token
tokens = [tokenizer.vocab[i] for i in ids] # tokens = ["not", "all", "heroes", "wear"]

# decode() 方法将整数列表转换回字符串
text = tokenizer.decode(ids) # text = "not all heroes wear"
```

简单来说：首先将字符串用分词器拆解为 token，再通过词汇表将每个 token 映射为整数 ID。

在实际中，我们会使用一些更高级的分词器，如 [Byte-Pair Encoding](https://huggingface.co/learn/llm-course/chapter6/5) 或者 [WordPiece](https://huggingface.co/learn/llm-course/chapter6/6?fw=pt) 等，其原理是一致的：

1. 有一个 `vocab` 将字符串映射到整数索引；
2. 有一种 `encode` 方法可以转换 `str -> list[int]`；
3. 有一种 `decode` 方法可以转换 `list[int] -> str`。

GPT 输出是一个二维数组，其中 `output[i][j]` 表示模型的预测概率，这个概率代表了词汇表中位于 `vocab[j]` 的 token 是下一个 token `inputs[i+1]` 的概率，如：

| ![Hands-On Large Language Models  Figure 3-7. The tokens with the highest probability after the model’s forward pass. Our decoding strategy decides which of the tokens to output by sampling based on the probabilities.](./README.assets/the_tokens_with_the_highest_probability.png) | ![Hands-On Large Language Models Figure 2-5. Tokenizers are also used to process the output of the model by converting the output token ID into the word or token associated with that ID.](./README.assets/tokenizers_are_also_used_to_process_the_output_of_the_model.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

```python
vocab = ["all", "not", "heroes", "the", "wear", ".", "capes"]
inputs = [1, 0, 2, 4] # "not" "all" "heroes" "wear"
output = gpt(inputs)

#              ["all", "not", "heroes", "the", "wear", ".", "capes"]
# output[0] =  [0.75    0.1       0.0   0.15     0.0  0.0      0.0 ]
# 只给定 "not"，模型预测下一个词是 "all" 的概率最大

#              ["all", "not", "heroes", "the", "wear", ".", "capes"]
# output[1] =  [ 0.0    0.0       0.8    0.1     0.0   0.0     0.1 ]
# 给定序列 ["not", "all"]，模型预测下一个词是 "heroes" 的概率最大

#              ["all", "not", "heroes", "the", "wear", ".", "capes"]
# output[-1] = [ 0.0    0.0       0.0    0.1     0.0 0.05     0.85 ]
# 给定整个序列 ["not", "all", "heroes", "wear"]，模型预测下一个词是 "capes" 的概率最大
```

为了获得整个序列的下一个 token 预测，我们只需取概率最高的标记  `output[-1]`：

```python
vocab = ["all", "not", "heroes", "the", "wear", ".", "capes"]
inputs = [1, 0, 2, 4] # "not" "all" "heroes" "wear"
output = gpt(inputs)
next_token_id = np.argmax(output[-1]) # next_token_id = 6
next_token = vocab[next_token_id]     # next_token = "capes"
```

将具有最高概率的 token 作为结果，叫做[**贪婪解码**(Greedy decoding)](https://docs.cohere.com/docs/controlling-generation-with-top-k-top-p#1-pick-the-top-token-greedy-decoding)或者**贪婪采样**(Greedy sampling)。

预测序列中下一个最合理 token 的任务被称为**语言建模**(Language modeling)，我们可以把 GPT 称为一种**语言模型**(Language model)。

## 2.2 生成文本

### 自回归(Autoregressive)

我们可以迭代地从模型中获取下一个 token 来生成完整的句子。在每次迭代中，我们将预测的 token 添加回输入：

![Hands-On Large Language Models Figure 1-27. The context length is the maximum context an LLM can handle.](./README.assets/the_context_length_is_the_maximum_context_an_llm_can_handle.png)

```python
def generate(inputs, n_tokens_to_generate):
    for _ in range(n_tokens_to_generate): # 自回归 decode 循环
        output = gpt(inputs)              # 模型推理
        next_id = np.argmax(output[-1])   # 贪婪采样
        inputs.append(int(next_id))       # 将预测结果添加回输入
    return inputs[len(inputs) - n_tokens_to_generate :]  # 只返回生成的 token id

input_ids = [1, 0]                             # "not" "all"
output_ids = generate(input_ids, 3)            # output_ids = [2, 4, 6]
output_tokens = [vocab[i] for i in output_ids] # "heroes" "wear" "capes"
```

模型对序列元素进行回归分析(Regression)，预测未来值并将其自动添加回输入的过程就是 GPT 被描述为**自回归**(Autoregressive)的原因。

### 采样(Sampling)

我们在选择 token 时可以引入一些**随机性**(Stochasticity)，通过采样而非总是选择最大概率的 token：

```python
inputs = [1, 0, 2, 4] # "not" "all" "heroes" "wear" 的 token ID
output = gpt(inputs)
np.random.choice(np.arange(vocab_size), p=output[-1]) # capes
np.random.choice(np.arange(vocab_size), p=output[-1]) # hats
np.random.choice(np.arange(vocab_size), p=output[-1]) # capes
np.random.choice(np.arange(vocab_size), p=output[-1]) # capes
np.random.choice(np.arange(vocab_size), p=output[-1]) # pants
```

通过随机采样，我们可以基于相同的输入生成多样化的文本。

结合 [top-k](https://docs.cohere.ai/docs/controlling-generation-with-top-k-top-p#2-pick-from-amongst-the-top-tokens-top-k)、[top-p](https://docs.cohere.ai/docs/controlling-generation-with-top-k-top-p#3-pick-from-amongst-the-top-tokens-whose-probabilities-add-up-to-15-top-p) 和 [temperature](https://docs.cohere.ai/docs/temperature) 等采样策略，可以显著提升生成文本的质量和多样性。这些技术还引入了一些**超参数**(Hyperparameter)，这些参数需要在训练前由开发者手动设定，而非通过模型学习获得。通过调整这些超参数，我们可以控制模型的生成行为，实现从保守到创造性的不同生成风格。

> 1. Top-k ：在每一步生成中，仅从概率最高的 k 个词汇中采样，有效平衡了确定性和多样性，防止模型总是选择最高概率的词。
> 2. Top-p ：动态选择累积概率达到阈值 p 的最小词汇集合进行采样，相比 top-k 更灵活，能根据概率分布的形状自适应调整候选集大小。
> 3. Temperature ：通过缩放未归一化得分(Logits) `logits/temperature` 调整概率分布的锐度，较低的温度(如 0.7)使分布更集中，生成更可预测；较高的温度(如 1.3)使分布更平坦，生成更多样化但可能不太连贯。
>
> “I am driving a…” Temperature 示例：
>
> ![Hands-On Large Language Models Figure 6-4. A higher temperature increases the likelihood that less probable tokens are generated and vice versa.](./README.assets/temperature.png)

## 2.3 训练(Training)

**损失函数**(Loss Function)是衡量模型预测结果与真实结果之间差距的一个函数。训练 GPT 与其它神经网络类似，针对特定的损失函数，使用[**梯度下降**(Gradient descent optimization algorithms)](https://huggingface.co/papers/1609.04747)训练模型，该方法计算损失函数相对于模型参数的梯度(导数)，然后沿着能够减小损失的方向调整参数。

![Hands-On Large Language Models Figure 10-10. Multiple negatives ranking loss aims to minimize the distance between related pairs of text, such as questions and answers, and maximize the distance between unrelated pairs, such as questions and unrelated answers.](./README.assets/loss_aims.png)

对于 GPT，我们使用**语言建模任务**(Language Modeling) —— 即给定一段文本的前面部分，预测下一个最有可能出现的 token。该任务通常使用[**交叉熵损失**(Cross Entropy Loss)](https://www.youtube.com/watch?v=ErfnhcEV1O8)作为损失函数：

> 假设有三个类别，真实标签是类别 2：
>
> 1. 如果模型预测概率分布为 `[0.1, 0.7, 0.2]`(正确地给了类别 2 最高概率)，损失值为 -log(0.7) ≈ 0.36；
> 2. 如果模型预测概率分布为 `[0.1, 0.2, 0.7]`(错误地给了类别 3 最高概率)，损失值为 -log(0.2) ≈ 1.61。
>
> 显然，第二种情况损失更大，这正是我们期望的：模型预测错误时应当受到更大的惩罚。

```python
def lm_loss(inputs: list[int], params) -> float:
    # 计算语言模型的交叉熵损失
    # 参数：
    # inputs: 输入 token ID 序列，例如 [not, all, heroes, wear, capes]
    # params: 模型参数
    # 返回：
    # float: 平均交叉熵损失值
    #
    # 构建输入-标签对：
    # 输入 x 是除最后一个 token 外的序列，标签是除第一个 token 外的序列
    # 标签 y 只是 inputs 向左移动 1 位
    # 例如：
    # inputs = [not, all, heroes, wear, capes]
    #      x = [not, all, heroes, wear]
    #      y =      [all, heroes, wear, capes]
    # 
    # 对于长度为 N 的 inputs，我们可以构建 N-1 个输入-标签对
    # 两者的形状都是 [序列中的 token 数 - 1]
    x, y = inputs[:-1], inputs[1:] 
    
    # 前向传播(Forward pass)：获取模型在每个位置对下一个 token 的预测概率分布
    output = gpt(x, params) # 形状为 [序列中的 token 数 - 1, 词表中的 token 数]
    
    # 计算交叉熵损失：
    # -log(p(y_i))，其中 p(y_i) 是模型对真实下一个 token 的预测概率
    # np.arange(len(output)) 生成行索引，y 提供列索引，两者共同定位每个位置的真实 token 概率
    # np.mean(token_losses) 返回平均损失
    loss = np.mean(-np.log(output[np.arange(len(output)), y]))
    
    # 最终得到的是一个一维数组，长度等于行数，每个元素是模型在该位置预测真实下一个 token 的概率
    return loss

def train(texts: list[list[str]], params) -> dict:
    # 训练语言模型
    # 参数：
    # texts: 训练文本列表
    # params: 初始模型参数
    # 返回：
    # dict: 训练后的模型参数
    for text in texts:
        #
        # 用 tokenizer.encode(text) 把文本转成 token id 的序列，方便模型处理
        inputs = tokenizer.encode(text)
        #
        # 计算当前样本的损失：
        # 计算当前模型在这条数据上的损失，衡量模型预测和真实答案的差距
        loss = lm_loss(inputs, params)
        #
        # 计算梯度：
        # 反向传播算法(Backpropagation): 根据损失函数的结果，自动计算每个参数对损失的影响，用于指导参数调整
        # 通过反向传播算法计算损失对参数的梯度，即每个参数该怎么调整能让损失变小
        gradients = compute_gradients_via_backpropagation(loss, params)
        #
        # 使用梯度下降更新参数：
        # 根据梯度调整参数，即“优化”模型，让模型表现更好
        params = gradient_descent_update_step(gradients, params)
    #
    # 所有文本都训练一遍后，返回更新后的参数
    return params
```

这就是一个极度简化但典型的神经网络训练循环：编码输入、计算损失、反向传播求梯度、用梯度下降法更新参数。不断重复这个过程，模型就会越来越“聪明”，预测能力越来越强。

值得注意的是，语言模型训练不需要人工标注的数据。相反，我们利用文本数据本身的内在结构——每个 token 都可以作为其前面 tokens 的预测目标，从而自动生成大量的输入-目标对。这种方法被称为[**自监督学习**(Self-supervised learning)](https://en.wikipedia.org/wiki/Self-supervised_learning)。自监督使我们能够大规模扩展训练数据。我们只需要获取尽可能多的原始文本，并将其输入到模型中即可。例如，GPT-3 使用了来自互联网和书籍的 3000 亿个文本 tokens 进行训练，以下图表来自 [Language Models are Few-Shot Learners](https://huggingface.co/papers/2005.14165)：

![Language Models are Few-Shot Learners Table 2.2: Datasets used to train GPT-3](./README.assets/datasets_used_to_train_gpt_3.png)

这个自监督训练的步骤称之为**预训练**(Pre-training)，我们可以重复使用预训练模型权重来训练下游任务上的特定模型，预训练模型有时也被称为**基础模型**(Foundation models)。

在下游任务上训练模型被称之为**微调**(Fine-tuning)，因为模型权重已经过预先训练以理解语言，因此它只是针对手头的特定任务进行微调。这种“在通用任务上预训练+在特定任务上微调”的策略，称之为[**迁移学习**(Transfer learning)](https://en.wikipedia.org/wiki/Transfer_learning)。

![Hands-On Large Language Models Figure 1-30. Compared to traditional machine learning, LLM training takes a multistep approach.](./README.assets/compared_to_traditional_machine_learning.png)

> 传统的机器学习通常涉及针对特定任务(例如分类)训练模型，我们认为这是一个单步过程：![Hands-On Large Language Models Figure 1-29. Traditional machine learning involves a specific target task, like classification or regression.](./README.assets/traditional_machine_learning_involves_a_single_step.png)

> 自回归、前向传播、反向传播的定义：
>
> 1. 自回归：是一种模型结构，指模型在生成序列时，每一步的输出都依赖于之前已经生成的内容。GPT 就是典型的自回归模型。
> 1. 反向传播：是在前向传播得到输出后，根据损失函数计算误差，并将误差信息从输出层反向传递到输入层，逐层计算梯度，用于更新模型参数。
>
> 2. 前向传播：指神经网络从输入到输出的计算过程，即数据依次经过每一层网络，最终得到输出结果(如 logits 或概率)。
>
> 自回归、前向传播、反向传播的关系 ：
>
> 1. 自回归是模型结构和生成方式，前向/反向传播是神经网络训练和推理的基本计算过程，两者可以结合：自回归模型的每一步都用前向传播计算输出，训练时再用反向传播优化参数。
> 1. 训练时，前向传播用于计算输出，反向传播用于根据损失调整参数。
> 1. 推理时，每生成一个 token，都会进行一次前向传播，得到当前 token 的预测概率分布。

## 2.4 提示(Prompting)

2018 年发表的 [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) 论文首次提出了 GPT-1 模型，介绍了一种基于 Transformer 架构的生成式预训练方法，确立了如今广泛采用的两阶段训练范式：首先在大规模无标注文本上进行自监督预训练，然后在特定任务的标注数据上进行有监督微调。实验表明，拥有1.17 亿参数的 GPT-1 模型经过微调后，在多种**自然语言处理**(Natural Language Processing，NLP)任务上取得了当时最先进的性能，证明了这种预训练-微调范式的有效性。

随着 2019 年 [Language Models are Unsupervised Multitask Learners ](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) GPT-2 和 [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) GPT-3 论文发表，研究人员发现：当模型规模和训练数据量达到足够大时，语言模型会表现出**涌现**能力(Emergent Abilities) ，指能够在没有任何参数更新的情况下执行全新任务。

这种能力通过**提示工程**(Prompting) 来激活，即通过精心设计的文本指令引导模型执行特定任务。这种新范式被称为**上下文学习**(In-context Learning)。根据提示中包含的示例数量，可分为三种模式：

- 零样本学习(Zero-shot Learning) ：提示中不包含任何完成任务的示例；
- 单样本学习(One-shot Learning) ：提示中包含一个示例；
- 少样本学习(Few-shot Learning) ：提示中包含少量示例。

![Hands-On Large Language Models Figure 6-13. An example of a complex prompt with many components.](./README.assets/x_shor_prompt.png)

> ![Language Models are Few-Shot Learners Figure 2.1: Zero-shot, one-shot and few-shot, contrasted with traditional fine-tuning](./README.assets/zero_shot_one_shot_and_few_shot_contrasted_with_traditional_fine_tuning.png)

这种基于提示的文本生成在技术上被称为**条件生成**(Conditional Generation) ——模型的输出受到输入 Prompt(条件)的约束和引导。

值得注意的是，GPT 模型的应用已远超传统 NLP 任务。通过不同的 Prompt 设计，同一模型能够：

1. 执行文本摘要、翻译、问答等多种语言任务；

2. 生成代码、解决数学问题等结构化任务；
3. 作为对话系统的核心，其中对话历史作为条件引导回复生成；
4. 处理多模态任务(如在 GPT-4)。

**提示工程**(Prompt Engineering)已成为一门重要技术，通过精心设计提示可以显著提升模型性能，但同时也需认识到模型在事实准确性、偏见、幻觉等方面的固有局限。

以下是一个包含多个组件的复杂提示示例：

![Hands-On Large Language Models Figure 6-4 Figure 6-11. An example of a complex prompt with many components.](./README.assets/an_example_of_a_complex_prompt_with_many_components.png)

该图展示了一个包含多个组件的复杂 Prompt 示例。通过在输入中加入任务说明、上下文信息、示例输入输出等不同部分，可以有效引导大语言模型按照预期方式完成复杂任务。这种多组件提示设计体现了提示工程的重要性，有助于提升模型的理解能力和输出质量。

# 三、实现 GPT

## 3.1 下载和了解项目

克隆本教程的存储库，并安装依赖项：

```
git clone https://github.com/LLLLLayer/PicoGPT.git
cd PicoGPT
pip install -r requirements.txt
```

项目文件包含：

| 文件名             | 功能描述                                                     |
| ------------------ | ------------------------------------------------------------ |
| **`encoder.py`**   | OpenAI 的 BPE(Byte Pair Encoding)分词器实现，源自 [openai gpt-2](https://github.com/openai/gpt-2/blob/master/src/encoder.py) |
| **`utils.py`**     | 提供下载和加载 GPT-2 模型权重、分词器和超参数的工具函数      |
| **`gpt2.py`**      | 完整实现的 GPT-2 模型代码，包含详细注释，支持直接运行        |
| **`gpt2_pico.py`** | 与 gpt2.py 功能相同的精简版本，移除注释以突出核心代码        |

请将 `gpt2.py` 文件内容替换为如下代码：

> 本节重点关注代码流程，相关概念将在后文详细解释。

```python
import numpy as np

def main(prompt: str, 
         n_tokens_to_generate: int = 40, 
         model_size: str = "124M", 
         models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params
    # 1. 从 openai gpt-2 文件中加载编码器、超参数和参数，这将下载必要的文件到 models/124M 中
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    
    # 2. 使用编码器对输入字符串进行编码
    input_ids = encoder.encode(prompt)

    # 3. 确保不超过模型的最大序列长度
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # 4. 生成输出的 token ID
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # 5. 将生成的 token ID 序列通过解码器映射回可读文本
    output_text = encoder.decode(output_ids)
    
    # 6. 返回结果
    return output_text

# 使用自回归方式生成文本
def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm
    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        # 模型前向传播
        logits = gpt2(inputs, **params, n_head=n_head) 
        # 贪婪采样
        next_id = np.argmax(logits[-1])
        # 将预测结果添加到输入中
        inputs.append(int(next_id))
    # 只返回新生成的 token ID
    return inputs[len(inputs) - n_tokens_to_generate :] 

# GPT-2 模型的前向传播函数
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    pass

if __name__ == "__main__":
    import fire
    fire.Fire(main)
```

上述代码包含四个部分：

1. `main` 函数：协调整体工作流程：初始化模型环境、输入处理、文本生成、输出处理；
2. `generate` 函数：实现自回归生成过程，为了简洁这里将使用贪婪解码，其中 [`tqdm`](https://github.com/tqdm/tqdm) 提供进度展示，以直观地看到解码过程；
3. `gpt2` 函数：待实现的前向传播函数；
4. 命令行接口：通过 [`fire.Fire(main)`](https://github.com/google/python-fire) 将 Python 脚本转换为命令行应用，以支持 `python gpt2.py "prompt"` 调用。

## 3.2 GPT 架构概览

GPT 的架构是基于 [Transformer](https://huggingface.co/papers/1706.03762) 的，但与原始 Transformer 不同，GPT 仅使用解码器部分，移除了编码器-解码器之间的交叉注意力机制，这种设计使模型专注于自回归语言建模任务。以下左图为原始 Transformer 架构，右图为 GPT 架构：

| ![Attention Is All You Need Figure 1: The Transformer - model architecture](./README.assets/the_transformer_model_architecture_1.png) | ![The GPT architecture](./README.assets/the_transformer_model_architecture_2.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

**左图：这张图展示了原始 Transformer 模型的完整架构，包含了编码器和解码器两部分：**

1. 左侧是编码器(Encoder)部分，用于处理输入序列；
2. 右侧是解码器(Decoder)部分，用于生成输出序列；
3. 图中的 N× 表示这些模块重复 N 次，形成多层结构；
4. 底部是输入嵌入(Input Embedding)和位置编码(Positional Encoding)的组合；
5. 中间有多头注意力(Multi-Head Attention)机制和前馈神经网络(Feed Forward)层；
6. 每个主要组件后都有 Add & Norm 层，实现残差连接(Residual Connection)和层归一化(Layer Normalization)；
7. 解码器部分有额外的掩码多头注意力(Masked Multi-Head Attention)层，确保自回归生成过程中只能看到已生成的 token。

**右图：展示了 GPT 模型的简化架构，与原始 Transformer 相比有以下区别：**

1. GPT只使用了 Transformer 的解码器部分 ，移除了编码器-解码器间的交叉注意力机制；
2. 底部是文本和位置的组合嵌入(Text + Position Embedding)；
3. 中间是N个重复的 Transformer 块，每个块包含：
   1. 多头因果自注意力机制(Multi-Head Casual Self-Attention)，"Casual"表示它是单向的，只能看到当前及之前的 token；
   2. 前馈神经网络(Feed Forward)；
   3. 残差连接(+ 符号表示)、层归一化(Layer Norm)；
4. 顶部是输出处理部分：
   1. 输出投影、线性变换(Linear)、Softmax 层、将输出转换为下一个 token 的概率分布。

## 3.3 分词器、模型参数与超参数

### 分词器(Tokenizer)

模型生成响应时，并非一次性输出全部内容，而是每次生成一个 token。token 既作为模型的输出，也是模型的输入单位。在将 Prompt 呈现给语言模型之前，首先必须通过分词器的处理。我们可以在 OpenAI 上找到 [GPT-4o 分词器的示例](https://platform.openai.com/tokenizer)：

| ![OpenAI Platform Tokenizer Text](./README.assets/gpt_4o_tokenizer_text.png) | ![OpenAI Platform Tokenizer Token IDs](./README.assets/gpt_4o_tokenizer_token_ids.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

回到代码中，`encoder` 是 GPT-2 使用的 BPE 分词器：

```python
ids = encoder.encode("Not all heroes wear capes.")
print(ids)
# [3673, 477, 10281, 5806, 1451, 274, 13]
```

使用分词器的词汇表，可以查看实际的 token：

```python
tokens = [encoder.decoder[i] for i in ids]
print(tokens)
# ['Not', 'Ġall', 'Ġheroes', 'Ġwear', 'Ġcap', 'es', '.']
```

请注意，有时 token 是单词(如 `Not`)，有时是前面有一个空格的单词(如 `Ġall`，[`Ġ`表示空格](https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/bpe.py#L22-L33))，有时是部分单词(如 capes 分为  `Ġcap` 和  `es`)，有时是标点符号(如 `.`)。

**词汇表**(Vocabulary)和**字节对组合**(Byte-pair merges)是现代自然语言处理中**分词器**(Tokenizer)的核心组成部分。词汇表类似一本字典，它包含了模型能够理解的所有单词(Token)及其对应的数字 ID。在 GPT 模型中，这些 token 可能是真实单词、单个字符或常见词组片段。

[字节对编码](https://huggingface.co/learn/llm-course/chapter6/5)(Byte-Pair Encoding，BPE)是一种数据驱动的分词算法。最初是作为一种文本压缩算法开发的，后来被 OpenAI 在预训练 GPT 模型时用于标记化。许多 Transformer 模型都使用了它，包括 GPT、GPT-2、RoBERTa、BART 和 DeBERTa。

它首先将文本看作单个字符，然后逐步合并最常一起出现的字符对，形成新的 token，这个过程不断重复，直到达到预设的词汇量。假设"机器学习"这个词在语料库中经常出现 BPE 算法可能会将其作为一个完整的 token，而不是分解为"机"、"器"、"学"、"习"四个 token 这样可以更高效地表示常见词组。可参考视频 [Byte Pair Encoding Tokenization](https://youtu.be/HEikzVL-lZU) 了解该流程：

| ![Byte Pair Encoding Tokenization 1](./README.assets/byte_pair_encoding_tokenization_1.png) | ![Byte Pair Encoding Tokenization 2](./README.assets/byte_pair_encoding_tokenization_2.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![Byte Pair Encoding Tokenization 3](./README.assets/byte_pair_encoding_tokenization_3.png) | ![Byte Pair Encoding Tokenization 4](./README.assets/byte_pair_encoding_tokenization_4.png) |

BPE 的一个优点是它可以编码任意字符串，若遇到词汇表中不存在的内容，它会将其分解为它能理解的子字符串：

```python
print([encoder.decoder[i] for i in encoder.encode("zjqfl")])
# ['z', 'j', 'q', 'fl']
```

> 这些文件在运行 `load_encoder_hparams_and_params` 时被下载。
>
> 可以查看 `models/124M/encoder.json` (词汇表)和 `models/124M/vocab.bpe` (字节对组合)。

### 超参数(Hyperparameters)

**超参数**(Hyperparameters)是在模型设计和训练前确定的关键配置，它们定义了模型的结构、容量和训练过程中的行为。GPT-2 的核心超参数包含在 `hparams` 字典中：

```python
print(hparams)
# {
#    "n_vocab": 50257,
#    "n_ctx"  : 1024,
#    "n_embd" : 768,
#    "n_head" : 12, 
#    "n_layer": 12 
# }
```

| 超参数    |              全称              |   值   | 描述             | 影响                                             |
| --------- | :----------------------------: | :----: | ---------------- | ------------------------------------------------ |
| `n_vocab` |      Number of Vocabulary      | 50,257 | 词表大小         | 决定模型的语言覆盖范围，影响嵌入层和输出层参数量 |
| `n_ctx`   |    Number of Context Tokens    | 1,024  | 最大序列长度     | 限制模型的记忆窗口，影响注意力计算复杂度 O(n²)   |
| `n_embd`  | Number of Embedding Dimensions |  768   | 嵌入维度         | 控制模型表达能力的"宽度"，影响所有线性层的参数量 |
| `n_head`  |   Number of Attention Heads    |   12   | 注意力头数       | 提供多角度特征提取，并行计算单元数量             |
| `n_layer` |        Number of Layers        |   12   | Transformer 层数 | 控制特征抽象层次深度，决定前向传播的计算步数     |

>  此外， 使用 `n_seq` (Number of sequence)表示输入序列的长度，即 `n_seq = len(inputs)`。这是一个动态值，取决于输入数据，最大不超过 `n_ctx`。

### 参数(Parameters)

参数 `params` 是一个嵌套的  JSON 字典，用于保存模型的训练权重。JSON 的叶节点是 NumPy 数组，用于存储实际的权重参数。如果打印  `params`，并将数组替换为其形状，我们将得到：

```python
import numpy as np
def shape_tree(d):
    if isinstance(d, np.ndarray):
        return list(d.shape)
    elif isinstance(d, list):
        return [shape_tree(v) for v in d]
    elif isinstance(d, dict):
        return {k: shape_tree(v) for k, v in d.items()}
    else:
        ValueError("uh oh")
print(shape_tree(params))
# {
#     "wpe": [ 1024, 768],
#     "wte": [50257, 768],
#     "ln_f": {"b": [768], "g": [768]},
#     "blocks": [
#         {
#             "attn": {
#                 "c_attn": {"b": [2304], "w": [768, 2304]},
#                 "c_proj": {"b":  [768], "w": [768,  768]},
#             },
#             "ln_1": {"b": [768], "g": [768]},
#             "ln_2": {"b": [768], "g": [768]},
#             "mlp": {
#                 "c_fc"  : {"b": [3072], "w": [ 768, 3072]},
#                 "c_proj": {"b":  [768], "w": [3072,  768]},
#             },
#         },
#         ... # repeat for n_layers
#     ]
# }
```

为了对比，这里显示了 `params` 的形状：

```python
{
    "wpe": [  n_ctx, n_embd], # 位置嵌入: [上下文长度, 嵌入维度]
    "wte": [n_vocab, n_embd], # 词嵌入: [词汇表大小, 嵌入维度]
    "ln_f": {"b": [n_embd], "g": [n_embd]}, # 最终层归一化
    "blocks": [ # 每个 Transformer 块的参数
        {
            "attn": { # 自注意力机制
                "c_attn": {"b": [3*n_embd], "w": [n_embd, 3*n_embd]},
                "c_proj": {"b":   [n_embd], "w":   [n_embd, n_embd]},
            },
            "ln_1": {"b": [n_embd], "g": [n_embd]}, # 第一层归一化
            "ln_2": {"b": [n_embd], "g": [n_embd]}, # 第二层归一化
            "mlp": { # 前馈网络
                "c_fc"  : {"b": [4*n_embd], "w": [n_embd, 4*n_embd]},
                "c_proj": {"b":   [n_embd], "w": [4*n_embd, n_embd]},
            },
        },
        # ... 重复 n_layer 层
    ]
}
```

GPT-2 的参数可以按功能分为三大类：包括嵌入层参数、Transformer 块参数(自注意力机制、前馈网络、层归一化)和输出层参数。

| 参数 | 全称 | 形状 | 参数量 | 功能 |
|------|------|------|--------|------|
| `wte` | Word Token Embeddings | [50257, 768] | 38.6M | 将 token ID 映射为语义向量 |
| `wpe` | Word Position Embeddings | [1024, 768] | 0.8M | 为每个位置提供位置编码 |

| 参数 | 形状 | 参数量/层 | 功能 |
|------|------|-----------|------|
| `c_attn.w` | [768, 2304] | 1.77M | QKV 联合投影矩阵 |
| `c_attn.b` | [2304] | 2.3K | QKV 投影偏置 |
| `c_proj.w` | [768, 768] | 0.59M | 注意力输出投影 |
| `c_proj.b` | [768] | 768 | 输出投影偏置 |

| 参数 | 形状 | 参数量/层 | 功能 |
|------|------|-----------|------|
| `c_fc.w` | [768, 3072] | 2.36M | 扩展投影(4×放大) |
| `c_fc.b` | [3072] | 3.1K | 扩展投影偏置 |
| `c_proj.w` | [3072, 768] | 2.36M | 压缩投影 |
| `c_proj.b` | [768] | 768 | 压缩投影偏置 |

| 参数 | 形状 | 参数量/层 | 功能 |
|------|------|-----------|------|
| `ln_1.g/b` | [768] × 2 | 1.5K | 注意力前归一化 |
| `ln_2.g/b` | [768] × 2 | 1.5K | MLP 前归一化 |

| 参数 | 形状 | 参数量 | 功能 |
|------|------|--------|------|
| `ln_f.g/b` | [768] × 2 | 1.5K | 最终层归一化 |

这些参数是从 OpenAI 提供的 TensorFlow 检查点文件加载的：

```python
import tensorflow as tf
# 加载最新的检查点文件
tf_ckpt_path = tf.train.latest_checkpoint("models/124M")
# 遍历检查点中的所有变量
for name, _ in tf.train.list_variables(tf_ckpt_path):
    # 加载变量并移除多余的维度
    arr = tf.train.load_variable(tf_ckpt_path, name).squeeze()
    # 输出结果展示了模型中每个参数的名称和形状
    print(f"{name}: {arr.shape}")
# model/h0/attn/c_attn/b: (2304,)
# model/h0/attn/c_attn/w: (768, 2304)
# model/h0/attn/c_proj/b: (768,)
# model/h0/attn/c_proj/w: (768, 768)
# model/h0/ln_1/b: (768,)
# model/h0/ln_1/g: (768,)
# model/h0/ln_2/b: (768,)
# model/h0/ln_2/g: (768,)
# model/h0/mlp/c_fc/b: (3072,)
# model/h0/mlp/c_fc/w: (768, 3072)
# model/h0/mlp/c_proj/b: (768,)
# model/h0/mlp/c_proj/w: (3072, 768)
# model/h1/attn/c_attn/b: (2304,)
# model/h1/attn/c_attn/w: (768, 2304)
...
# model/h9/mlp/c_proj/b: (768,)
# model/h9/mlp/c_proj/w: (3072, 768)
# model/ln_f/b: (768,)
# model/ln_f/g: (768,)
# model/wpe: (1024, 768)
# model/wte: (50257, 768)
```

在实现 GPT 时，我们需要回来参考该字典来检查权重的形状。为了保持一致性，我们会将代码中的变量名与字典的键进行匹配。

## 3.4 基础的神经网络层

在实现 GPT 架构本身之前，我们需要先实现一些基础的神经网络层。

### 高斯误差线性单元(GELU)

由 Dan Hendrycks 和 Kevin Gimpel 在 2016 年提出的 [**GELU**(Gaussian Error Linear Units)](https://huggingface.co/papers/1606.08415) 是 GPT-2 的非线性激活函数，在 Transformer 架构中表现优于 ReLU 和其他激活函数。

![Gaussian Error Linear Units Figure 1: The GELU (µ = 0, σ = 1), ReLU, and ELU (α = 1)](./README.assets/gaussian_error_linear_units.png)

> ReLU 激活函数对于任何负数都会在 0 处强制截止，否则会产生线性结果。[激活函数随时间推移的使用情况](https://paperswithcode.com/method/gelu)如下：
>
> ![Usage Over Time](./README.assets/usage_over_time.png)

神经网络的基础运算是线性变换(矩阵乘法与偏置加法)。如果没有非线性激活函数，无论神经网络有多少层，整个网络本质上仍然只是一个线性模型。非线性激活函数打破了这种限制，使网络能够学习复杂的非线性关系。

GELU 激活函数 `GELU(x) = x * Φ(x)` 可以被理解为：将输入值 x 乘以该输入被保留的概率，体现了一种概率性的特征选择机制。这个概率由标准正态分布的累积分布函数(Cumulative Distribution Function，CDF)给出。由于标准正态分布的 CDF 计算复杂，代码使用了一个常用的近似公式：

```python
def gelu(x):
     # GELU 激活函数的近似实现
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
```

GELU 对输入进行逐元素操作：

```python
# GELU 对输入进行逐元素操作
output = gelu(np.array([[1, 2], [-2, 0.5]]))
print(output)
# array([[ 0.84119,  1.9546],
#        [ -0.0454, 0.34571]])
```

> 在 GPT-2 模型中，GELU 主要应用于前馈网络层，作为非线性变换组件。

### Softmax 函数(Softmax)

Softmax 函数在神经网络和深度学习中扮演着非常重要的角色，Softmax 的核心作用是将一组实数值(Logits)转换为概率分布。它确保所有输出值在 0 到 1 之间，且总和为 1：

![Softmax 函数](./README.assets/softmax.png)

在 GPT-2 等语言模型中，Softmax 用于词汇表上的概率分布，帮助模型预测序列中的下一个词。Softmax 函数的标准形式如下，其中 `x_i` 是输入向量的第 `i` 个元素，分母为所有元素指数的总和：
$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

```python
def softmax(x):
    # 1. 数值稳定化：减去每个样本的最大值，防止指数计算溢出
    # 首先从输入 x 中减去每个样本的最大值(最大的输入值变为 0，其他值变为负数)，防止指数计算时出现数值溢出
    # 对调整后的值计算指数 `exp_x = np.exp(x - max_x)`，这将所有值转换为正数
    # axis=-1 表示沿着最后一个维度操作，对于 GPT-2 模型的输出 logits，最后一个维度的大小等于词汇表大小
    # keepdims=True 保持数组的维度结构，便于后续操作
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    # 2. 归一化：确保所有概率值之和为 1
    # 计算指数值的总和，将每个指数值除以总和，这确保输出的所有值在 0 到 1 之间，且总和为 1。
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

```python
x = softmax(np.array([[2, 100], [-5, 0]]))
x
# array([[0.00034, 0.99966],
#        [0.26894, 0.73106]])
x.sum(axis=-1) # 验证概率和为 1
# array([1., 1.])
```

> 在 GPT-2 模型中，Softmax 主要用于两个关键位置： 
>
> | 应用位置   | 作用           | 输入                   | 输出               |
> | ---------- | -------------- | ---------------------- | ------------------ |
> | 注意力机制 | 计算注意力权重 | Query-Key 相似度分数   | 注意力概率分布     |
> | 输出层     | 生成词汇概率   | 最终隐藏状态的线性变换 | 词汇表上的概率分布 |

### 层归一化(Layer Normalization)

[**层归一化**(Layer Normalization，LN)](https://huggingface.co/papers/1607.06450) 是一种用于稳定深度神经网络训练并提升性能的归一化技术。

层归一化将数据标准化为均值为 0、方差为 1 的分布，具体步骤如下：
$$
LayerNorm(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2}} + \beta
$$

>  μ 和 σ² 分别是特征维度上的均值和方差，缩放参数(γ)和偏移参数(β)是可学习的参数。

```python
def layer_norm(x, g, b, eps: float = 1e-5):
    # 层归一化实现
    # 参数：
    # x: 输入张量 [batch_size, hidden_size]
    # gamma: 缩放参数 [hidden_size]
    # beta: 偏移参数 [hidden_size]
    # eps: 数值稳定性常数
    # 返回：
    # 归一化后的张量
    # 
    # 计算特征维度上的均值和方差
    mean = np.mean(x, axis=-1, keepdims=True)    # 对输入张量 x 在最后一个维度(通常是特征维度)上计算均值
    variance = np.var(x, axis=-1, keepdims=True) # 同样在最后一个维度上计算方差
     # 标准化处理：减均值除以标准差
    x = (x - mean) / np.sqrt(variance + eps)     # 将输入减去均值并除以标准差，实现标准化
                                                 # eps 是一个小常数(默认为1e-5)，用于数值稳定性，防止除以零。
    # 应用可学习的缩放和偏移参数
    return g * x + b # 使用可学习的参数 g(gamma)和 b(beta)对标准化后的数据进行线性变换
```

```python
# 示例：对矩阵应用层归一化
x = np.array([[2, 2, 3], [-5, 0, 1]])
x = layer_norm(x, g=np.ones(x.shape[-1]), b=np.zeros(x.shape[-1]))
print(x)
# array([[-0.70709, -0.70709,  1.41418],
#.       [  -1.397,    0.508,    0.889]])
# 验证归一化效果：均值接近 0，方差接近 1
x.var(axis=-1)
# array([0.99996, 1.])
x.mean(axis=-1)
#array([-0., -0.])
```

> 在 GPT 的架构中，层归一化具体应用在两个关键位置：
>
> 1. 多头注意力机制之前：
>
> ```python
> x_norm = layer_norm(x)                    # 先进行层归一化
> attention_output = multi_head_attention(x_norm)  # 再进行注意力计算
> x = x + attention_output                  # 残差连接
> ```
>    
> 2. 前馈神经网络之前：
> 
>   ```python
>   x_norm = layer_norm(x)                    # 先进行层归一化
>   ffn_output = feed_forward_network(x_norm) # 再进行前馈网络计算
>   x = x + ffn_output                        # 残差连接
>   ```

### 线性变换(Linear Transformation)

线性变换是深度神经网络的基础构建模块，通过可学习的权重矩阵实现向量空间之间的映射。在 Transformer 架构中，线性变换承担着特征转换、维度调整和信息处理的关键作用。线性变换的数学表达式为：
$$
y = xW + b
$$
其中：
1. x：输入张量(批次大小 × 输入维度)；
2. W：权重矩阵(输入维度 × 输出维度)；
3. b：偏置向量(输出维度)；
4. y：输出张量(批次大小 × 输出维度)。

线性变换的几何直观理解：

1. 旋转：改变向量的方向；
2. 缩放：改变向量的长度；
3. 剪切：改变向量间的角度关系；
4. 投影：将高维空间映射到低维空间。

`linear` 函数实现了标准的线性变换(矩阵乘法加偏置)，通常被称为**投影**(Projection)。这个名称来源于线性代数中的向量空间投影概念，它将向量从一个向量空间映射到另一个向量空间：

```python
def linear(x, w, b): 
    # 执行线性变换(全连接层操作)
    # [m, in], [in, out], [out] -> [m, out]
    # 参数：
    # x: 输入张量，形状为 [batch_size, in_features](m个样本，每个样本有in个特征)
    # w: 权重矩阵，形状为 [in_features, out_features](将in维输入映射到out维输出)
    # b: 偏置向量，形状为 [out_features](为每个输出维度添加一个偏移)
    # @: 矩阵乘法运算符
    # 返回：
    # 形状为 [batch_size, out_features] 的输出张量(m个样本，每个样本有out个特征)
    return x @ w + b
```

```python
# 表示 64 个样本，每个样本有 784 个特征
x = np.random.normal(size=(64, 784))
# 第一个维度(784)必须匹配输入的特征维度，第二个维度(10)决定了输出的特征数量
w = np.random.normal(size=(784, 10)) 
# 偏置向量长度匹配输出特征数
b = np.random.normal(size=(10,))
# 线性投影前的形状
x.shape # (64, 784)
# 线性投影后的形状
linear(x, w, b).shape # (64, 10)
```

> 在 GPT-2 中，线性变换通常与层归一化、残差连接和非线性激活函数(如GELU)结合使用，形成完整的 Transformer 块结构：
>
> | 位置                 | 作用                 | 输入维度  | 输出维度   |
> | -------------------- | -------------------- | --------- | ---------- |
> | Query/Key/Value 投影 | 注意力机制的特征映射 | d_model   | d_k/d_v    |
> | 注意力输出投影       | 多头注意力结果合并   | d_model   | d_model    |
> | 前馈神经网络第一层   | 特征扩展             | d_model   | 4×d_model  |
> | 前馈神经网络第二层   | 特征压缩             | 4×d_model | d_model    |
> | 输出投影             | 词汇表映射           | d_model   | vocab_size |

## 3.5 GPT 架构实现

GPT-2 采用基于 Transformer 的解码器架构，整个模型可以分为三个核心组件：

1. 输入表示层：词元嵌入 (Word Token Embeddings) + 位置嵌入(Position Embeddings)；
2. Transformer 解码器堆栈(Transformer Decoder Stack)：多层 Transformer 块的堆叠；
3. 输出投影层(Output Projection)：将隐藏状态投影回词汇表空间。

| 组件                | 功能                            | 输入维度          | 输出维度           | 关键操作            |
| ------------------- | ------------------------------- | ----------------- | ------------------ | ------------------- |
| **输入表示层**      | 将离散 token 转换为连续向量表示 | `[n_seq]`         | `[n_seq, n_embd]`  | 嵌入查找 + 位置编码 |
| **Transformer堆栈** | 序列建模与特征提取              | `[n_seq, n_embd]` | `[n_seq, n_embd]`  | 自注意力 + 前馈网络 |
| **输出投影层**      | 将隐藏状态映射回词汇空间        | `[n_seq, n_embd]` | `[n_seq, n_vocab]` | 权重共享的线性投影  |

在代码里表示如下：

```python
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    # GPT-2 模型的前向传播函数
    # 参数：
    # inputs: 输入的 token 序列，形状为 [n_seq]
    #    wte: 词元嵌入矩阵 (Word Token Embeddings)
    #    wpe: 位置嵌入矩阵 (Word Position Embeddings)
    # blocks: Transformer块的列表
    #   ln_f: 最终的层归一化参数
    # n_head: 注意力头的数量
    # 返回：
    # 形状为 [n_seq, n_vocab] 的 logits，表示每个位置上词汇表中每个 token 的概率分布
    
    # 1. 输入表示：词元嵌入 + 位置嵌入
    # 将输入序列中的每个 token 转换为嵌入向量、为每个位置添加位置编码
    x = wte[inputs] + wpe[range(len(inputs))]  # 输入形状从 [n_seq] 变为 [n_seq, n_embd]
    
    # 2. 通过 Transformer 解码器堆栈
    # 依次通过每个 Transformer 块，嵌入维度保持不变
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # 输入和输出形状都是 [n_seq, n_embd]
    
    # 3. 最终层归一化和输出投影
    # 投影到词汇表：对最终的隐藏状态应用层归一化、使用词元嵌入矩阵的转置将隐藏状态投影回词汇表空间
    x = layer_norm(x, **ln_f)  # 输入和输出形状都是 [n_seq, n_embd]
    
    # 输出形状从 [n_seq, n_embd] 变为 [n_seq, n_vocab]
    return x @ wte.T
```

让我们更详细地分解这三个部分。

### 嵌入(Embeddings)

![Hands-On Large Language Models Figure 3-5. The tokenizer has a vocabulary of 50,000 tokens. The model has token embeddings associated with those embeddings.](./README.assets/the_components_of_the_forward_pass_2.png)

**词元嵌入 (Word Token Embeddings)**

单独的 Token 编号无法有效表征神经网络中的语义信息。首先，Token 的相对大小会错误地传达信息(若词汇表中有 `Apple = 5` 和 `Table = 10`，但并不意味着 `2 * Apple = Table`)。其次，单个数字对于神经网络来说维度不够高，即信息容量有限。神经网络无法直接处理离散的符号，我们需要将这些离散符号转换为连续的数值向量，这个过程就是嵌入。

利用[词向量(Word vector)](https://jaykmody.com/blog/attention-intuition/#word-vectors-and-similarity)，通过学习嵌入矩阵，将离散的 token 转换为连续的向量表示、将单一数值扩展为高维向量，提供更丰富的特征表示：

| ![Hands-On Large Language Models Figure 2-7. A language model holds an embedding vector associated with each token in its tokenizer.](./README.assets/embedding_vector_associated.png) | ![Hands-On Large Language Models Figure 2-8. Language models produce contextualized token embeddings that improve on raw, static token embeddings.](./README.assets/improve_on_token_embeddings.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

```python
wte[inputs] # [n_seq] -> [n_seq, n_embd]
# n_vocab: 词汇表大小(50,257)
# n_embd: 嵌入维度(768)
```

`wte` 是一个 `[n_vocab, n_embd]` 矩阵。它充当查找表，矩阵中的第 i 行对应词汇表中的第 i 个 token 的向量表示。`wte[inputs]` 使用整数数组索引来检索与输入中的每个 token 相对应的向量。与神经网络中的其他参数一样，`wte` 是学习而来的。它在训练开始时随机初始化，然后通过梯度下降进行更新。

这些连续的向量表示捕捉了词元的语义信息：语义相似的词元在向量空间中距离较近，向量之间可以进行数学运算；这些向量作为模型的输入，使模型能够理解和处理自然语言。通过这种方式，GPT-2 模型将离散的、计算机易于处理的 token 转换为连续的、包含丰富语义信息的向量表示，为后续的神经网络处理奠定基础。

> 以下为一个  `vec("国王") - vec("男人") + vec("女人") ≈ vec("王后")` [示例](https://jalammar.github.io/illustrated-word2vec/)，这是“king”这个词的词嵌入(GloVe)：
>
> ```
> [ 0.50451 , 0.68607 , -0.59517 , -0.022801, 0.60046 , -0.13498 , -0.08813 , 0.47377 , -0.61798 , -0.31012 , -0.076666, 1.493 , -0.034189, -0.98173 , 0.68229 , 0.81722 , -0.51874 , -0.31503 , -0.55809 , 0.66421 , 0.1961 , -0.13495 , -0.11476 , -0.30344 , 0.41177 , -2.223 , -1.0756 , -1.0783 , -0.34354 , 0.33505 , 1.9927 , -0.04234 , -0.64319 , 0.71125 , 0.49159 , 0.16754 , 0.34344 , -0.25663 , -0.8523 , 0.1661 , 0.40102 , 1.1685 , -1.0137 , -0.21585 , -0.15155 , 0.78321 , -0.91241 , -1.6106 , -0.64426 , -0.51042 ]
> ```
>
> 这是一个包含 50 个数字的列表，我们可以可视化一下，以便与其他词向量进行比较：
>
> ![词元嵌入示例 1](./README.assets/word_token_embeddings_1.png)
>
> 我们可以对词向量进行加减运算，得到有趣的结果：
>
> ![词元嵌入示例 2](./README.assets/word_token_embeddings_2.png)

**位置嵌入 (Position Embeddings)**

Transformer 架构本身是位置无关的，如果我们随机打乱输入序列中词元的顺序，模型的输出可能保持不变，这意味着输入的顺序对输出没有影响。然而，在自然语言中，词的顺序显然是至关重要的。例如，"猫追狗"和"狗追猫"表达的是完全不同的意思，尽管使用了相同的词。因此，我们需要某种方式将位置信息编码到输入中。

原始 Transformer 论文和一些早期变体有绝对位置嵌入，本质上将第一个词元标记为位置 1，第二个标记为位置 2...等等。这些可以是静态方法(位置向量使用几何函数生成)或学习的(模型训练在学习过程中为它们分配值)。

> Rotary Position Embedding (RoPE) 是 2021 年 [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://huggingface.co/papers/2104.09864) 提出的技术，是在注意力计算中应用旋转变换，不需要额外的位置嵌入参数。

GPT-2 使用了一个可学习的位置嵌入矩阵，使用固定位置编码表，为每个位置(0 到 max_position - 1)学习一个独立的嵌入向量：

```python
wpe[range(len(inputs))] # [n_seq] -> [n_seq, n_embd]
# n_seq: 实际序列长度，但是不超过 n_ctx 最大上下文长度(1024)
# n_embd: 与词元嵌入相同的维度
```

```python
# 位置嵌入矩阵的结构
wpe = [
    [pos_0_dim_0, pos_0_dim_1, ..., pos_0_dim_767], # 位置0的编码
    [pos_1_dim_0, pos_1_dim_1, ..., pos_1_dim_767], # 位置1的编码
    [pos_2_dim_0, pos_2_dim_1, ..., pos_2_dim_767], # 位置2的编码
    # ...
    [pos_1023_dim_0, pos_1023_dim_1, ..., pos_1023_dim_767]  # 位置 1023 的编码
]

# 每一行都是 768 维的可学习向量
# 通过训练学习到位置的语义表示
```

`wpe` 是一个形状为 `[n_ctx, n_embd]` 的矩阵，矩阵的第 i 行包含一个向量，用于编码输入序列中第 i 个位置的信息，与词元嵌入矩阵类似，这个位置嵌入矩阵也是通过梯度下降学习得到的。

需要注意的是，这种方法限制了模型能处理的最大序列长度为 `n_ctx` 。这意味着必须满足条件： `len(inputs) <= n_ctx`。这是因为位置嵌入矩阵 `wpe` 只包含了 `n_ctx` 个不同位置的嵌入向量。如果输入序列长度超过 `n_ctx`，模型将无法为超出部分的位置提供有效的位置编码。在 GPT-2 中， `n_ctx` 通常设置为1024，这意味着模型一次最多可以处理 1024个 token 的序列。

**组合**

在 GPT-2 模型中，最终的输入表示是词元嵌入和位置嵌入的元素级相加：

```python
# 词元嵌入 + 位置嵌入
x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
# x[i] 将两个相同形状的矩阵对应元素相加，结果 x 的形状仍为 [n_seq, n_embd]
```

对于输入序列中的第 `i` 个词元，`x[i]` 包含了该词元的语义信息和它在序列中第 `i` 个位置的位置信息，即使相同的词元出现在不同位置，它们的最终表示也会不同模型能够理解'猫追狗'和'狗追猫'的区别，因为虽然词元相同，但位置嵌入不同。这种组合嵌入方法是 GPT-2 等 Transformer 模型成功处理序列数据的关键技术之一。

> 关于为什么用加法而不是其他方式的讨论：[r/MachineLearning 讨论](https://www.reddit.com/r/MachineLearning/comments/rfssk6/d_in_transformers_why_are_positional_embeddings/)、[TensorFlow Tensor2Tensor Issue](https://github.com/tensorflow/tensor2tensor/issues/1591) 等。尽管向量相加，这两个子空间仍然可以通过某个学习到的变换进行基本独立的操作。

### 解码器堆栈(Decoder Stack)

下述代码展示了 GPT-2 模型的核心实现部分，体现了深度学习中深度的本质——即网络层数的堆叠：

![Hands-On Large Language Models Figure 3-4. A Transformer LLM is made up of a tokenizer, a stack of Transformer blocks, and a language modeling head.](./README.assets/the_components_of_the_forward_pass_1.png)

```python
# 通过 n_layer 个 Transformer 解码器块的前向传播
for block in blocks: # blocks是一个包含 n_layer 个 Transformer 块参数的列表
    # 每个 block 包含注意力机制和前馈网络的参数
    # n_head 参数控制多头注意力机制中的头数
    x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]
```

> 这里的"深度"指的是神经网络中层的数量。在这个循环中，输入数据 x 依次通过多个 Transformer 解码器块，每个块都对数据进行一系列复杂的转换，然后将结果传递给下一个块。[这是不同 GPT-2 模型 Size 的主要区别因素之一](https://jalammar.github.io/illustrated-gpt2/)：
>
> ![The Illustrated GPT-2 How high can we stack up these blocks?](./README.assets/how_high_can_we_stack_up_these_blocks.png)

每个 Transformer 块包含四个关键组件：

1. 多头自注意力机制：捕捉序列中不同位置之间的依赖关系；
2. 前馈神经网络：对每个位置独立进行非线性变换；
3. 残差连接：缓解梯度消失问题，促进信息流动；
4. 层归一化：稳定训练过程，加速收敛。

堆叠更多的 Transformer 块(增加 n_layer 的值)可以：增强模型的表达能力、使模型能够学习更复杂的模式和关系、提高模型处理长距离依赖的能力。

随着深度(层数)和宽度(嵌入维度)的增加，模型的参数数量和计算复杂度也会显著增加。这就是为什么大型语言模型需要强大的计算资源进行训练和推理的原因。

总之，这个简单的循环是 GPT-2 模型能力的核心所在，通过堆叠多个 Transformer 块，使模型能够逐层构建对输入序列的深入理解，最终生成高质量的文本输出。

### 词汇投影(Projection to Vocab)

词汇投影是 GPT-2 模型的最终输出层，负责将高维语义表示转换为词汇表上的未归一化分数。这一步骤决定了模型生成下一个 token 的能力：

| ![Hands-On Large Language Models Figure 3-6. At the end of the forward pass, the model predicts a probability score for each token in the vocabulary.](./README.assets/the_components_of_the_forward_pass_3.png) | ![Hands-On Large Language Models Figure 3-8. Each token is processed through its own stream of computation (with some interaction between them in attention steps, as we’ll later see).](./README.assets/the_components_of_the_forward_pass_4.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

```python
# 最终层归一化
x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
# 使用词元嵌入矩阵的转置进行投影到词汇表空间
return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]
# 注意：这里返回的是 logits 而非概率分布
# 如需概率分布，可以应用 softmax: probs = softmax(logits, axis=-1)
```

其工作原理是：

1. 最终层归一化：在投影之前，对 Transformer 块的输出应用层归一化，这是 GPT-2 架构特有的设计(原始 GPT 和 Transformer 论文中没有)，层归一化有助于稳定深层网络的训练和推理；
2. 矩阵乘法投影：使用词元嵌入矩阵的转置(wte.T)进行投影，输入形状为 `[n_seq, n_embd]`，输出形状为 `[n_seq, n_vocab]` 每个位置的输出向量包含词汇表中每个 token 的未归一化分数。

其中，模型输出的是 Logits，而不是应用 Softmax 后的概率：

1. 贪婪采样的等效性：由于 Softmax 是单调函数， `np.argmax(logits)` 等同于 `np.argmax(softmax(logits))`，对于贪婪采样来说，应用 Softmax 是多余的；

2. 信息保留：Logits 包含更多信息，可以随时通过应用 Softmax 转换为概率，但从概率无法恢复回 Logits，因此输出 :ogits 提供了最大的灵活性；

3. 数值稳定性：在计算损失函数时，直接使用 logits 通常更稳定，`log(softmax(logits))` 可以用更稳定的 `log_softmax(logits)` 替代。

投影到词汇表的步骤有时被称为**语言建模头**(language modeling head)：

1. "头"的含义：模型可以有多个不同类型的输出层(头)；
2. 灵活性：预训练完成后，可以将语言建模头替换为其他类型的投影；
3. 迁移学习：例如，可以添加分类头用于特定任务的微调；
4. 多任务能力：就像神话中的九头蛇一样，模型可以有多个"头"来处理不同任务。

这种设计使 GPT-2 能够通过简单替换输出头来适应各种下游任务，而无需重新训练整个模型。

### 解码器堆栈的解码器块(Decoder Block)

Transformer 解码器块是 GPT-2 的基本构建单元，每个块包含两个关键子层，通过残差连接和层归一化实现信息的有效传递：

1. 多头因果自注意力机制(Multi-Head Causal Self-Attention)；

   - 这是唯一允许不同位置 token 交流信息的组件；
   - 因果性质确保模型能够进行自回归生成；
   - 多头机制支持并行计算，能够捕获多种依赖关系。

2. 逐位置的前馈神经网络(Position-wise Feed-Forward Network)。

   - 对每个位置独立进行非线性变换、两层全连接网络，中间使用 GELU 激活、增强模型的表达能力。

   > 在 Transformer 架构中，Multi-Layer Perceptron(MLP) 和 Position-wise Feed-Forward Network(PWFFN) 实际上指的是同一个组件。

| ![Hands-On Large Language Models Figure 3-11 The bulk of the Transformer LLM processing happens inside a series of Transformer blocks, each handing the result of its processing as input to the subsequent block.](./README.assets/a_series_of_transformer_blocks.png) | ![Hands-On Large Language Models Figure 3-12. A Transformer block is made up of a self-attention layer and a feedforward neural network.](./README.assets/a_transformer_block_made_up.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

让我们详细分析 `transformer_block` 函数的实现：

```python
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # Transformer 解码器块的实现
    # 1. 多头因果自注意力子层(带前置层归一化和残差连接)
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]
    # 2. 前馈神经网络子层(带前置层归一化和残差连接)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x
```

每个子层都在输入上使用了层归一化，也使用了残差连接，即将子层的输入直接连接到子层的输出。

前置层归一化(Pre-Norm)相比后置归一化有显著优势，[已被证明对提升 Transformer 的性能非常重要](https://huggingface.co/papers/2002.04745)。残差连接(Residual connections，由 [ResNet](https://huggingface.co/papers/1512.03385) 推广)有几个不同的作用，如梯度流优化、解决深度退化问题等。

#### 逐位置的前馈神经网络

前馈神经网络(Feed-Forward Network，FFN)是 Transformer 架构中负责信息存储和模式学习的核心组件，多头注意力矩进行升维、非线性过滤、然后再降回原来的维度。

[Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913) 中有“FFN 是一个 Key-Value 记忆网络”的结论。一个直观的解释：当我们向语言模型输入"The Shawshank"，期望它生成"Redemption"作为下一个词时，这种词汇关联知识主要存储在前馈网络的参数权重中。当模型在包含大量"The Shawshank Redemption"文本的语料库上训练时，前馈网络学习并编码了这种词汇共现模式。但是大语言模型不仅仅是一个巨型查找表——它需要在已见过的数据点之间进行插值和泛化，从而在未见过的输入上也能表现良好。

> 另一个关于前馈神经网络的比喻是： 思考空间 —— 多头因果自注意力机制帮助模型正确地分配注意力，逐位置的前馈神经网络帮助模型仔细地思考，提取更加抽象的特征。注意力机制专注于在 token 层级优化权重，在 token 之间建立丰富的联系，解决了序列中的长短程依赖问题；而逐位置的前馈神经网络专注于在特征层次优化权重，让不同特征之间相互融合，丰富局部的表现力。两者相辅相成，各自独立又互相配合。

![Hands-On Large Language Models Figure 3-13. The feedforward neural network component of a Transformer block likely does the majority of the model’s memorization and interpolation.](./README.assets/the_feedforward_neural_network_component.png)

Position-wise Feed-Forward Network 是 Feed-Forward Neural Network 的一个特殊应用形式，[Attention Is All You Need](https://arxiv.org/pdf/1706.03762) 使用 Position-wise Feed-Forward Network 作为标题。"逐位置的"意味着这个前馈网络独立地应用于序列中的每个位置。对于输入序列中的每个 token，都使用完全相同的前馈网络进行处理，且各个位置之间的处理是相互独立的。

这个函数实现了一个两层的前馈神经网络：

```python
def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd] 
    # 逐位置的前馈神经网络
    #
    # 1. 向上投影：扩展维度并引入非线性
    # 将输入 x 通过线性变换 linear(x, **c_fc) 映射到一个更高维度的空间
    # 输入维度从 [n_seq, n_embd] 变为 [n_seq, 4*n_embd] (维度扩大了4倍)
    # 然后应用 GELU 激活函数 gelu() 引入非线性
    a = gelu(linear(x, **c_fc))
    # 2. 向下投影：将维度压缩回原始大小
    # 将激活后的结果 a 通过另一个线性变换 linear(a, **c_proj) 映射回原始维度
    # 输出维度从 [n_seq, 4*n_embd] 变回 [n_seq, n_embd]
    # Position-wise 体现：linear() 函数对输入张量 x 的 每一行(每个位置)独立应用相同的线性变换
    x = linear(a, **c_proj)
    return x
```

这个 `ffn` 函数实现了 Transformer 架构中的位置式前馈神经网络，它通过先扩展维度、应用非线性激活函数，然后再压缩回原始维度的方式，增强了模型的表达能力。虽然结构简单，但它是 Transformer 模型中不可或缺的组成部分。

前馈网络通常将维度扩展到原来的 4 倍，回忆一下我们的 `params` 字典，我们的 `mlp` 参数如下：

```python
"mlp": {
    "c_fc"  : {"b": [4*n_embd], "w": [  n_embd, 4*n_embd]},
    "c_proj": {"b":   [n_embd], "w": [4*n_embd,   n_embd]}
}
```

| 参数名 | 子参数 | 形状               | 作用             |
| ------ | ------ | ------------------ | ---------------- |
| c_fc   | w      | [n_embd, 4*n_embd] | 向上投影权重矩阵 |
| c_fc   | b      | [4*n_embd]         | 向上投影偏置向量 |
| c_proj | w      | [4*n_embd, n_embd] | 向下投影权重矩阵 |
| c_proj | b      | [n_embd]           | 向下投影偏置向量 |

`params` 字典中的 mlp 参数直接对应了 `ffn` 函数中的线性变换参数，它们共同实现了 Transformer 架构中的位置式前馈神经网络组件。

#### 多头因果自注意力机制

这一层可能是 Transformer 中最难理解的部分。我们可以逐词来理解“多头因果自注意力”：自注意力(Self-Attention)、因果(Causal)、多头(Multi-Head)。

##### 自注意力(Self-Attention)

参考[示例](https://lilianweng.github.io/posts/2018-06-24-attention/)，人类的视觉注意力使我们能够聚焦于“高分辨率”的特定区域(例如黄色框中的尖耳朵)，同时以“低分辨率”感知周围的图像(例如雪景背景和服装的细节)。同样，我们可以用一个句子或一个紧密相关的语境来解释单词之间的关系。当我们看到“eating”时，我们预计很快就会遇到一个与食物相关的词。而“green”与“eating”并不直接相关：

| ![Attention? Attention! Figure 1 A Shiba Inu in a men’s outfit](./README.assets/attention_attention_1.png) | ![Attention? Attention! Figure 2 One word "attends" to other words in the same sentence differently.](./README.assets/attention_attention_2.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

根据 [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) 介绍，Seq2Seq 模型的固定长度上下文向量设计的一个关键且明显的缺点是无法记住长句子，它通常在处理完整个输入后就忘记了第一部分。注意力机制的诞生是为了帮助神经机器翻译([Neural Machine Translation，NMT](https://arxiv.org/pdf/1409.0473.pdf))记住较长的源语句。注意力机制允许模型关注输入序列中的特定部分，而不是平等地处理所有输入。在 Transformer 中，这使得模型能够捕捉序列中的长距离依赖关系和复杂模式。

“The animal didn't cross the street because it was too tired” —— 这句话中的 “it” 指代什么？是动物还是街道？对人类来说，这是一个简单的问题，但对算法来说却并非易事。当模型处理 “it” 这个词时，[自注意力机制](https://jalammar.github.io/illustrated-transformer/)让模型将 “it” 与 “animal” 联系起来：

![As we are encoding the word "it" in encoder #5 (the top encoder in the stack), part of the attention mechanism was focusing on "The Animal", and baked a part of its representation into the encoding of "it".](./README.assets/self_attention_it.png)

以下是注意力机制的一个简化框架：一个输入序列和一个正在处理的当前位置。我们主要关注的是当前位置，图中展示了一个输入向量和一个输出向量，输出向量根据注意力机制整合了序列中先前元素的信息：

| ![Hands-On Large Language Models Figure 3-15. A simplified framing of attention: an input sequence and a current position being processed. As we’re mainly concerned with this position, the figure shows an input vector and an output vector that incorporates information from the previous elements in the sequence according to the attention mechanism.](./README.assets/a_simplified_framing_of_attention.png) | ![Hands-On Large Language Models Figure 3-16. Attention is made up of two major steps: relevance scoring for each position, then a step where we combine the information based on those scores.](./README.assets/attention_is_made_up_of_two_major_steps.png) | ![Hands-On Large Language Models Figure 3-17. We get better LLMs by doing attention multiple times in parallel, increasing the model’s capacity to attend to different types of information.](./README.assets/attention_multiple_times_in_parallel.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |

注意力机制主要涉及两个步骤：

1. 对每个先前输入的 token 与当前正在处理的 token (粉色箭头所示)的相关性进行评分；
2. 利用这些分数，我们将来自不同位置的信息组合成一个输出向量。

为了赋予 Transformer 更广泛的注意力能力，注意力机制被复制并并行执行多次，每个注意力头独立执行一次注意力计算。这提升了模型对输入序列中需要同时关注的复杂内容进行建模的能力，但我们首先关注单个注意力头。其他注意力头的计算过程相同，但各自使用各自的投影矩阵。

该层的输入包括：当前位置 token 的向量表示、先前 token 的向量表示。目标是生成当前 token 的新表示，该表示融合了先前 token 的相关信息。

训练过程会生成三个投影矩阵：查询(Query)投影矩阵、键(Key)投影矩阵、值(Value)投影矩阵：

| ![Hands-On Large Language Models Figure 3-18. Before starting the self-attention calculation, we have the inputs to the layer](./README.assets/before_the_attention_calculations_start.png) | ![Hands-On Large Language Models Figure 3-19. Attention is carried out by the interaction of the queries, keys, and values matrices. Those are produced by multiplying the layer’s inputs with the projection](./README.assets/attention_is_carried_out_by_the_interaction.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

注意力机制首先将输入乘以投影矩阵 `Q = X @ W_q`、 `K = X @ W_k` 、`V = X @ W_v`，从而创建三个新的矩阵：查询(Query)矩阵、键(Key)矩阵和值(Value)矩阵。这三个矩阵是输入信息在不同表示空间中的投影，有助于执行注意力机制的两个步骤：1. 相关性评分、2. 信息合并。

在 Transformer 推理阶段，我们一次生成一个 token。这意味着我们一次每次仅处理序列中的一个位置。因此，这里的注意力机制只关注这一个位置，以及如何提取来自其他位置的信息来指导这个位置。

对于特定位置计算，注意力机制的相关性评分步骤是通过将当前位置的 Q 向量与 K 矩阵的转置相乘来实现的。这将产生一个分数，表示特定位置与每个先前 token 的相关性。将其通过 Softmax 运算进行归一化，使这些分数的总和为 1。

现在我们有了相关性分数，我们用每个 token 的相关性分数加权其对应的 Value 向量。将这些得到的向量相加，就得到了此注意力机制步骤的输出。

| ![Hands-On Large Language Models Figure 3-20. Scoring the relevance of previous tokens is accomplished by multiplying the query associated with the current position with the keys matrix.](./README.assets/scoring_the_relevance_of previous_tokens_is_accomplished.png) | ![Hands-On Large Language Models Figure 3-21. Attention combines the relevant information of previous positions by multiplying their relevance scores by their respective value vectors.](./README.assets/attention_combines_the_relevant_information.png) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

这个函数实现了缩放点积注意力机制：
$$
\text{attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

```python
def attention(q, k, v):  # [n_q, d_k], [n_k, d_k], [n_k, d_v] -> [n_q, d_v]
    # 计算缩放点积注意力
    # q: 查询矩阵，决定"关注什么"
    # k: 键矩阵，决定"被关注的内容"
    # v: 值矩阵，决定"传递的信息"
    return softmax(q @ k.T / np.sqrt(q.shape[-1])) @ v
```

> Q、K、V 是通过将输入序列的每个位置的嵌入向量，分别与三个不同的线性投影矩阵相乘得到的，这些投影矩阵是模型训练过程中学习到的参数。

计算步骤 ：

- `q @ k.T`：计算查询和键的点积，得到注意力分数矩阵，形状为 [n_q, n_k]；
- `/np.sqrt(q.shape[-1])` ：除以键维度的平方根进行缩放，这是为了防止点积值过大导致 Softmax 梯度消失；
- `softmax(...)`：对缩放后的注意力分数应用 softmax 函数，得到注意力权重；
- `... @ v`：用注意力权重对值进行加权求和，得到最终的注意力输出，形状为 [n_q, d_v]。

这个 `attention` 函数是 Transformer 架构的核心，它通过计算查询与键的相似度，并用这些相似度对值进行加权，实现了序列中不同位置之间的信息交流，这种机制是现代大型语言模型强大能力的基础。

当查询(Q)、键(K)和值(V)都来自同一个源时，我们执行的是自注意力(让输入序列关注自身)，自注意力机制的核心思想是：让序列中的每个位置都能“看到”并“关注”序列中的所有其他位置，从而动态地学习词汇之间的关联关系：

```python
def self_attention(x): # [n_seq, n_embd] -> [n_seq, n_embd] 
    return attention(q=x, k=x, v=x)
```

上述 `q = k = v = x` 这样做有局限性：所有的 Q、K、V 都是相同的，模型无法学习到针对不同用途的最优表示。

我们可以通过为 Q、K、V 和注意力输出引入投影矩阵来增强自注意力机制：

```python
def self_attention(x, w_k, w_q, w_v, w_proj): # [n_seq, n_embd] -> [n_seq, n_embd] 
    # 通过投影矩阵让模型学习最优的 Q、K、V 表示
    
    # Q、K、V 投影
    q = x @ w_q # [n_seq, n_embd] @ [n_embd, n_embd] -> [n_seq, n_embd] 
    k = x @ w_k # [n_seq, n_embd] @ [n_embd, n_embd] -> [n_seq, n_embd] 
    v = x @ w_v # [n_seq, n_embd] @ [n_embd, n_embd] -> [n_seq, n_embd] 
    
    # 执行自注意力
    x = attention(q, k, v) # [n_seq, n_embd] -> [n_seq, n_embd] 
    
    # 输出投影
    x = x @ w_proj # [n_seq, n_embd] @ [n_embd, n_embd] -> [n_seq, n_embd] 
    
    return x 
```

这使我们的模型能够学习 Q、V、V 的映射，以最好地帮助注意力区分输入之间的关系。

我们可以通过合并 Q、K、V 投影，将矩阵乘法从 4 次减少到 2 次( 1 次 Q、K、V 合并投影 + 1次输出投影)，将 w_q、w_k 和 w_v 合并为单个矩阵 w_fc，执行投影，然后拆分结果：

```python
def self_attention(x, w_fc, w_proj): # [n_seq, n_embd] -> [n_seq, n_embd] 
    # 合并 Q、K、V 投影：4 次矩阵乘法 → 2 次矩阵乘法
    
    # Q、K、V 合并投影
    x = x @ w_fc # [n_seq, n_embd] @ [n_embd, 3*n_embd] -> [n_seq, 3*n_embd] 
    
    # 拆分为独立的 Q、K、V 矩阵
    q, k, v = np.split(x, 3, axis=-1) # [n_seq, 3*n_embd] -> 3个 [n_seq, n_embd] 
    
    # 执行自注意力
    x = attention(q, k, v) # [n_seq, n_embd] -> [n_seq, n_embd] 
    
    # 输出投影
    x = x @ w_proj # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd] 
    
    return x 
```

这更高效，因为现代 GPU 可以更好地利用一个大型矩阵乘法，而不是顺序发生的 3 个单独的小矩阵乘法。

最后，我们添加偏置向量、使用线性函数以符合 GPT-2 的实现，并重命名参数以匹配我们的 `params` 字典：

```python
def self_attention(x, c_attn, c_proj): # [n_seq, n_embd] -> [n_seq, n_embd] 
    # Q、K、V 投影
    x = linear(x, **c_attn) # [n_seq, n_embd] -> [n_seq, 3*n_embd] 
    
    # 拆分为独立的 Q、K、V 矩阵
    q, k, v = np.split(x, 3, axis=-1) # [n_seq, 3*n_embd] -> 3个 [n_seq, n_embd] 
    
    # 执行自注意力
    x = attention(q, k, v) # [n_seq, n_embd] -> [n_seq, n_embd] 
    
    # 输出投影
    x = linear(x, **c_proj) # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd] 
    
    return x 
```

```python
# GPT-2参数命名约定
"attn": { 
    "c_attn": {  # Combined attention: 合并的 QKV 投影
        "b": [3*n_embd],        # 偏置: [Q偏置|K偏置|V偏置]
        "w": [n_embd, 3*n_embd] # 权重: [输入→Q|输入→K|输入→V]
    }, 
    "c_proj": { # Combined/Output projection: 输出投影
        "b": [n_embd],          # 输出偏置
        "w": [n_embd, n_embd]   # 输出权重
    },
}

# 参数量计算：
# c_attn: (512 * 1536) + 1536 = 787,968 + 1536 = 789,504
# c_proj: (512 * 512) + 512 = 262,144 + 512 = 262,656
# 总计: 1,052,160 参数
```

这种实现方式既高效又符合 GPT-2 的原始设计，通过合并矩阵乘法和添加适当的偏置，优化了计算过程。

##### 因果(Causal)

我们当前的自注意力设置存在一个问题：输入可以"看到"未来。例如，如果我们的输入是["not", "all", "heroes", "wear", "capes"]，在自注意力计算中，我们允许"wear"看到"capes"。这意味着"wear"的输出概率会有偏差，因为模型已经知道正确答案是"capes"。

因果(Causal)指的是严格的时间顺序约束：原因必须在结果之前发生。在语言模型中，这意味着：

1. 位置 i 的词只能依赖位置 ≤ i 的词；
2. 未来的信息不能影响当前的预测；

这确保了模型学习真正的语言规律，而非简单的"作弊"。

为了防止这种情况，我们需要修改注意力矩阵，使输入无法看到未来。例如，假设我们的注意力矩阵如下，其中注意力权重矩阵 A[i,j] 表示位置 i 对位置 j 的关注度：

```python
       not    all    heroes wear   capes 
   not 0.116  0.159  0.055  0.226  0.443 
   all 0.180  0.397  0.142  0.106  0.175 
heroes 0.156  0.453  0.028  0.129  0.234 
  wear 0.499  0.055  0.133  0.017  0.295 
 capes 0.089  0.290  0.240  0.228  0.153 
```

每行对应一个查询(query)，列对应一个键(key)。在这个例子中，查看"wear"那一行，你可以看到它在最后一列对"capes"的注意力权重为0.295。为了防止这种情况，我们希望将该条目设置为0.0：

```python
       not    all    heroes wear   capes 
   not 0.116  0.159  0.055  0.226  0.443 
   all 0.180  0.397  0.142  0.106  0.175 
heroes 0.156  0.453  0.028  0.129  0.234 
  wear 0.499  0.055  0.133  0.017  0. 
 capes 0.089  0.290  0.240  0.228  0.153 
```

一般来说，为了防止输入中的所有查询看到未来，我们将所有位置 j>i 的值设为 0：

```python
       not    all    heroes wear   capes 
   not 0.116  0.     0.     0.     0. 
   all 0.180  0.397  0.     0.     0. 
heroes 0.156  0.453  0.028  0.     0. 
  wear 0.499  0.055  0.133  0.017  0. 
 capes 0.089  0.290  0.240  0.228  0.153 
```

我们称这为**掩码**(Masking)：

![Hands-On Large Language Models Figure 3-24. Attention figures show which token is being processed, and which previous tokens an attention mechanism allows it to attend to.](./README.assets/attention_figures_show_which_token_is_being_processed.png)

上述掩码方法有一个问题：每行的和不再为 1(因为我们在 Softmax 应用后将它们设置为 0)。为了确保行仍然总和为 1，我们需要在应用 Softmax 之前修改注意力矩阵。这可以通过在 Softmax 之前将要掩码的条目设置为 -∞ 来实现：

```python
def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v] 
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v 
```

其中mask是矩阵(对于 n_seq=5)：

```python
0 -1e10 -1e10 -1e10 -1e10 
0   0   -1e10 -1e10 -1e10 
0   0     0   -1e10 -1e10
0   0     0     0   -1e10 
0   0     0     0     0
```

我们使用 -1e10 而不是 -np.inf，因为 -np.inf 可能导致 NaN 值。将 mask 添加到我们的注意力矩阵而不是直接将值设置为 -1e10 是有效的，因为实际上，任何数加上 -∞ 仍然是 -∞。我们可以用 NumPy 计算掩码矩阵：` (1 - np.tri(n_seq)) * -1e10`。

将所有内容放在一起，我们得到：

```python
def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v] 
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v 

def causal_self_attention(x, c_attn, c_proj): # [n_seq, n_embd] -> [n_seq, n_embd]
    # 因果自注意力：确保每个位置只能关注之前的位置
    # 输入：
    # x: 输入序列 [n_seq, n_embd]
    # c_attn: QKV投影参数 {"w": [n_embd, 3*n_embd], "b": [3*n_embd]}
    # c_proj: 输出投影参数 {"w": [n_embd, n_embd], "b": [n_embd]}
    # mask_value: 掩码值，默认-1e10
    # 返回：
    # 输出序列 [n_seq, n_embd]
    
    # 1. qkv 投影：将输入映射到查询、键、值空间
    x = linear(x, **c_attn) # [n_seq, n_embd] -> [n_seq, 3*n_embd] 

    # 2. 拆分为 qkv：每个都是 [n_seq, n_embd]
    q, k, v = np.split(x, 3, axis=-1) # [n_seq, 3*n_embd] -> 3个 [n_seq, n_embd] 

    # 3. 构造因果掩码：上三角部分设为 mask_value
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq] 

    # 4. 计算注意力：包含因果约束
    x = attention(q, k, v, causal_mask) # [n_seq, n_embd] -> [n_seq, n_embd] 

    # 5. 输出投影：整合注意力结果
    x = linear(x, **c_proj) # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd] 

    return x
```

##### 多头(Multi-Head)

想象一下阅读理解：单头注意力就像只用一种思路理解文章，可能会遗漏重要信息；多头注意力就像同时用多种角度分析文章，比如语法角度、语义角度、情感角度等。若把多头注意力想象成一个专家团队，其核心思想：每个专家负责不同的任务，最后把所有专家的意见综合起来。

![Hands-On Large Language Models Figure 3-26. Attention is conducted using matrices of queries, keys, and values. In multi-head attention, each head has a distinct version of each of these matrices.](./README.assets/in_multi_head_attention.png)

多头注意力的核心是将高维空间分解为多个低维子空间。

我们可以通过执行 n_head 个独立的注意力计算来进一步改进我们的实现，将查询、键和值分割成多个头：

```python
def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd] 
    # 多头因果自注意力机制
    # 输入：
    # x: 输入序列 [n_seq, n_embd]
    # c_attn: QKV投影参数 {"w": [n_embd, 3*n_embd], "b": [3*n_embd]}
    # c_proj: 输出投影参数 {"w": [n_embd, n_embd], "b": [n_embd]}
    # n_head: 注意力头数，必须能整除 n_embd
    
    # 返回
    # 输出序列 [n_seq, n_embd]
    
    # QKV 投影：生成所有头的查询、键、值
    # [n_seq, n_embd] -> [n_seq, 3*n_embd] 
    x = linear(x, **c_attn)
    # [n_seq, 3*n_embd] -> 3个 [n_seq, n_embd]
    qkv = np.split(x, 3, axis=-1)
    
    # 拆分为多头
    # 3个 [n_seq, n_embd] -> 3个 [n_head, n_seq, n_embd/n_head]
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))
    
    # 构造因果掩码，所有头共享
    # [n_seq, n_seq] 
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10

    # 对每个头执行注意力计算
    # -> [n_head, n_seq, n_embd/n_head] 
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    
    # 合并多头结果 
    # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd] 
    x = np.hstack(out_heads)
    
    # 输出投影
    # [n_seq, n_embd] -> [n_seq, n_embd] 
    x = linear(x, **c_proj)
    
    return x 
```

这里主要添加了三个步骤：

1. 将q、k、v拆分为 n_head 个头 ：

```python
# 拆分为多头
# 3 个 [n_seq, n_embd] -> 3个 [n_head, n_seq, n_embd/n_head]
# 输入：qkv = [Q, K, V]
# 输出：qkv_heads = [[Q1,Q2,Q3], [K1,K2,K3], [V1,V2,V3]]
qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))
# qkv 是一个包含3个矩阵的列表： [Q, K, V]，每个矩阵形状： [n_seq, n_embd]
# ambda x: np.split(x, n_head, axis=-1) ：
# 	x 代表 Q、K、V 中的每一个矩阵
#   np.split(x, n_head, axis=-1) 在最后一个维度上分割
```

2. 为每个头计算注意力 ：

```python
# 对每个头执行注意力计算 
# -> [n_head, n_seq, n_embd/n_head] 
out_heads = [attention(q, k, v) for q, k, v in zip(*qkv_heads)]
# 遍历每个头的 Q、K、V 组合，对每个头独立计算注意力，收集所有头的结果
```

3. 合并每个头的输出 ：

```python
# 合并多头结果
# [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd] 
x = np.hstack(out_heads)
# hstack 表示水平堆叠，在最后一个维度上连接所有矩阵
```

上述实现为顺序执行，实际应用中建议并行计算以提升效率。为了简单起见，我们保留这种顺序执行的方式。

至此，我们终于完成了GPT的实现。现在，就是把所有内容放在一起并运行代码。

## 3.6 运行 GPT

将所有内容整合在一起，我们得到了 `gpt2.py`，整个文件仅有 120 行代码，去除注释和空白行后仅 60 行代码。

我们可以通过以下命令测试我们的实现：

```bash
python gpt2.py \
    "Alan Turing theorized that computers would one day become" \
    --n_tokens_to_generate 8
```

示例输出如下：

```
the most powerful machines on the planet.
```

# 四、接下来做什么

1. GPU/TPU 支持：如将 `import numpy as np` 替换为 `import jax.numpy as np` 即可获得硬件加速能力。
2. 反向传播(Backpropagation)：如使用 `jax.grad(loss_fn)(params)` 自动计算梯度，无需手动实现反向传播。
3. 批处理(Batching)：如通过 `jax.vmap(gpt2, in_axes=[0, None, ...])` 实现自动批处理，提升训练效率。
4. 推理优化(Inference Optimization)：实现 KV 缓存和并行注意力头计算是最重要的性能优化，详见 [Transformer 推理优化](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)。
5. 训练(Training)：真正的挑战在于数据和模型的规模化，参考 [大模型训练指南](https://lilianweng.github.io/posts/2021-09-25-train-large/) 了解分布式训练。
6. 评估(Evaluation)：使用 [HELM](https://arxiv.org/abs/2211.09110) 等综合基准测试，但需对评估指标保持批判性思维。
7. 架构改进(Architecture Improvements)：探索 [X-Transformers](https://github.com/lucidrains/x-transformers) 了解最新的 Transformer 架构研究。
8. 停止生成(Stopping Generation)：引入 EOS 标记控制生成长度，避免固定 token 数量的限制。
9. 微调(Fine-tuning)：
   1. 分类微调：替换语言建模头为分类头，使用最后一个 token 的输出进行分类；
   2. 指令微调：在人工标记的指令-完成对上训练，提升模型的指令遵循能力和实用性；
   3. 参数高效微调(PEFT)：使用 [Adapters](https://huggingface.co/papers/1902.00751) 等方法，仅训练少量参数即可获得接近全量微调的效果。

# 五、其他推荐内容

1. [Transformers (how LLMs work) explained visually | DL5](https://www.youtube.com/watch?v=wjZofJX0v4M) - 3Blue1Brown
2. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar
3. [Hands-On Large Language Models](https://www.oreilly.com/library/view/hands-on-large-language/9781098150952/) - Jay Alammar
4. [Transformer Explainer: LLM Transformer Model Visually IExplained](https://poloclub.github.io/transformer-explainer/) - Georgia Institute of Technology
5. [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) - Stanford University
