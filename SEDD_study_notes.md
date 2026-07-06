# SEDD Study Notes

这份总笔记把当前目录下的几份 SEDD 学习笔记与 README 内容合并成一份更适合连续阅读的主文档。原始文件都保留不动，这份文档相当于一个“总入口”。

## Merged From

- [README.md](./README.md)
- [continuous_vs_discrete_score_notes.md](./continuous_vs_discrete_score_notes.md)
- [loglinear_and_dsigma_notes.md](./loglinear_and_dsigma_notes.md)
- [sigma_time_embedding_notes.md](./sigma_time_embedding_notes.md)
- [transformer_block_attention_notes.md](./transformer_block_attention_notes.md)
- [graph_dim_and_output_semantics.md](./graph_dim_and_output_semantics.md)
- [absorbing_diffusion_interview_notes.md](./absorbing_diffusion_interview_notes.md)

## 1. 仓库概览

SEDD 是论文 [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834) 的 PyTorch 实现。这个仓库的代码结构比较模块化，阅读时最关键的几个入口是：

- `noise_lib.py`：噪声日程 $\sigma(t)$ 与 $\frac{d\sigma(t)}{dt}$
- `graph_lib.py`：离散前向扩散图、前向转移、reverse 相关量、score entropy
- `sampling.py`：Euler / analytic predictor、反向采样
- `model/transformer.py`：SEDD / DDiT 主模型
- `losses.py`：训练时怎么采样 $t$、构造 $x_\sigma$、计算 loss
- `train.py` / `run_train.py`：原始预训练入口
- `train_sft.py` / `eval_sft.py`：监督微调原型

从阅读顺序上，比较推荐：

1. 先理解“连续 score vs 离散 ratio score”
2. 再理解 $\sigma / dsigma$ 和前向扩散
3. 再看模型如何把 $\sigma$ 注入到 Transformer
4. 最后看 sampling 和 absorbing 图细节

## 2. 连续空间与离散空间：SEDD 到底在学什么

### 2.1 连续空间里学的是什么

在连续扩散里，时刻 $t$ 的 noisy distribution 是：

$$
p_t(x)
$$

对应的 score 是：

$$
\nabla_x \log p_t(x)
$$

它表示：

> 在当前噪声层级下，如果把点 $x$ 稍微移动一点，往哪个方向走会让密度上升得最快。

因此，连续空间里的 reverse diffusion 本质上是由这个 score direction field 决定的。

### 2.2 连续 Fisher divergence 是什么

给真实分布 $p$ 和模型分布 $q$，连续 Fisher divergence 通常写成：

$$
D_F(p \,\|\, q)=\frac12 \mathbb{E}_{x \sim p}\left[\left\|\nabla_x \log p(x) - \nabla_x \log q(x)\right\|^2\right]
$$

它比较的不是概率值本身，而是两个分布的 score function 是否接近。连续 score matching 的核心思想也正是：不直接学 $p(x)$，而是学 $\nabla_x \log p(x)$。

### 2.3 为什么离散空间里不能直接照搬

在离散 token 空间里，状态不是连续变量，所以不能定义：

$$
\nabla_x \log p_t(x)
$$

因为 token 没有“无穷小位移”的概念。于是 SEDD 改学的是状态间的概率比值：

$$
\frac{p_t(y)}{p_t(x)}
$$

这里：

- $x$ 是当前离散状态
- $y$ 是某个候选状态

它表示：

> 当前在状态 $x$ 时，候选状态 $y$ 相对 $x$ 有多合理。

### 2.4 为什么这个比值是自然的离散替代物

因为离散 reverse CTMC 里真正需要的就是：

$$
\bar Q_t(y,x)\propto Q_t(x,y)\frac{p_t(y)}{p_t(x)}
$$

所以在连续空间里，reverse 过程靠 $\nabla_x \log p_t(x)$ 定义；
在离散空间里，reverse 过程靠 $p_t(y)/p_t(x)$ 定义。

### 2.5 `score entropy` 的定位

因此，SEDD 里的 `score entropy` 可以看作：

> 连续空间 score matching / Fisher divergence 思想在离散扩散上的对应版本。

但它不是把连续公式直接离散化一下，而是：

1. 连续空间匹配 score
2. 离散空间没有导数
3. reverse CTMC 需要的是状态间比值
4. 所以重新构造了一个离散版 proper loss 去匹配 ratio score

## 3. 前向扩散、$\sigma$、$dsigma$

### 3.1 前向扩散主方程

SEDD 的前向过程是连续时间马尔可夫链：

$$
\frac{dp_t}{dt}=g(t)\,Q\,p_t
$$

这里：

- $p_t$：时刻 $t$ 的分布
- $Q$：基础离散转移图的 generator / rate matrix
- $g(t)$：噪声速率

这句话的意思是：

> 时刻 $t$ 的分布变化率，由基础离散转移图 $Q$ 作用在当前分布 $p_t$ 上得到，再由噪声速率 $g(t)$ 控制快慢。

### 3.2 为什么会出现 $\sigma$

定义累计噪声：

$$
\sigma(t)=\int_0^t g(s)\,ds
$$

那么前向分布可以写成：

$$
p_t = e^{\sigma(t)Q}p_0
$$

所以真正决定“当前已经脏到什么程度”的是累计噪声 $\sigma$，不是 $t$ 本身。$t$ 更像是 schedule 的参数化坐标，$\sigma$ 则更像实际累计噪声量。

### 3.3 公式在代码里的对应位置

这件事在代码里最直接体现在：

- `graph.transition(i, sigma)`：返回 $e^{\sigma Q}$ 作用在初始状态 $i$ 上的分布
- `graph.sample_transition(i, sigma)`：从该分布采样
- `losses.py` 里：

```python
perturbed_batch = graph.sample_transition(batch, sigma[:, None])
```

这就是从 $x_0$ 直接采样 $x_\sigma$。

### 3.4 $\sigma$ 和 $dsigma$ 分别是什么

在这份实现里，`noise(t)` 返回两个量：

- $\sigma$：累计噪声强度
- $dsigma$：累计噪声对 $t$ 的导数，也就是瞬时噪声速率

也就是：

$$
\sigma = \bar\sigma(t), \qquad dsigma = \frac{d\bar\sigma(t)}{dt}
$$

你可以把它们记成：

- $\sigma$：当前已经脏到什么程度
- $dsigma$：这一时刻噪声增长得有多快

### 3.5 为什么前向转移用 $\sigma$，loss 又要乘 $dsigma$

前向转移用 $\sigma$，因为分布只由累计噪声决定：

$$
p_t = e^{\sigma(t)Q}p_0
$$

loss 里乘 $dsigma$，是因为训练时按 $t$ 采样，但更自然的理论目标是对噪声层级积分：

$$
\int \ell(\sigma)\,d\sigma
$$

而：

$$
d\sigma = \frac{d\sigma}{dt}dt = dsigma \cdot dt
$$

所以代码里会出现：

```python
loss = (dsigma[:, None] * loss).sum(dim=-1)
```

## 4. `loglinear` 噪声为什么这样定义

`LogLinearNoise` 定义的是：

$$
\sigma(t)= -\log(1-(1-\varepsilon)t)
$$

以及：

$$
\frac{d\sigma(t)}{dt} = \frac{1-\varepsilon}{1-(1-\varepsilon)t}
$$

它叫 `loglinear`，是因为：

$$
e^{-\sigma(t)} = 1-(1-\varepsilon)t
$$

右边对 $t$ 是线性的。

对于 absorbing 图，token 被 mask 的概率是：

$$
1-e^{-\sigma(t)}=(1-\varepsilon)t
$$

于是：

- $t=0$ 时几乎不 mask
- $t=1$ 时 mask 概率接近 $1-\varepsilon$
- mask 概率随时间线性增长

这就是它特别适合 `absorb` 图的原因。

## 5. $\sigma$ 是怎么进入模型的

### 5.1 $\sigma$ 不是每个 token 一个，而是每条样本一个

模型前向定义是：

```python
def forward(self, indices, sigma):
```

其中：

- `indices.shape = [B, L]`
- `sigma.shape = [B]`

也就是说，$\sigma$ 是样本级条件，不是 token 级条件。整条序列共用同一个噪声层级。

### 5.2 `time embedding` 其实更准确叫 `sigma embedding`

虽然代码类名叫 `TimestepEmbedder`，但实际喂进去的是 $\sigma$，不是离散 timestep id：

```python
self.sigma_map = TimestepEmbedder(config.model.cond_dim)
...
c = F.silu(self.sigma_map(sigma))
```

所以在这份实现里，更准确的叫法是：

- `noise-level embedding`
- `sigma embedding`

### 5.3 为什么模型需要它

同一个 noisy token，在不同 $\sigma$ 下含义不一样。

例如当前某个位置是 `MASK`：

- 如果 $\sigma$ 很大，说明噪声很重，`MASK` 很正常
- 如果 $\sigma$ 很小，说明快去噪完了，这个 `MASK` 就很异常，模型应该更积极恢复真实 token

因此模型必须知道当前噪声层级。

## 6. 模型结构：SEDD / DDiT 主干

### 6.1 三类信息源

模型里最重要的三类输入信号是：

1. `token embedding`：当前每个位置是什么 token
2. `rotary position embedding`：当前 token 在序列哪个位置
3. `sigma embedding`：当前这整条序列处在哪个噪声层级

### 6.2 DDiTBlock 的核心结构

标准 pre-LN Transformer attention 分支大致是：

$$
h = x + W_o \,\mathrm{Attn}(\mathrm{LN}(x))
$$

SEDD / DDiT block 更接近：

$$
h = x + g_{att}\odot W_o\,\mathrm{Attn}(\mathrm{modulate}(\mathrm{LN}(x), shift, scale))
$$

所以它比普通 LLM 多了两层条件调制：

- attention 前，对 $\mathrm{LN}(x)$ 做 `shift/scale`
- attention 输出回残差前，再乘一个 `gate`

### 6.3 `shift / scale / gate` 分别在干什么

- `shift`：平移归一化后的通道值
- `scale`：放大或缩小归一化后的通道值
- `gate`：控制整条 attention / MLP 分支往 residual 里注入多少

它们都由：

```python
adaLN_modulation(c)
```

从 `sigma embedding` 生成。

### 6.4 为什么是先 `LayerNorm` 再 modulate

因为先标准化 hidden states，再做由 $\sigma$ 决定的仿射调制，会更稳定、更可控。

所以：

```python
x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
```

的含义是：

> 在 attention 前，按当前噪声层级去重新缩放和平移每个 token 的 hidden representation。

## 7. `graph.dim = 50258` 到底是什么意思

在当前 absorbing 图配置里：

- GPT-2 词表大小：`50257`
- absorbing 图会额外增加一个 `MASK` 状态

所以：

$$
\text{graph.dim} = 50257 + 1 = 50258
$$

这 `50258` 个状态分别是：

- `0 ~ 50256`：真实 token
- `50257`：额外的 `MASK`

模型输出层也必须对齐这个状态空间，所以每个位置最后输出：

$$
\text{log\_score} \in \mathbb{R}^{50258}
$$

## 8. 为什么输出不是普通 logits，而是 `log_score`

### 8.1 输出头本身只是参数化

`DDitFinalLayer` 本身只是把 hidden 映射到所有候选状态维度：

$$
\mathbb{R}^{hidden}\rightarrow \mathbb{R}^{|\mathcal X|}
$$

单看这一层，它还只是“状态分数头”，不是概率、也不是 softmax logits。

### 8.2 为什么它会被解释成 `log_score`

关键在三件事一起成立：

1. 模型输出覆盖图上的所有候选状态
2. 当前 noisy state 那一维会被钉成 $0$
3. loss 和 sampling 都把它按 ratio score 来使用

尤其是这句：

```python
x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
```

作用是：

> 对每个位置，把当前 noisy token 对应的那一维强行设成 0。

这相当于把当前状态选成 reference state，于是其他维就可以解释成：

$$
\log s_\theta(y)-\log s_\theta(x)
$$

也就是相对于当前 noisy state 的 log-ratio。

### 8.3 为什么这不再适合普通 cross-entropy

因为它已经不是一组“自由竞争的分类 logits”，而是：

> 相对于当前 noisy state 的 log-score

采样时也不是走 softmax，而是直接：

```python
return score.exp()
```

把它转成正的 ratio score。

## 9. reverse 后验和采样在代码里怎么写

如果把当前 noisy 状态记为 $x$，把上一层更干净的候选状态记为 $z$，那么一步反向采样需要的分布可以写成：

$$
p_{\sigma-\Delta\sigma}(z \mid x_\sigma=x)\propto p_{\Delta\sigma}(x \mid z)\,p_{\sigma-\Delta\sigma}(z)
$$

除以和 $z$ 无关的 $p_\sigma(x)$ 后，可以写成：

$$
p_{\sigma-\Delta\sigma}(z \mid x_\sigma=x)\propto \underbrace{\frac{p_{\sigma-\Delta\sigma}(z)}{p_\sigma(x)}}_{\text{staggered score}}\,\underbrace{p_{\Delta\sigma}(x \mid z)}_{\text{transpose transition}}
$$

这在 analytic predictor 里就对应：

```python
stag_score = self.graph.staggered_score(score, dsigma)
probs = stag_score * self.graph.transp_transition(x, dsigma)
return sample_categorical(probs)
```

所以：

- `staggered_score(...)`：把当前层级的 score 搬回前一层
- `transp_transition(...)`：给出从前一层状态 $z$ 前向走一步变成当前 $x$ 的概率
- 两者相乘：得到上一层状态的后验

### 9.1 Euler vs analytic

- `Euler`：基于 reverse rate 做一阶小步数值近似
- `analytic`：直接利用图结构构造有限步后验，再从后验采样

通常可以把 `analytic` 理解成更像“后验采样”，而不是简单微分方程步进。

## 10. `score entropy` loss 到底在做什么

训练时的主线是：

$$
x_0 \rightarrow x_\sigma \rightarrow s_\theta(x_\sigma,\sigma) \rightarrow \text{score entropy}
$$

代码在 `losses.py` 里大致是：

1. 采样 $t$
2. 得到 $\sigma, dsigma$
3. 从 $x_0$ 采样 noisy 状态 $x_\sigma$
4. 模型输出 `log_score`
5. 用图对应的 `score_entropy(...)` 监督它
6. 再乘 $dsigma$ 做连续时间加权

它和普通 CE 的区别在于：

- 普通 LLM：问“下一个 token 对不对”
- SEDD：问“当前 noisy state 下，你学到的 reverse score 对不对”

所以 `score entropy` 更像：

> reverse process 所需打分规则的监督，而不是普通 token 分类误差。

## 11. Absorbing 图：为什么只在 `MASK` 位置算 loss

在 absorbing 图里，一个真实 token 到时刻 $\sigma$ 只有两种命运：

- 保持原样
- 变成 `MASK`

如果当前位置当前还不是 `MASK`，那它一定还等于原 token，没有歧义；真正需要模型“恢复原 token”的只有那些已经被 mask 掉的位置。

所以：

```python
rel_ind = x == self.dim - 1
```

的意思就是：

> 只在当前是 `MASK` 的位置算 absorbing score entropy。

## 12. Absorbing 图里几个最容易混淆的量

### 12.1 $p_\sigma(M)$ 是什么

$$
p_\sigma(M)
$$

表示：

> 在噪声层级 $\sigma$ 时，当前状态恰好是 `MASK` 的概率。

### 12.2 $p_{\mathrm{keep}}(\sigma)$ 是什么

$$
p_{\mathrm{keep}}(\sigma)
$$

表示：

> 一个原始 token 在累计噪声为 $\sigma$ 时，仍保持原样、没有被吸收到 `MASK` 的概率。

它满足：

$$
\frac{d}{d\sigma}p_{\mathrm{keep}}(\sigma)=-p_{\mathrm{keep}}(\sigma)
$$

解为：

$$
p_{\mathrm{keep}}(\sigma)=e^{-\sigma}
$$

于是被 mask 的概率就是：

$$
1-e^{-\sigma}
$$

### 12.3 $e^{-d\sigma}$ 和 $e^{d\sigma}$ 从哪里来

- $e^{-d\sigma}$：表示在一个很小前向步里“还保持原 token 不动”的概率
- $e^{d\sigma}$：在 reverse / staggered 公式里，表示把概率质量从当前噪声层级往前一小步“恢复回去”的修正因子

普通 token 和 `MASK` 的公式不对称，是因为：

- 普通 token 只会流出到 `MASK`
- `MASK` 会接收所有普通 token 的概率流入

所以 `MASK` 维度总需要额外补一项。

## 13. SFT 原型：数据、训练、评测

### 13.1 数据格式

SFT 原型采用“前缀拼接”的形式，单条样本会被格式化成：

```text
Question:
{question}

Solution:
{solution}
```

训练时：

- prompt 前缀保持干净
- target 后缀才被离散扩散加噪
- loss 只算 target 部分

### 13.2 关键默认配置

SFT 配置在 `configs/sft.yaml`，重要默认项包括：

- `data.task_type=sft`
- `data.dataset_name=simplescaling/s1K-1.1`
- `data.tokenizer_name_or_path=/gpt_tokenizer`
- `data.local_path=/root/workspace/data/train-00000-of-00001.parquet`
- `data.prompt_field=question`
- `data.target_field=solution`
- `data.max_length=512`
- `training.target_only_diffusion=True`
- `sampling.prompt_clamp=True`

### 13.3 训练与评测命令

训练：

```bash
python train_sft.py
```

评测：

```bash
python eval_sft.py
```

评测会输出：

- `eval_outputs/metrics.json`
- `eval_outputs/*_generations.json`
- `eval_outputs/*_samples.txt`

## 14. 一张总的阅读地图

如果你之后要继续深入代码，最推荐的顺序是：

1. 先看 [continuous_vs_discrete_score_notes.md](./continuous_vs_discrete_score_notes.md)
2. 再看 [loglinear_and_dsigma_notes.md](./loglinear_and_dsigma_notes.md)
3. 再看 [sigma_time_embedding_notes.md](./sigma_time_embedding_notes.md)
4. 再看 [transformer_block_attention_notes.md](./transformer_block_attention_notes.md)
5. 再看 [graph_dim_and_output_semantics.md](./graph_dim_and_output_semantics.md)
6. 最后看 [absorbing_diffusion_interview_notes.md](./absorbing_diffusion_interview_notes.md)

## 15. 从连续 Score Matching 到离散 Score Entropy

这一节把图像 diffusion 里的连续 score matching、文本 diffusion 里的离散 ratio score，以及 SEDD 的 `score entropy` 损失放在同一条线上理解。

### 15.1 连续空间里的 $x$ 是什么

在图像 diffusion 里，可以把 $x$ 理解成整张图像的像素向量，而不是某一个单独 pixel：

$$
x \in \mathbb{R}^{H\times W\times C}
$$

比如一张 $32\times32$ RGB 图片，$x$ 就是一个 $32\times32\times3$ 维向量。连续 score 是：

$$
\nabla_x \log p_t(x)
$$

它的形状和图像一样，每个维度都对应一个 pixel channel 的偏导数。直觉上，它告诉我们：在当前噪声层级 $t$ 下，应该怎样微调这张 noisy image，才能让它更像真实数据分布。

虽然原始图像常常是 8-bit 整数，但训练时通常会归一化到 $[0,1]$ 或 $[-1,1]$，再加入 Gaussian noise，所以可以把它当成连续变量求导。

### 15.2 图像 diffusion 为什么可以用 score matching

图像扩散的 forward process 通常是不断加 Gaussian noise，例如 DDPM 里可以写成：

$$
x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon
$$

反向去噪需要知道 noisy distribution 的 score：

$$
\nabla_{x_t}\log p_t(x_t)
$$

这个 score 是一个方向场：当前 noisy image 往哪个连续方向移动，数据密度会上升得最快。

DDPM 常常不直接预测 score，而是预测噪声 $\epsilon$。两者本质上是同一件事的不同参数化，因为条件 score 满足类似关系：

$$
\nabla_{x_t}\log p(x_t\mid x_0) = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}
$$

所以在连续空间里，学 noise、学 denoising direction、学 score，是强相关的几种说法。

### 15.3 文本为什么不能直接照搬

文本 token 是离散状态，例如：

$$
x \in \{\text{cat},\text{dog},\text{MASK},\dots\}
$$

这里没有“把 token 稍微移动一点”的概念，所以不能直接定义：

$$
\nabla_x \log p_t(x)
$$

也就是说，图像里的 $x$ 是连续像素向量，可以求导；文本里的 $x$ 是 token id 或离散状态，不能对 token id 做连续梯度。

SEDD 的关键替换是：不再问“沿哪个连续方向移动”，而是问“从当前状态 $x$ 跳到候选状态 $y$ 后，相对概率是多少”。于是连续 score 被替换为离散 ratio score：

$$
s_t(x)_y \approx \frac{p_t(y)}{p_t(x)}
$$

这里 $x$ 是当前 noisy token 或 noisy sequence，$y$ 是候选替换状态。

### 15.4 CTMC 在这里扮演什么角色

CTMC 是 continuous-time Markov chain，也就是连续时间马尔可夫链。它的特点是：状态是离散的，但时间是连续的。

在 SEDD 里，forward CTMC 描述 token 如何被连续时间地加噪。例如 absorbing graph 中：

$$
\text{real token} \rightarrow \text{MASK}
$$

这个过程由 generator / rate matrix $Q$ 控制：

$$
\frac{d p_t}{dt}=g(t)Qp_t
$$

其中 $Q(x,y)$ 不是普通概率，而是从状态 $x$ 跳到状态 $y$ 的瞬时速率。很短时间 $dt$ 内大致有：

$$
P(x\to y\text{ in }dt)\approx Q(x,y)dt
$$

反向 CTMC 要从 noisy state 回到 clean state，它的跳转速率就需要用到离散 ratio score：

$$
\frac{p_t(y)}{p_t(x)}
$$

所以，连续图像 diffusion 的 reverse process 靠 $\nabla_x\log p_t(x)$；离散文本 diffusion 的 reverse CTMC 靠状态间概率比。

### 15.5 Score Entropy 损失在学什么

SEDD 让模型输出 log-ratio score。对一个当前位置 $x$，模型一次 forward 输出整个词表上的向量：

$$
u_\theta(x)\in\mathbb{R}^{V}, \qquad s_\theta(x)_y=\exp(u_\theta(x)_y)
$$

目标是让：

$$
s_\theta(x)_y \approx \frac{p_t(y)}{p_t(x)}
$$

对单个 ratio，记真实 ratio 为 $r$，模型预测为 $s$，Score Entropy 的基本形式是：

$$
\ell_{SE}(s;r)=s-r\log s+r(\log r-1)
$$

最后一项是常数项，用来让最优值归一到 0。对 $s$ 求导：

$$
\frac{\partial \ell_{SE}}{\partial s}=1-\frac{r}{s}
$$

最优点满足：

$$
s=r
$$

所以这个损失确实会把模型推向真实 probability ratio。

如果模型输出的是 log-score $u=\log s$，那么损失可以看成：

$$
\ell_{SE}(u;r)=e^u-ru+\text{const}
$$

这正好对应代码里的 `score.exp()` 项和 `ratio * score` 项。

### 15.6 positive term 和 negative term 怎么理解

以 absorbing graph 中的 `MASK` 位置为例。当前位置是：

$$
x=\text{MASK}
$$

真实 token 是：

$$
x_0=\text{cat}
$$

模型一次 forward 输出整个词表的 log-score：

$$
u_\theta(x)=[u_{\text{cat}},u_{\text{dog}},u_{\text{apple}},\dots]
$$

Score Entropy 可以直观写成：

$$
\mathcal L=\underbrace{\sum_y \exp(u_y)}_{\text{positive term}}-\underbrace{r\,u_{x_0}}_{\text{negative term}}+\text{const}
$$

positive term 看所有候选 $y$ 的预测 score：

$$
\sum_y s_\theta(y)=\sum_y \exp(u_y)
$$

它的作用是惩罚模型到处给高分。如果模型觉得每个 token 都很好，这个总和会变大，loss 也会变大。

negative term 只看真实 token $x_0$ 那一维：

$$
r\,u_{x_0}
$$

因为 loss 里是减去它，所以提高真实 token 的 log-score 会降低 loss。

代码里的 `gather` 就是在模型输出的 vocab 向量中，把真实 token id 对应的那一格取出来：

```python
other_ind = x0[rel_ind]
neg_term = ratio * torch.gather(
    score[rel_ind],
    -1,
    other_ind[..., None]
).squeeze(-1)
```

合起来看：

- positive term：所有候选别乱高
- negative term：真实 token 应该高
- 平衡点：真实 token 的 ratio score 接近理论 ratio

这和 softmax cross entropy 有点像，但不是同一个东西。CE 学的是归一化概率：

$$
-\log \frac{e^{u_{x_0}}}{\sum_y e^{u_y}}
$$

Score Entropy 学的是非归一化的概率比：

$$
\frac{p_t(y)}{p_t(x)}
$$

### 15.7 为什么“一次评估 $s_\theta(x)$”就可扩展

朴素做法可能需要为每个候选 token $y$ 单独评估一次模型：

$$
s_\theta(x,y)
$$

如果词表大小是 $V=50000$，那每个位置都要跑 50000 次模型，完全不可扩展。

SEDD 的做法是让模型像普通语言模型输出 logits 一样，一次 forward 直接输出整个词表的 score 向量：

$$
x \mapsto s_\theta(x)\in\mathbb{R}^{V}
$$

于是所有候选 token 的 ratio score 都在一次前向里得到：

$$
\left[\frac{p_t(1)}{p_t(x)},\frac{p_t(2)}{p_t(x)},\dots,\frac{p_t(V)}{p_t(x)}\right]
$$

训练时：

- positive term 对整个向量求和
- negative term 只 `gather` 真实 token 那一维
- 不需要对每个候选 token 重新调用模型

这就是论文里说这种版本 scalable 的核心原因。

### 15.8 一句话串起来

图像 diffusion 学的是：

$$
\text{noisy image }x_t \mapsto \nabla_x\log p_t(x_t)
$$

SEDD 学的是：

$$
\text{noisy tokens }x_t \mapsto \left[\frac{p_t(y)}{p_t(x_t)}\right]_y
$$

Score Entropy 是为了稳定地学习这些非负 probability ratios；学会以后，reverse CTMC 就可以用这些 ratio score 把全噪声或全 `MASK` 的序列逐步恢复成文本。

## 16. 最后一句总结

SEDD 可以用一句话记住：

> 连续扩散里学的是 noisy distribution 的梯度 score，离散扩散里因为不能对 token 求导，所以改学状态间的概率比值；Transformer 负责在给定 noisy sequence 和 $\sigma$ 时预测这些 ratio score，`score entropy` 负责训练它们，reverse CTMC 和 analytic / Euler sampling 则负责把这些 score 变回一步步的去噪生成过程。
