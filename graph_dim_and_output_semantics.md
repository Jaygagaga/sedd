# SEDD 里的 `graph.dim = 50258` 与输出 `log_score` 语义

这份短笔记整理一个很容易混淆、但很关键的问题：

- 为什么这里会出现 `50258`
- 模型每个位置输出的长度为 `50258` 的向量到底代表什么
- 它和传统 LLM 的 `next-token logits` 有什么区别
- 后验概率相关公式在代码里对应哪里

## 1. 为什么是 `50258`

在当前配置里：

- GPT-2 原始词表大小是 `50257`
- `graph.type = absorb`

`absorb` 图会在原始词表之外，再增加一个额外的 absorbing state，也就是 `MASK` 状态。

所以：

\[
\text{graph.dim} = 50257 + 1 = 50258
\]

这 `50258` 个状态分别是：

- `0 ~ 50256`：真实 token
- `50257`：额外的 `MASK` / absorbing state

## 2. `50258` 在模型输出里的语义

模型在每个位置输出的是一个长度为 `50258` 的向量：

\[
\text{log\_score} \in \mathbb{R}^{50258}
\]

它表示的是：

> 在当前 noisy token 和当前噪声层级 `sigma` 条件下，图上所有候选状态的 `log-score` / `log-ratio`。

这里的“候选状态”不是只有普通词表 token，还包括 absorbing 图里的额外 `MASK` 状态。

因此，输出维度必须和 `graph.dim` 对齐。

## 3. 它不是普通 LLM 的 `next-token logits`

这点非常重要。

传统自回归 LLM 在某个位置输出的通常是：

\[
\ell_t(w)
= \text{logits}_{\theta}(w \mid x_{<t})
\quad \text{for next token } w
\]

经过 softmax 之后才得到下一个 token 的概率：

\[
p_{\theta}(x_t = w \mid x_{<t})
=
\frac{\exp(\ell_t(w))}
{\sum_{v \in \mathcal{V}} \exp(\ell_t(v))}
\]

也就是：

> 给“下一个 token 是什么”打分。

而 SEDD 在这里输出的不是“下一个 token logits”，而是：

> 给“当前 noisy 状态下，这个位置对应图上每个可能状态的相对分数”打分。

更准确地说，模型输出的是 `log_score`，其指数形式近似对应某种 ratio score：

\[
\text{score}(y) \approx \frac{p_{\sigma}(y)}{p_{\sigma}(x)}
\]

其中：

- `x` 是当前 noisy 状态
- `y` 是候选状态
- `sigma` 是当前累计噪声强度

所以它回答的问题不是：

> “下一个 token 是什么？”

而更像是：

> “在当前噪声层级下，相对于当前 noisy token，其他候选状态各有多合理？”

## 4. 后验概率相关代码在哪里

如果把当前 noisy 状态记为 `x`，把上一步更干净的候选状态记为 `z`，那么一次反向采样需要的分布可以写成：

\[
p_{\sigma - \Delta\sigma}(z \mid x_{\sigma}=x)
\propto
p_{\Delta\sigma}(x \mid z)\,
p_{\sigma - \Delta\sigma}(z)
\]

因为归一化时只关心不同 `z` 之间的相对大小，所以可以除以一个和 `z` 无关的常数 \(p_{\sigma}(x)\)：

\[
p_{\sigma - \Delta\sigma}(z \mid x_{\sigma}=x)
\propto
\underbrace{
\frac{p_{\sigma - \Delta\sigma}(z)}{p_{\sigma}(x)}
}_{\text{staggered score}}
\underbrace{
p_{\Delta\sigma}(x \mid z)
}_{\text{transpose transition}}
\]

这正是 `sampling.py` 里的 analytic predictor：

```python
stag_score = self.graph.staggered_score(score, dsigma)
probs = stag_score * self.graph.transp_transition(x, dsigma)
return sample_categorical(probs)
```

对应位置：

- `sampling.py:82`：`score = score_fn(x, curr_sigma)`，先让模型给当前 noisy token 输出 ratio score。
- `model/utils.py:49-51`：采样时把模型输出的 `log_score` 通过 `exp()` 变成正的 `score`。
- `graph_lib.py:91-97`：`staggered_score` 的注释直接写明它在算 \(p_{\sigma-\Delta\sigma}(z) / p_{\sigma}(x)\)。
- `graph_lib.py:218-226`：`Absorbing.transp_transition` 在 absorbing 图上计算 \(p_{\Delta\sigma}(x \mid z)\)。
- `sampling.py:84-86`：把两项相乘，得到未归一化的后验概率 `probs`，再采样。
- `catsample.py:10-13`：`sample_categorical(probs)` 实际从这个 categorical 分布里抽样。

最后的 denoising step 也用同样的结构，只是把 \(\Delta\sigma\) 换成当前的 \(\sigma\)，近似从 noisy token 直接采样干净 token：

\[
p_0(z \mid x_{\sigma}=x)
\propto
\frac{p_0(z)}{p_{\sigma}(x)}
p_{\sigma}(x \mid z)
\]

对应 `sampling.py:97-105`：

```python
stag_score = self.graph.staggered_score(score, sigma)
probs = stag_score * self.graph.transp_transition(x, sigma)
if self.graph.absorb:
    probs = probs[..., :-1]
return sample_categorical(probs)
```

这里 `probs[..., :-1]` 是 absorbing 图特有的处理：最终去噪时不要再把 `MASK` 当成可输出的真实 token。

如果用 Euler predictor，则不是直接构造上面的离散后验，而是用反向跳转率：

\[
\bar{Q}_{\sigma}(x \to y)
=
Q(y \to x)
\frac{p_{\sigma}(y)}{p_{\sigma}(x)}
\]

对应 `graph_lib.py:77-85` 的：

```python
normalized_rate = self.transp_rate(i) * score
```

以及 `sampling.py:62-66` 的：

```python
score = score_fn(x, sigma)
rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
x = self.graph.sample_rate(x, rev_rate)
```

## 5. 一句话总结

`50258` 表示 absorbing 图上的状态空间大小，不只是词表大小；模型每个位置输出一个 `50258` 维 `log_score` 向量，用来刻画当前 noisy 状态下所有候选状态的相对分数，而不是像传统 LLM 那样直接输出 `next-token logits`。
