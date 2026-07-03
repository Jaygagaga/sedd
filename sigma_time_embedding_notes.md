# SEDD 里的 `sigma`、Time Embedding 与条件调制笔记

这份笔记整理 SEDD 实现里和 `sigma` 相关的核心概念，重点说明：

- `sigma` 和 `dsigma` 分别是什么；
- 为什么论文或实现里常说 `time embedding`；
- 但在这份代码里，更准确地说它其实是 `noise-level embedding` / `sigma embedding`；
- `sigma` 是如何进入模型的；
- `adaLN_modulation` 在做什么；
- 为什么同一个 noisy token 在不同 `sigma` 下含义不一样；
- 为什么 absorbing noise 下，当前 token 是 `MASK` 时，模型必须知道当前 `sigma` 才能决定该多激进地恢复。

---

## 1. `sigma` 和 `dsigma` 是什么

在这份实现里，噪声模块 `noise(t)` 返回两个量：

- `sigma`：累计噪声强度；
- `dsigma`：累计噪声对时间的一阶导数，也就是瞬时噪声速率。

对应代码在 [noise_lib.py](./noise_lib.py)：

```python
def forward(self, t):
    return self.total_noise(t), self.rate_noise(t)
```

因此这里可以直接记成：

- `sigma = total_noise(t)`
- `dsigma = rate_noise(t)`

从理论上看，它们大致对应：

- `sigma \approx \bar\sigma(t)`：累计噪声；
- `dsigma \approx d\bar\sigma(t) / dt`：噪声增长速度。

直觉上：

- `sigma` 决定“当前这条序列已经脏到什么程度”；
- `dsigma` 决定“这一小步里噪声变化得有多快”。

---

## 2. 为什么经常说 `time embedding`

扩散模型里经常会把“当前扩散阶段”叫做 `time`、`timestep` 或 `diffusion time`，所以很多实现里习惯把对应的条件输入叫做：

- `time embedding`
- `timestep embedding`

这份代码里的模块名也是：

```python
self.sigma_map = TimestepEmbedder(config.model.cond_dim)
```

对应代码在 [model/transformer.py](./model/transformer.py)。

但要特别注意：

**这份实现虽然模块名叫 `TimestepEmbedder`，真正喂进去的不是离散步数编号，也不是原始 `t`，而是 `sigma`。**

前向代码是：

```python
c = F.silu(self.sigma_map(sigma))
```

所以从实现语义上，更准确的叫法其实是：

- `noise-level embedding`
- `sigma embedding`

换句话说：

> 理论上常说 “time embedding”，  
> 但在这份实现里，更准确地说它其实是 “noise-level embedding” 或 “sigma embedding”。

---

## 3. 为什么模型需要 `sigma embedding`

这是最重要的一点。

同一个 noisy token `x_t`，在不同 `sigma` 下含义并不一样。

模型不能只看“当前 token 是什么”，还必须知道：

> 它是在一个什么噪声层级下看到这个 token 的。

### 3.1 一个最直观的例子：当前 token 是 `MASK`

设当前某个位置是 `MASK`。

这时如果：

- `sigma` 很大，说明噪声很重；
- 那么出现 `MASK` 很正常；
- 模型会更保守，因为此时很多位置都还很脏。

但如果：

- `sigma` 很小，说明已经快去噪完了；
- 那么这个位置还是 `MASK` 就很异常；
- 模型应该更积极地把它恢复成真实词。

也就是说：

> 同样是一个 `MASK`，  
> 当 `sigma` 大时，它表示“现在还在很早的、很脏的阶段”；  
> 当 `sigma` 小时，它表示“已经接近最终结果了，这里还没恢复出来就不太对”。

所以模型必须知道当前噪声层级，才能决定：

- 该多保守；
- 该多激进；
- 该更多依赖上下文；
- 还是更果断地恢复具体 token。

### 3.2 更一般地说

这不只是 `MASK` 的问题。

对于任意 noisy token，模型都需要知道：

- 当前 token 是“轻微污染”下出现的；
- 还是“重污染”下出现的。

在不同 `sigma` 下，同一个观察值对应的后验语义不同。

因此：

> `sigma embedding` 的作用，就是告诉模型：  
> “你现在是在第几层噪声下解释这串 noisy tokens。”

---

## 4. `sigma_map` 的输入 `sigma` 维度是什么

模型前向定义是：

```python
def forward(self, indices, sigma):
```

其中：

- `indices` 的形状是 `[B, L]`
- `sigma` 的形状是 `[B]`

这里的意思是：

- 一个 batch 有 `B` 条序列；
- 每条序列有一个自己的噪声强度 `sigma`；
- 同一条序列内部所有 token 共用同一个 `sigma`。

这点在 [model/utils.py](./model/utils.py) 里也能看到：

```python
sigma = sigma.reshape(-1)
```

也就是说最后会整理成一维：

```python
sigma.shape == [B]
```

再送进：

```python
self.sigma_map(sigma)
```

输出是：

```python
[B, cond_dim]
```

默认 `cond_dim = 128`。

所以可以记成：

```text
sigma: [B]
  -> TimestepEmbedder
  -> sigma embedding c: [B, cond_dim]
```

这里的 `sigma` 不是 `[B, L]`，因为它不是每个 token 一个噪声值，而是每条样本一个噪声层级。

---

## 5. `sigma_map` 在模型里用在哪里

前向中：

```python
x = self.vocab_embed(indices)
c = F.silu(self.sigma_map(sigma))
```

这里：

- `x` 是 token embedding；
- `c` 是由 `sigma` 映射得到的条件向量。

然后 `c` 会被送进每一个 Transformer block 的条件调制模块：

```python
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
```

这说明 `sigma embedding` 并不是简单加到 token embedding 上，而是用来**调制整个网络的工作方式**。

---

## 6. `adaLN_modulation` 是做什么的

`adaLN_modulation` 可以理解成：

> 把 `sigma embedding` 变成每一层 Transformer 的“控制参数”。

在每个 block 中有：

```python
self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
```

它把每个样本的条件向量 `c` 映射成 6 组参数：

- `shift_msa`
- `scale_msa`
- `gate_msa`
- `shift_mlp`
- `scale_mlp`
- `gate_mlp`

这些参数分别作用在：

- attention 分支前的 LayerNorm 输出；
- attention 分支残差的门控；
- MLP 分支前的 LayerNorm 输出；
- MLP 分支残差的门控。

### 6.1 `shift` 和 `scale`

它们作用在归一化后的 hidden state 上：

\[
\text{AdaLN}(x) = \text{LN}(x)\cdot (1 + \text{scale}) + \text{shift}
\]

所以：

- `scale` 决定某些通道要不要放大/缩小；
- `shift` 决定某些通道要不要平移。

这让网络能根据当前噪声层级，改变特征的解释方式。

### 6.2 `gate`

`gate` 控制这一支残差该“开多大”。

直觉上相当于：

- 噪声大时，某些分支可以更保守；
- 噪声小时，某些分支可以更积极；
- 同一个 block 在不同 `sigma` 下不必用完全相同的行为模式。

因此：

> `adaLN_modulation` 不是提供内容信息，  
> 而是根据 `sigma` 动态调节每一层该如何处理当前 noisy sequence。

---

## 7. `token embedding`、`position embedding`、`sigma embedding` 各管什么

可以把它们并排记：

### 7.1 `token embedding`

回答：

> 当前这个位置上是什么 token？

它提供内容信息。

### 7.2 `rotary position embedding`

回答：

> 当前 token 在序列里的哪个位置？

它提供位置信息。

### 7.3 `sigma embedding`

回答：

> 当前这整条序列处于哪个扩散阶段 / 噪声层级？

它提供扩散阶段信息。

所以一句话总结是：

> `token embedding` 管内容，  
> `rotary embedding` 管位置，  
> `sigma embedding` 管当前噪声层级。

---

## 8. `sigma[:, None]` 是什么意思

训练里经常会看到：

```python
graph.sample_transition(batch, sigma[:, None])
```

这里原始：

```python
sigma.shape == [B]
```

而：

```python
sigma[:, None].shape == [B, 1]
```

意思是给 `sigma` 新增一个维度，方便和序列张量 `[B, L]` 广播。

这样每个样本的一个 `sigma`，就可以自动扩展到该样本的所有 token 位置。

也就是说：

- 第 1 条样本整句共用 `sigma_1`
- 第 2 条样本整句共用 `sigma_2`

---

## 9. `analytic` 采样里为什么也离不开 `sigma`

在 `analytic` predictor 里，会先计算当前和下一步的噪声强度：

```python
curr_sigma = self.noise(t)[0]
next_sigma = self.noise(t - step_size)[0]
dsigma = curr_sigma - next_sigma
```

然后模型会用当前 `curr_sigma` 计算 score：

```python
score = score_fn(x, curr_sigma)
```

这说明在 reverse 采样时，模型每一步都要知道：

- 当前噪声层级是多少；
- 正在从哪个噪声层级往哪个噪声层级回退。

如果没有 `sigma`，模型就无法判断：

- 当前 `MASK` 是不是合理；
- 当前 token 应该更保守还是更激进地更新。

---

## 10. 一个和 `MASK` 有关的核心直觉

这一点非常值得单独记住：

> 同一个 noisy token `x_t`，在不同 `sigma` 下含义不一样。

特别是当前某个位置是 `MASK` 时：

- 如果 `sigma` 很大，说明噪声很重，`MASK` 很正常；
- 如果 `sigma` 很小，说明已经快去噪完了，`MASK` 就很异常，模型应该更积极恢复成真实词。

所以 `sigma embedding` 的根本作用，就是让模型知道：

> “你现在看到的 noisy sequence，到底是处在很早期的高噪声阶段，还是已经接近最终恢复的低噪声阶段。”

这会直接影响模型对当前 token 的解释，也会影响它接下来对 reverse 过程的预测。

---

## 11. 最后一句话总结

在这份 SEDD 实现里，虽然代码模块名叫 `TimestepEmbedder`，理论上也常说 `time embedding`，但从真正输入和作用来看，它更准确地说是：

- `noise-level embedding`
- `sigma embedding`

它的职责不是告诉模型“第几个 token 在哪里”，而是告诉模型：

> 当前这整条 noisy sequence 处在什么噪声层级下，
> 所以同一个 noisy token 应该被如何解释，以及网络该采取什么样的去噪策略。
