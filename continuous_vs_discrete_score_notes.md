# 连续空间 Score 与离散空间 SEDD Score 对比笔记

这份笔记整理我们前面讨论过的核心问题：

- 连续扩散里学的 `score` 是什么
- 为什么连续空间里会出现 Fisher divergence
- 为什么离散空间里不能再直接用 `\nabla_x \log p_t(x)`
- SEDD 为什么改学状态间概率比值
- `score entropy` 为什么可以看作连续 score matching 在离散空间里的对应物

## 1. 连续空间里学的是什么

在连续空间里，时刻 `t` 的 noisy distribution 是：

\[
p_t(x)
\]

对应的 score 是：

\[
\nabla_x \log p_t(x)
\]

它表示：

> 在当前噪声层级下，如果把点 `x` 稍微移动一点，往哪个方向走会让密度上升得最快。

所以它本质上是一个“局部去噪方向场”。

## 2. 连续 Fisher divergence 是什么

给真实分布 `p` 和模型分布 `q`，连续 Fisher divergence 通常写成：

\[
D_F(p \,\|\, q)
=
\frac12
\mathbb{E}_{x \sim p}
\left[
\left\|
\nabla_x \log p(x) - \nabla_x \log q(x)
\right\|^2
\right]
\]

它衡量的是：

> 两个分布的 score function 有多接近。

因此，连续 score matching 的核心思想不是直接拟合 `p(x)` 本身，而是拟合：

\[
\nabla_x \log p(x)
\]

## 3. 为什么连续扩散里学的是 noisy distribution 的 score

在连续 diffusion / score-based model 里，训练目标通常不是只学原始分布 `p_0(x)` 的 score，而是学很多噪声层级下的：

\[
\nabla_x \log p_t(x)
\]

因为 reverse diffusion 要在很多中间噪声层级上工作，所以模型需要知道：

- 高噪声时往哪去
- 中噪声时往哪去
- 低噪声时往哪去

## 4. 为什么离散空间里不能直接照搬

在离散 token 空间里，状态不是连续变量，不能对 token id 做无穷小扰动，因此：

\[
\nabla_x \log p_t(x)
\]

没有直接意义。

也就是说：

- 连续空间里可以问“往哪个无穷小方向走更像数据”
- 离散空间里没有“无穷小方向”这个概念

所以离散空间不能直接复用连续 score 的定义。

## 5. 离散空间里改学什么

SEDD 在离散空间里改学的是：

\[
\frac{p_t(y)}{p_t(x)}
\]

其中：

- `x` 是当前离散状态
- `y` 是另一个候选状态

它表示：

> 当前在 `x` 时，候选状态 `y` 相对于 `x` 有多合理。

这不是导数，而是状态间的概率比值。

## 6. 为什么这个比值是自然的离散替代物

因为离散 reverse CTMC 里真正需要的就是：

\[
\bar Q_t(y,x)\propto Q_t(x,y)\frac{p_t(y)}{p_t(x)}
\]

这里：

- `Q_t(x,y)`：基础前向图里从 `y` 到 `x` 的跳转结构
- `p_t(y)/p_t(x)`：`y` 相对当前状态 `x` 的合理程度

所以在连续空间里，reverse process 由 `\nabla_x \log p_t(x)` 决定；
在离散空间里，reverse process 则由：

\[
\frac{p_t(y)}{p_t(x)}
\]

决定。

## 7. 连续 score 与离散 ratio 的对应关系

可以把它们并排记成：

### 连续空间

\[
\text{score}_t(x) = \nabla_x \log p_t(x)
\]

作用：

> 给出局部密度上升方向，从而定义 reverse diffusion 的去噪方向。

### 离散空间

\[
\text{score}_t(y,x) \approx \frac{p_t(y)}{p_t(x)}
\]

作用：

> 给出候选状态 `y` 相对当前状态 `x` 的合理程度，从而定义 reverse CTMC 的跳转倾向。

## 8. `score entropy` 在这里扮演什么角色

连续空间里，score matching / Fisher divergence 的核心任务是：

> 让模型学到正确的连续 score。

SEDD 里的 `score entropy` 则对应：

> 让模型学到正确的离散 ratio score。

所以它们的精神是一致的：

- 都不是直接学概率分布本身
- 都是学足以定义 reverse process 的 score-like object

但形式不同：

- 连续空间：匹配 `\nabla \log p`
- 离散空间：匹配 `p_t(y)/p_t(x)`

## 9. 为什么说 `score entropy` 是对应物，而不是简单离散化

`score entropy` 不是把连续 Fisher divergence 生硬离散化一下得到的。

更准确地说：

1. 连续空间的核心思想是“匹配 score”
2. 离散空间里没有连续导数
3. reverse CTMC 公式告诉我们需要的是状态间比值
4. 所以 SEDD 基于这个比值重新构造了离散空间的训练目标

因此：

> `score entropy` 继承的是连续 score matching 的思想，而不是连续公式的表面形式。

## 10. 最后一句总结

连续空间里，模型学的是 noisy distribution 的梯度 score：

\[
\nabla_x \log p_t(x)
\]

离散空间里，因为不能对状态求导，SEDD 改学状态间概率比值：

\[
\frac{p_t(y)}{p_t(x)}
\]

而 `score entropy` 就是训练这个离散 ratio score 的目标。因此它可以看作连续 score matching / Fisher divergence 思想在离散扩散上的对应版本，但不是简单的直接离散化。
