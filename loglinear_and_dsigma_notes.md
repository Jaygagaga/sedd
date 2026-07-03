# SEDD 里的 `loglinear` 噪声与 `sigma / dsigma` 作用笔记

这份笔记整理两个经常一起出现的问题：

1. `LogLinearNoise` 到底在定义什么，为什么这样写；
2. 为什么前向转移公式里主要用的是 `sigma`，而 loss 里又要乘一个 `dsigma`。

对应代码主要在：

- [noise_lib.py](./noise_lib.py)
- [losses.py](./losses.py)
- [graph_lib.py](./graph_lib.py)

---

## 1. `LogLinearNoise` 的定义

代码在 [noise_lib.py](./noise_lib.py)：

```python
class LogLinearNoise(Noise, nn.Module):
    def __init__(self, eps=1e-3):
        self.eps = eps

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)
```

因此它定义的是：

\[
\sigma(t)= -\log\big(1-(1-\varepsilon)t\big)
\]

以及它对时间的导数：

\[
\frac{d\sigma(t)}{dt}
=
\frac{1-\varepsilon}{1-(1-\varepsilon)t}
\]

在这份代码的命名里：

- `total_noise(t)` 对应 `sigma(t)`
- `rate_noise(t)` 对应 `dsigma/dt`

所以调用：

```python
sigma, dsigma = noise(t)
```

时，实际上就是：

- `sigma = sigma(t)`
- `dsigma = d sigma(t) / dt`

---

## 2. 为什么叫 `loglinear`

从定义出发：

\[
\sigma(t)= -\log\big(1-(1-\varepsilon)t\big)
\]

两边取负指数：

\[
e^{-\sigma(t)} = 1-(1-\varepsilon)t
\]

右边是关于 \(t\) 的线性函数。

所以它叫 `loglinear` 的原因就是：

> `sigma(t)` 是一个线性函数取负对数之后得到的量。

换句话说，它不是让 `sigma` 本身线性，而是让：

\[
e^{-\sigma(t)}
\]

或者等价地某些前向转移概率，随 \(t\) 呈现线性结构。

---

## 3. 为什么 `loglinear` 特别适合 absorbing noise

在 absorbing 图里，前向采样会用到：

\[
1 - e^{-\sigma}
\]

例如在 [graph_lib.py](./graph_lib.py) 的 absorbing 前向采样中：

```python
move_chance = 1 - (-sigma).exp()
```

它表示某个 token 到当前时刻已经被打成 `MASK` 的概率。

如果把 `loglinear` 的定义代进去：

\[
1-e^{-\sigma(t)}
=
1-\big(1-(1-\varepsilon)t\big)
=
(1-\varepsilon)t
\]

这说明在 `loglinear` schedule 下：

\[
p(\text{token 被 mask 到时刻 } t) = (1-\varepsilon)t
\]

这是一个非常漂亮的结果：

- \(t=0\) 时几乎不 mask
- \(t=1\) 时 mask 概率接近 \(1-\varepsilon\)
- mask 概率随时间近似线性增长

因此：

> `loglinear` 的真正好处不是让 `sigma` 线性，  
> 而是让 absorbing 噪声里的“被 mask 概率”随时间线性增长。

这就是为什么它特别适合 `absorb` 图。

---

## 4. `eps` 的作用

如果没有 `eps`，则在 \(t=1\) 时：

\[
\sigma(1)= -\log(0)=\infty
\]

数值上会发散。

所以作者引入一个很小的：

\[
\varepsilon = 10^{-3}
\]

于是：

\[
\sigma(1)= -\log(\varepsilon)
\]

这会是一个很大的有限数。

同时：

\[
1-e^{-\sigma(1)} = 1-\varepsilon
\]

也就是“几乎完全污染，但不真的到无穷大”。

---

## 5. 为什么 `rate_noise` 会越来越大

对 `loglinear`：

\[
\frac{d\sigma(t)}{dt}
=
\frac{1-\varepsilon}{1-(1-\varepsilon)t}
\]

当 \(t\to 1\) 时，分母越来越小，因此导数越来越大。

这表示：

- 前期 `sigma` 增长比较平缓；
- 后期为了让污染概率逼近 1，`sigma` 必须迅速抬升。

注意这里不要混淆：

- `sigma(t)` 的增长不是线性的；
- 但 absorbing 模型里真正关心的
  \[
  1-e^{-\sigma(t)}
  \]
  是线性的。

---

## 6. 为什么前向转移公式里用的是 `sigma`

前向连续时间马尔可夫链写成：

\[
\frac{d p_t}{dt} = g(t)\,Q\,p_t
\]

其中：

- \(Q\) 是基础转移矩阵；
- \(g(t)\) 是噪声速率；
- 在代码里可以把它理解成 `dsigma`。

定义累计噪声：

\[
\sigma(t) = \int_0^t g(s)\,ds
\]

那么方程解是：

\[
p_t = \exp(\sigma(t)Q)\,p_0
\]

也就是说，从 \(x_0\) 到当前时刻的前向分布，取决于：

\[
\sigma(t)
\]

这个**累计量**，而不是单独取决于当前瞬时速率 \(g(t)\)。

所以前向转移、前向采样、前向概率这些地方主要都用 `sigma`。

在代码里，训练中的前向加噪也是：

```python
perturbed_batch = graph.sample_transition(batch, sigma[:, None])
```

见 [losses.py](./losses.py)。

直觉上：

> 当前已经累计了多少噪声，决定了当前有多脏；  
> 所以前向状态分布看的是 `sigma`。

---

## 7. 为什么 loss 里要乘 `dsigma`

在训练里，loss 最后会做：

```python
loss = (dsigma[:, None] * loss).sum(dim=-1)
```

见 [losses.py](./losses.py)。

这里乘 `dsigma` 的原因是：

> 训练时通常是按时间 \(t\) 采样，  
> 但理论目标本质上更像是在对噪声层级 \(\sigma\) 积分。

因为：

\[
\sigma(t)=\int_0^t g(s)\,ds
\]

所以：

\[
d\sigma = \frac{d\sigma}{dt}dt = g(t)\,dt
\]

也就是：

\[
d\sigma = dsigma \cdot dt
\]

因此如果理论目标是：

\[
\int f(\sigma)\,d\sigma
\]

而你实际是对 \(t\) 采样，那么变量替换后应写成：

\[
\int f(\sigma(t)) \frac{d\sigma}{dt}\,dt
\]

也就是：

\[
\int f(\sigma(t))\,dsigma\,dt
\]

因此在 Monte Carlo 训练里，必须乘上：

\[
dsigma
\]

作为权重。

---

## 8. 一个更直观的理解

可以把 `sigma` 和 `dsigma` 的角色分得很清楚：

### 8.1 `sigma`

回答：

> 现在已经累计了多少噪声？

所以它控制：

- 当前前向分布；
- 当前 token 被污染到什么程度；
- 当前前向转移概率。

### 8.2 `dsigma`

回答：

> 在当前时间点附近，噪声增长得有多快？

所以它控制：

- 连续时间目标中这一段噪声层级的权重；
- 按时间采样时，对不同噪声层级的测度修正。

一句话说：

> `sigma` 决定“当前位置在噪声轴上的哪里”，  
> `dsigma` 决定“时间走一小步时在噪声轴上移动得有多快”。

---

## 9. 为什么不乘 `dsigma` 会出问题

如果只按 \(t\) 均匀采样而不乘 `dsigma`，那你训练时实际优化的是“按时间平均”的目标，而不是“按噪声层级平均”的目标。

这会导致不同噪声区间的贡献被扭曲：

- 如果某段时间里 `sigma` 变化很慢，这段噪声层级会被过度采样；
- 如果某段时间里 `sigma` 变化很快，这段噪声层级会被低估。

乘上 `dsigma` 就是在做这个修正。

因此：

> loss 里乘 `dsigma`，本质上是在补偿时间参数化带来的不均匀。

---

## 10. 把两件事连起来记

### 为什么前向转移公式里用的是 `sigma`

因为前向分布解是：

\[
p_t = \exp(\sigma(t)Q)\,p_0
\]

它只看累计噪声，不看局部导数。

### 为什么 loss 里又要乘 `dsigma`

因为训练是按时间 \(t\) 采样，而理论目标更像对噪声层级 \(\sigma\) 积分；
变量替换时需要：

\[
d\sigma = dsigma \cdot dt
\]

所以必须乘 `dsigma`。

---

## 11. 最后一段总总结

在 SEDD 这套实现里：

- `loglinear` 定义的是
  \[
  \sigma(t)= -\log(1-(1-\varepsilon)t)
  \]
  它的好处是让 absorbing 噪声下 token 被 mask 的概率
  \[
  1-e^{-\sigma(t)}
  \]
  随时间近似线性增长；

- 前向转移公式用 `sigma`，因为当前分布只由累计噪声决定；

- loss 里乘 `dsigma`，因为训练按时间采样时，需要通过
  \[
  d\sigma = dsigma \cdot dt
  \]
  来修正不同噪声层级的权重。

一句话压缩成最核心的版本：

> `sigma` 决定“当前已经脏到什么程度”，  
> `dsigma` 决定“当前这段时间在噪声轴上走得有多快”；  
> 前向分布看前者，连续时间训练权重看后者。
