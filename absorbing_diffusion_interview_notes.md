# Absorbing Diffusion Interview Notes

这份笔记整理 SEDD 里 absorbing diffusion 最容易被面试追问的几个点，重点包括：

- `p_\sigma(M)` 到底表示什么
- `p_keep(\sigma)` 是什么
- `e^{-d\sigma}` 从哪里来
- `e^{d\sigma}` 为什么会出现
- 为什么可以用“概率流入 / 流出”来理解
- 为什么普通 token 和 `MASK` 的推导不一样

对应代码主要在：

- [graph_lib.py](./graph_lib.py)
- [noise_lib.py](./noise_lib.py)
- [sampling.py](./sampling.py)

---

## 1. 一句话背景

在 absorbing diffusion 里，每个普通 token 只有两种命运：

1. 保持原样
2. 跳到 `MASK`

这里最后一个状态 `M = self.dim - 1` 表示 `MASK` / absorbing state。

---

## 2. `p_\sigma(M)` 是什么

\[
p_\sigma(M)
\]

表示：

> 在噪声层级 `\sigma` 时，当前状态恰好是 `MASK` 的概率。

它不是“所有概率之和”，只是整个分布里的一个分量。

如果状态空间是

\[
\{A, B, M\}
\]

那么整个位于噪声层级 `\sigma` 的分布是：

\[
p_\sigma = \bigl(p_\sigma(A),\; p_\sigma(B),\; p_\sigma(M)\bigr)
\]

其中 `p_\sigma(M)` 只是最后一项。

总概率和才是：

\[
p_\sigma(A)+p_\sigma(B)+p_\sigma(M)=1
\]

---

## 3. `p_keep(\sigma)` 是什么

\[
p_{\text{keep}}(\sigma)
\]

表示：

> 当累计噪声为 `\sigma` 时，一个原始 token 还保持原样、没有被吸收到 `MASK` 的概率。

初始条件是：

\[
p_{\text{keep}}(0)=1
\]

因为 `\sigma=0` 还没加噪，token 一定还在原状态。

---

## 4. 为什么会有

\[
\frac{d}{d\sigma}p_{\text{keep}}(\sigma)=-p_{\text{keep}}(\sigma)
\]

这不是额外拍脑袋规定的，而是由前向连续时间马尔可夫链的设计自动推出的。

### 4.1 人为设计的是 generator

在 [graph_lib.py](./graph_lib.py) 里，absorbing 图把普通 token 的跳转设计成：

- 从当前 token 流出速率为 `1`
- 唯一流入目标是 `MASK`

所以对于普通 token，“还留在原状态”的概率只会流失，不会有别的地方流回来。

### 4.2 直觉解释

如果当前还有很多概率质量留在原 token 上，那下一小步里会流失得更快；如果已经只剩很少，那流失得更慢。

因此流失速度和“当前还剩多少”成正比：

\[
\text{变化率} = -1 \times \text{当前剩余量}
\]

也就是：

\[
\frac{d}{d\sigma}p_{\text{keep}}(\sigma)=-p_{\text{keep}}(\sigma)
\]

它的解就是：

\[
p_{\text{keep}}(\sigma)=e^{-\sigma}
\]

---

## 5. `e^{-d\sigma}` 是怎么来的

因为在一个很小的噪声增量 `d\sigma` 上：

- 普通 token 不跳走的概率，就是保持原状态的概率
- 对上面的微分方程积分后，这个保持概率就是

\[
e^{-d\sigma}
\]

所以：

- **keep probability**：
\[
P(\text{stay}) = e^{-d\sigma}
\]

- **move-to-mask probability**：
\[
P(\text{move to MASK}) = 1-e^{-d\sigma}
\]

代码里对应：

```python
move_chance = 1 - (-sigma).exp()
```

以及：

```python
edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
```

---

## 6. `p_keep(\sigma)\cdot d\sigma` 表示什么

\[
p_{\text{keep}}(\sigma)\cdot d\sigma
\]

表示：

> 当前还没被 `MASK` 掉的那部分概率质量，在这一小步 `d\sigma` 内预计流失掉的量。

更一般地，如果跳转率是 `\lambda`，则会写成：

\[
\text{流失量} \approx p_{\text{keep}}(\sigma)\cdot \lambda \cdot d\sigma
\]

这个 repo 里吸收速率设计成了 `1`，所以就简化成：

\[
\text{流失量} \approx p_{\text{keep}}(\sigma)\cdot d\sigma
\]

---

## 7. 普通 token 的前向小步关系

记 `M` 是 `MASK`，`k \neq M` 是普通 token。

在前向过程中，普通 token 到当前时刻还能保持为 `k`，唯一可能是：

1. 前一刻就在 `k`
2. 这一小步没有跳去 `MASK`

所以：

\[
p_\sigma(k)=e^{-d\sigma}p_{\sigma-d\sigma}(k)
\]

这是 absorbing 图里最核心的小步公式之一。

---

## 8. `e^{d\sigma}` 是怎么来的

它就是把上式反推回去。

由

\[
p_\sigma(k)=e^{-d\sigma}p_{\sigma-d\sigma}(k)
\]

两边除以 `e^{-d\sigma}`：

\[
p_{\sigma-d\sigma}(k)=e^{d\sigma}p_\sigma(k)
\]

所以：

- `e^{-d\sigma}`：前向一小步的保留率
- `e^{d\sigma}`：反推回前一小步时的恢复因子

直觉上：

> 当前值已经经历了一次衰减；想回到衰减前，就要除以衰减率，也就是乘上它的倒数。

---

## 9. 为什么普通 token 没有流入项，而 `MASK` 有

对于普通 token `k \neq M`：

- 它只会流出到 `MASK`
- 不会从其他状态流回 `k`

所以普通 token 的公式特别简单：

\[
p_\sigma(k)=e^{-d\sigma}p_{\sigma-d\sigma}(k)
\]

但 `MASK` 不一样。它会接收所有普通 token 流进来的质量，因此：

\[
p_\sigma(M)
=
p_{\sigma-d\sigma}(M)
+
(1-e^{-d\sigma})\sum_{k\neq M}p_{\sigma-d\sigma}(k)
\]

右边两部分分别表示：

1. 前一刻就已经是 `MASK` 的那部分
2. 这一小步从所有普通 token 跳进 `MASK` 的那部分

---

## 10. 为什么可以用“概率流入 / 流出”来讲

因为这里的前向噪声过程被设计成了**连续时间马尔可夫链**。

马尔可夫性质保证：

> 下一小步的变化只依赖当前状态，不依赖更早历史。

因此我们可以把每个状态的变化写成：

\[
\text{变化}=\text{流入}-\text{流出}
\]

在连续时间下，这就是 master equation / generator 观点。

对面试而言，可以直接说：

> 这里不是随便用“流入流出”打比方，而是因为前向过程本身就是 CTMC，所以概率质量随 `\sigma` 的演化天然可以写成局部流动。

---

## 11. `transp_transition` 里的 `e^{-d\sigma}`

代码：

```python
edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
edge += torch.where(
    i == self.dim - 1,
    1 - (-sigma).squeeze(-1).exp(),
    0
)[..., None]
```

理解要点：

- 第一行给“保持当前状态”的部分放上 `e^{-d\sigma}`
- 第二行只在“当前状态本来就是 `MASK`”时，补上所有普通前驱状态流进来的额外质量

这里的 `i == self.dim - 1` 判断的是：

> 当前这个位置的状态是不是 `MASK`

不是在最后一维逐类别判断。

---

## 12. `staggered_score` 里为什么普通 token 乘 `e^{d\sigma}`，但 `MASK` 还要补一项

在 [graph_lib.py](./graph_lib.py) 里：

```python
def staggered_score(self, score, dsigma):
    score = score.clone()
    extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
    score *= dsigma.exp()[:, None]
    score[..., -1] += extra_const
    return score
```

这是在把当前层级 `\sigma` 的 score，转成前一小步 `\sigma-d\sigma` 的相对权重。

### 普通 token

对普通 token `k \neq M`：

\[
\tilde s_k = e^{d\sigma}s_k
\]

因为它们只需要做“反衰减”。

### `MASK`

对 `MASK`：

\[
\tilde s_M
=
s_M + (1-e^{d\sigma})\sum_{k\neq M}s_k
\]

因为 `MASK` 不只继承自己的质量，还会接收普通 token 流进来的贡献。

代码里先对所有类统一乘 `e^{d\sigma}`，再给最后一个类补 `extra_const`，两步合起来正好等价于上面这个公式。

---

## 13. `staggered_score` 的数值例子

假设状态只有 3 个：

- `A`
- `B`
- `M`（最后一个是 `MASK`）

当前时刻的 score 是：

\[
score = [2,\; 3,\; 5]
\]

并且假设：

\[
e^{d\sigma} = 2
\]

也就是：

\[
1 - e^{d\sigma} = -1
\]

代码：

```python
extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
score *= dsigma.exp()[:, None]
score[..., -1] += extra_const
```

### 第一步：算 `extra_const`

\[
score.sum() = 2 + 3 + 5 = 10
\]

所以：

\[
extra\_const = (1 - 2)\times 10 = -10
\]

### 第二步：所有类别先统一乘 `e^{d\sigma}`

\[
[2,3,5] \to [4,6,10]
\]

这一步对应：

\[
\tilde s_A = e^{d\sigma}s_A,\qquad \tilde s_B = e^{d\sigma}s_B
\]

### 第三步：只对 `MASK` 类补修正项

最后一个类别原本是 `10`，再加上 `-10`：

\[
10 + (-10) = 0
\]

所以最终结果是：

\[
[4,\; 6,\; 0]
\]

### 和目标公式对照

普通类：

\[
\tilde s_A = 2\times 2 = 4,\qquad \tilde s_B = 2\times 3 = 6
\]

`MASK` 类：

普通类和为：

\[
2+3=5
\]

因此：

\[
\tilde s_M = 5 + (1-2)\times 5 = 0
\]

和代码结果完全一致。

### 这个例子想说明什么

- `score *= e^{d\sigma}`：先假装所有类都像普通类一样反推
- `extra_const`：再专门把 `MASK` 类纠偏到正确公式

所以：

- 普通类只需要放大
- `MASK` 类需要“放大 + 修正”

---

## 14. 面试快答

### Q1. `p_\sigma(M)` 是什么？

是噪声层级 `\sigma` 下状态恰好为 `MASK` 的概率，不是总概率和。

### Q2. `e^{-d\sigma}` 是什么？

是普通 token 在一个小噪声步长 `d\sigma` 内不被吸收到 `MASK` 的概率。

### Q3. `e^{d\sigma}` 是什么？

是 `e^{-d\sigma}` 的倒数，用来把当前层级反推回前一小步的恢复因子。

### Q4. 为什么普通 token 可以写成
\[
p_\sigma(k)=e^{-d\sigma}p_{\sigma-d\sigma}(k)
\]

因为普通 token 没有流入项；到当前时刻还能在 `k`，只能说明前一刻在 `k` 且这一小步没跳走。

### Q5. 为什么 `MASK` 不能这么简单？

因为 `MASK` 会接收所有普通 token 流进来的质量，所以它有额外的求和项。

### Q6. 为什么能用概率流入 / 流出解释？

因为前向噪声过程被设计成连续时间马尔可夫链，状态概率的演化天然可以用 generator / master equation 来描述。

---

## 15. 最短总结

absorbing diffusion 的核心结构是：

- 普通 token 以速率 `1` 吸收到 `MASK`
- 因此 keep 概率满足
\[
\frac{d}{d\sigma}p_{\text{keep}}(\sigma)=-p_{\text{keep}}(\sigma)
\]
- 所以
\[
p_{\text{keep}}(\sigma)=e^{-\sigma}
\]
- 一小步上：
\[
P(\text{stay})=e^{-d\sigma},\qquad P(\text{move})=1-e^{-d\sigma}
\]
- 反推普通 token：
\[
p_{\sigma-d\sigma}(k)=e^{d\sigma}p_\sigma(k)
\]
- `MASK` 还要额外加上所有普通 token 流入的总贡献

如果面试官追问，你可以直接用一句话回答：

> 这套推导成立，是因为前向吸收过程被设计成一个非常简单的 CTMC：普通 token 只会以固定速率流向 `MASK`，因此普通状态是纯指数衰减，而 `MASK` 则是自保持加总流入。
