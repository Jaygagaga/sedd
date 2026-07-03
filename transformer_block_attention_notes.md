# 传统 LLM 与 SEDD / DDiT Block 的 Attention 分支对比笔记

这份笔记整理下面几个问题：

- 传统 LLM 里的 attention 分支到底是怎么接 residual 的；
- SEDD 这份实现里的 `DDiTBlock` 相比标准 pre-LN Transformer 多了什么；
- `shift / scale / gate` 各自作用在哪里；
- 为什么不能说 “attention weights 直接作用到 hidden states 上”；
- 为什么更准确地说，`gate` 作用的是 attention 分支输出，而不是 attention weights 本身。

对应代码主要在：

- [model/transformer.py](./model/transformer.py)
- [model/fused_add_dropout_scale.py](./model/fused_add_dropout_scale.py)

---

## 1. 先纠正一个常见说法

有时会把 Transformer attention 简化说成：

> attention weights 直接作用到 hidden states 上

这句话不够准确。

更准确的写法是：

1. 先从输入 hidden state 生成 `Q, K, V`
2. 再算 attention weights
3. attention weights 作用在 `V` 上
4. 然后结果再过输出投影 `W_o`
5. 最后加回 residual

也就是说：

> attention weights 不是直接乘原始 hidden state `x`，  
> 而是乘 `V`。

---

## 2. 传统 pre-LN LLM 的 attention 分支

如果用比较标准的 pre-LN Transformer 记号，attention 分支可以写成：

\[
h_0 = \text{LN}(x)
\]

\[
Q = W_q h_0,\qquad K = W_k h_0,\qquad V = W_v h_0
\]

\[
A = \text{softmax}(QK^\top / \sqrt{d})
\]

\[
\text{AttnOut} = A V
\]

\[
h = x + W_o(\text{AttnOut})
\]

如果还考虑 dropout，则更接近：

\[
h = x + \text{Dropout}(W_o(\text{AttnOut}))
\]

这里可以看到：

- `A` 是 attention weights；
- `A` 乘的是 `V`；
- 然后结果过 `W_o`；
- 最后作为一条残差分支，加回原始 `x`。

所以传统 LLM 的 attention 分支更准确地说是：

> `LN(x)` 生成 `Q/K/V`，  
> `softmax(QK^T)` 作用在 `V` 上，  
> 再经过输出投影，  
> 最后加回主残差。

---

## 3. SEDD / DDiT Block 的 attention 分支

这份仓库里的 `DDiTBlock` 不是直接对 `LN(x)` 做 attention，而是先做一层由 `sigma` 控制的调制。

对应代码在 [model/transformer.py](./model/transformer.py)：

```python
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
```

然后 attention 分支是：

```python
x_skip = x
x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
qkv = self.attn_qkv(x)
...
x = flash_attn_varlen_qkvpacked_func(...)
x = self.attn_out(x)
x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)
```

如果写成公式，更接近：

\[
h_0 = \text{LN}(x)
\]

\[
\tilde h_0 = (1 + \text{scale}_{att}) \odot h_0 + \text{shift}_{att}
\]

\[
Q = W_q \tilde h_0,\qquad K = W_k \tilde h_0,\qquad V = W_v \tilde h_0
\]

\[
A = \text{softmax}(QK^\top / \sqrt{d})
\]

\[
\text{AttnOut} = A V
\]

\[
h = x + g_{att} \odot \text{Dropout}(W_o(\text{AttnOut}))
\]

因此它和传统 pre-LN LLM 相比，多了两件事：

1. **attention 前**：`LN(x)` 会被 `shift/scale` 调制；
2. **attention 后**：整个分支输出还会被 `gate` 控制注入强度。

---

## 4. 不能说“平移和 shift”

有时会口头说：

> hidden states 平移和 shift 之后做 attention

这里其实混用了两个词。

更准确地说是：

- `shift`：平移；
- `scale`：缩放。

所以 attention 前发生的是：

\[
\tilde h = (1 + \text{scale}) \odot \text{LN}(x) + \text{shift}
\]

而不是“平移和 shift”，因为 `shift` 本身就是平移。

---

## 5. `gate` 作用在哪里

这点最容易和 attention weights 混在一起。

### 5.1 `gate` 不作用在 attention weights 上

`gate` 不会直接改：

\[
A = \text{softmax}(QK^\top / \sqrt{d})
\]

也就是说：

- 它不改 `softmax`；
- 不改单个 token 对另一个 token 的注意力概率；
- 不直接干预 `QK^\top` 这一步。

### 5.2 `gate` 作用在整个 attention 分支输出上

在代码里：

```python
x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)
```

对应公式就是：

\[
h = x + g_{att} \odot \text{Dropout}(W_o(\text{AttnOut}))
\]

所以 `gate` 的作用是：

> 决定这条 attention 残差分支，最后有多少被加回主干。

因此更准确地说：

> `gate` 控制的是 attention branch 的 residual injection strength，  
> 而不是 attention weights 本身。

---

## 6. `shift / scale / gate` 各自分别在干嘛

### 6.1 `shift`

公式：

\[
h' = h + \text{shift}
\]

作用：

- 平移归一化后的特征；
- 改变每个通道的基线。

直觉：

> 在当前噪声层级下，有些特征整体应该更高或更低。

### 6.2 `scale`

公式：

\[
h' = (1 + \text{scale}) \odot h
\]

作用：

- 放大或缩小归一化后的特征；
- 改变每个通道的灵敏度。

直觉：

> 在当前噪声层级下，有些特征应该被更重视，有些应该被抑制。

### 6.3 `gate`

公式：

\[
\text{out} = \text{residual} + \text{gate} \odot \text{branch}
\]

作用：

- 决定一条分支输出最后能向主残差注入多少。

直觉：

> 它像一个更新力度旋钮，告诉模型当前噪声层级下，这一步该有多激进。

---

## 7. 一张并排对比表

### 7.1 传统 pre-LN LLM

\[
h = x + W_o(\text{Attn}(\text{LN}(x)))
\]

含义：

- 对 `LN(x)` 做 attention；
- attention 输出经过投影；
- 直接加回 `x`。

### 7.2 SEDD / DDiT Block

\[
h = x + g_{att} \odot W_o(\text{Attn}(\text{modulate}(\text{LN}(x), \text{shift}, \text{scale})))
\]

含义：

- 先用 `shift/scale` 调制 `LN(x)`；
- 再做 attention；
- attention 输出经过投影；
- 再由 `gate` 控制注入强度；
- 最后加回 `x`。

---

## 8. 更准确的口头表述

如果要把上面的比较说成一句准确的话，可以写成：

> 相比传统 pre-LN LLM，这份 SEDD 的 attention 分支不是直接对 `LN(x)` 做 attention，而是先用由 `sigma` 决定的 `shift/scale` 去调制 `LN(x)`，再生成 `Q/K/V` 并计算 attention；attention 输出经过 `W_o` 后，不是直接加回 residual，而是先经过 dropout 和一个由 `sigma` 决定的 `gate` 缩放，最后再与原始 hidden states 做残差相加。

---

## 9. 最后一句话总结

传统 LLM 里：

- attention branch 的主要结构是 `LN -> QKV -> softmax -> V -> W_o -> residual add`

而这份 SEDD / DDiT block 里：

- 在 `QKV` 之前增加了 `sigma` 条件下的 `shift/scale` 调制；
- 在 `W_o` 之后增加了 `sigma` 条件下的 `gate` 控制；
- 所以它不是直接改 attention weights，而是让整个 attention 分支的输入表示和输出注入强度都依赖于当前噪声层级。
