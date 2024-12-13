---
title: kfa
date: 2024-11-06 20:35:37
tags:
categories: Kronecker-factored Approximate
---

- [约定](#约定)
- [目标](#目标)
  - [证明](#证明)
- [Kronecker](#kronecker)
  - [Definition](#definition)
  - [Properties](#properties)
- [Useful formula](#useful-formula)

## 约定

默认所有向量都是列向量

## 目标

求$F^{-1}v$

Fisher 信息阵$F = E\left[\frac{\partial L}{\partial \theta}(\frac{\partial L}{\partial \theta})^T\right]$

### 证明

把神经网络的每个权重矩阵逐列向量化，然后按一个确定的顺序排列这些向量，得到一个长向量$\theta$
对于$y = Wx + b$, 令$g = \frac{\partial L}{\partial y}$

$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot x^T
$$

$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} \cdot I
$$

把 $x$ 和 $I$ 在后文中用$a$表示

$$
F =
\begin{matrix}
E[\frac{\partial L}{\partial \theta} \cdot \frac{\partial L}{\partial \theta}^T]
\end{matrix} =
\begin{bmatrix}
    E[\frac{\partial L}{\partial \theta_1} \cdot \frac{\partial L}{\partial \theta_1}^T]
    E[\frac{\partial L}{\partial \theta_1} \cdot \frac{\partial L}{\partial \theta_2}^T]\cdots & E[\frac{\partial L}{\partial \theta_1} \cdot \frac{\partial L}{\partial \theta_n}^T]\\
    E[\frac{\partial L}{\partial \theta_2} \cdot \frac{\partial L}{\partial \theta_1}^T]  E[\frac{\partial L}{\partial \theta_2} \cdot \frac{\partial L}{\partial \theta_2}^T]  \cdots & E[\frac{\partial L}{\partial \theta_2} \cdot \frac{\partial L}{\partial \theta_n}^T] \\
    \vdots &\vdots\\
    E[\frac{\partial L}{\partial \theta_n} \cdot \frac{\partial L}{\partial \theta_1}^T]  E[\frac{\partial L}{\partial \theta_n} \cdot \frac{\partial L}{\partial \theta_2}^T]  \cdots & E[\frac{\partial L}{\partial \theta_n} \cdot \frac{\partial L}{\partial \theta_n}^T]
\end{bmatrix}
$$

$\theta_i$ 表示按前述顺序排列的第 i 个权重矩阵的向量化表示。
取$F$的对角线上的矩阵作为$F$的近似。

$$
\begin{align*}
E[\frac{\partial L}{\partial \theta_i}\cdot \frac{\partial L}{\partial \theta_i}^T] &=
E[vec(g_i \cdot a_i^T)vec(g_i \cdot a_i^T)^T] \\
&= E[(a_i \otimes g_i)(a_i^T \otimes g_i^T)] \\
&= E[(a_ia_i^T)\otimes(g_ig_i^T)]\\
&\approx E[a_ia_i^T] \otimes E[g_ig_i^T] \tag{未证明}\\
&=:A_i \otimes G_i
\end{align*}
$$

$$
\begin{align*}
F &\approx diag(A_1\otimes G_1, A_2 \otimes G_2, \ldots, A_n \otimes G_n) \tag{未证明}\\
&=
\begin{pmatrix}
    A_1 \otimes G_1, &0,&0,&\cdots,&0\\
    0, &A_2 \otimes G_2, &0, &\cdots, &0\\
    0, &0, &A_3 \otimes G_3, &\cdots, &0\\
    \vdots, &\vdots, &\vdots, &\ddots, &\vdots\\
    0, &0, &0, &\cdots, &A_n \otimes G_n
\end{pmatrix}
\end{align*}
$$

把$A_i \otimes G_i$记作$F_i$
$A_i$ 和 $G_i$ 都是对称矩阵，对称矩阵的逆矩阵为对称矩阵。

$$
\begin{align*}
F_i^{-1}\cdot v_i &\approx (A_i^{-1}\otimes G_i^{-1})vec(V_i)\\
&= vec(G_i^{-1}V_i(A_i^{-1})^T)\\
&= vec(G_i^{-1}V_iA_i^{-1})\\
&= vec(E[g_i \cdot g_i^T]^{-1}V_iE[a_i \cdot a_i^T]^{-1})
\end{align*}
$$

## Kronecker

### Definition

$$
A\otimes B= \begin{pmatrix}
a_{11}B & a_{12}B & \cdots & a_{1n}B \\
a_{21}B & a_{22}B & \cdots & a_{2n}B \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1}B & a_{m2}B & \cdots & a_{mn}B
\end{pmatrix}
$$

### Properties

$$
vec(uv^T) = v \otimes u
$$

$$
(A \otimes B)^T = (A^T \otimes B^T)
$$

$$
(A \otimes B)(A^T \otimes B^T) = AA^T \otimes BB^T
$$

$$
(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}
$$

$$
(A \otimes B)vec(X) = vec(BXA^T)
$$

## Useful formula

$$
(A+bc^T)^{-1} = A^{-1} - \frac{A^{-1}bc^TA^{-1}}{1+c^TA^{-1}b}\\
(A+bc^T)^{-1}d= A^{-1}d - \frac{A^{-1}bc^TA^{-1}d}{1+c^TA^{-1}b}
$$

$$
\begin{align*}
(F - \frac{\partial L}{\partial \theta} \frac{\partial L}{\partial \theta}^T)^{-1}d &=
F^{-1}d + \frac{F^{-1}\frac{\partial L}{\partial \theta} \frac{\partial L}{\partial \theta}^TF^{-1}d}{1-\frac{\partial L}{\partial \theta}^TF^{-1}\frac{\partial L}{\partial \theta}}\\
&= F^{-1}d + (\frac{\frac{\partial L}{\partial \theta}^TF^{-1}d}{1-\frac{\partial L}{\partial \theta}^TF^{-1}\frac{\partial L}{\partial \theta}})F^{-1}\frac{\partial L}{\partial \theta}\\
&= F^{-1}d + (\frac{\frac{\partial L}{\partial \theta}^T(F^{-1}d)}{1-\frac{\partial L}{\partial \theta}^T(F^{-1}\frac{\partial L}{\partial \theta})})(F^{-1}\frac{\partial L}{\partial \theta})\\
\end{align*}
$$

![kfa_optimizer.py](yc.png)
