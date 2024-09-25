
# Overview

| Date       | Theme                                   | Title                                                                                | Link                                      | Section-link        | Responsible for Recap | Responsible for Questions |
|:---------- | --------------------------------------- |:------------------------------------------------------------------------------------ |:----------------------------------------- |:------------------- |:--------------------- | ------------------------- |
| 02.10.2024 | Introduction 1 : SDE view               | Elucidating the Design Space of Diffusion-Based Generative Models                    | [arxiv](https://arxiv.org/abs/2206.00364) | [Week 1](#week-1)   | Carla                 |                           |
| 09.10.2024 | Introduction 2: ELBO view               | Understanding the Diffusion Objective as a Weighted Integral of ELBOs                | [arxiv](https://arxiv.org/abs/2402.04384) | [Week 2](#week-2)   |                       |                           |
| 16.10.2024 | Further Properties of DM                | On the Generalization Properties of Diffusion Models                                 | [arxiv](https://arxiv.org/abs/2311.01797) | [Week 3](#week-3)   |                       |                           |
| 23.10.2024 | Sampler and Solver for Diffusion Models | SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models  | [arxiv](https://arxiv.org/abs/2305.14267) | [Week 4](#week-4)   |                       |                           |
| 30.10.2024 | Bonus 1                                 | TBD                                                                                  | TBD                                       | [Week 5](#week-5)   |                       |                           |
| 06.11.2024 | Flow Matching 1                         | Diffusion Schrödinger Bridge Matching                                                | [arxiv](https://arxiv.org/abs/2303.16852) | [Week 6](#week-6)   |                       |                           |
| 13.11.2024 | Flow Matching 2                         | Flow Matching for Generative Modeling                                                | [arxiv](https://arxiv.org/abs/2210.02747) | [Week 7](#week-7)   |                       |                           |
| 20.11.2024 | Consistency models                      | Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion | [arxiv](https://arxiv.org/abs/2310.02279) | [Week 8](#week-8)   |                       |                           |
| 27.11.2024 | Discrete Diffusion                      | Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution        | [arxiv](https://arxiv.org/abs/2310.16834) | [Week 9](#week-9)   |                       |                           |
| 04.12.2024 | Bonus 2                                 | TBD                                                                                  | TBD                                       | [Week 10](#week-10) |                       |                           |

# Week 0
## Interesting Tutorials & blog articles 




# Week 1
## Diffusion Models: the stochastic differential equation view

| **Titel**                 | Elucidating the Design Space of Diffusion-Based Generative Models |
| ------------------------- | ----------------------------------------------------------------- |
| **Date**                  | 02.10.2024                                                        |
| **Authors**               | Tero Karras, Miika Aittala, Timo Aila, Samuli Laine                                                                  |
| **Responsible Recap**     |                                                                   |
| **Responsible Questions** |                                                                   |
| **Link**                  | [arxiv](https://arxiv.org/abs/2206.00364)                         |

_Abstract_
> We argue that the theory and practice of diffusion-based generative models are currently unnecessarily convoluted and seek to remedy the situation by presenting a design space that clearly separates the concrete design choices. This lets us identify several changes to both the sampling and training processes, as well as preconditioning of the score networks. Together, our improvements yield new state-of-the-art FID of 1.79 for CIFAR-10 in a class-conditional setting and 1.97 in an unconditional setting, with much faster sampling (35 network evaluations per image) than prior designs. To further demonstrate their modular nature, we show that our design changes dramatically improve both the efficiency and quality obtainable with pre-trained score networks from previous work, including improving the FID of a previously trained ImageNet-64 model from 2.07 to near-SOTA 1.55, and after re-training with our proposed improvements to a new SOTA of 1.36.

**Further Reading**
- Tutorial: [Denoising Diffusion Probabilistic Models in Six Simple Steps](https://arxiv.org/abs/2402.04384)

# Week 2
### Diffusion Models: the ELBO view

| **Titel**                 | Understanding the Diffusion Objective as a Weighted Integral of ELBOs |
| ------------------------- | --------------------------------------------------------------------- |
| **Date**                  | 09.10.2024                                                            |
| **Authors**               | Diederik P. Kingma, Ruiqi Gao                                                                      |
| **Responsible Recap**     |                                                                       |
| **Responsible Questions** |                                                                       |
| **Link**                  | [arxiv](https://arxiv.org/abs/2402.04384)                             |

_Abstract_
> To achieve the highest perceptual quality, state-of-the-art diffusion models are optimized with objectives that typically look very different from the maximum likelihood and the Evidence Lower Bound (ELBO) objectives. In this work, we reveal that diffusion model objectives are actually closely related to the ELBO.  
> Specifically, we show that all commonly used diffusion model objectives equate to a weighted integral of ELBOs over different noise levels, where the weighting depends on the specific objective used. Under the condition of monotonic weighting, the connection is even closer: the diffusion objective then equals the ELBO, combined with simple data augmentation, namely Gaussian noise perturbation. We show that this condition holds for a number of state-of-the-art diffusion models.  
> In experiments, we explore new monotonic weightings and demonstrate their effectiveness, achieving state-of-the-art FID scores on the high-resolution ImageNet benchmark.

**Further Reading**
- [Variational Diffusion Models](http://arxiv.org/abs/2107.00630)

# Week 3
## Titel

| **Titel**                 | On the Generalization Properties of Diffusion Models |
| ------------------------- | ---------------------------------------------------- |
| **Date**                  | 16.10.2024                                           |
| **Authors**               | Puheng Li, Zhong Li, Huishuai Zhang, Jiang Bian      |
| **Responsible Recap**     |                                                      |
| **Responsible Questions** |                                                      |
| **Link**                  | [arxiv](https://arxiv.org/abs/2311.01797)            |

_Abstract_
> Diffusion models are a class of generative models that serve to establish a stochastic transport map between an empirically observed, yet unknown, target distribution and a known prior. Despite their remarkable success in real-world applications, a theoretical understanding of their generalization capabilities remains underdeveloped. This work embarks on a comprehensive theoretical exploration of the generalization attributes of diffusion models. We establish theoretical estimates of the generalization gap that evolves in tandem with the training dynamics of score-based diffusion models, suggesting a polynomially small generalization error (O(n−2/5+m−4/5)) on both the sample size n and the model capacity m, evading the curse of dimensionality (i.e., not exponentially large in the data dimension) when early-stopped. Furthermore, we extend our quantitative analysis to a data-dependent scenario, wherein target distributions are portrayed as a succession of densities with progressively increasing distances between modes. This precisely elucidates the adverse effect of "modes shift" in ground truths on the model generalization. Moreover, these estimates are not solely theoretical constructs but have also been confirmed through numerical simulations. Our findings contribute to the rigorous understanding of diffusion models' generalization properties and provide insights that may guide practical applications.

**further reading**
- [How Diffusion Models Learn to Factorize and Compose](https://arxiv.org/abs/2408.13256)


# Week 4
## Titel

| **Titel**                 |SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models     |
| ------------------------- | --- |
| **Date**                  |     |
| **Authors**               |  
Martin Gonzalez, Nelson Fernandez, Thuy Tran, Elies Gherbi, Hatem Hajri, Nader Masmoudi   |
| **Responsible Recap**     |     |
| **Responsible Questions** |     |
| **Link**                  |  [arxiv](https://arxiv.org/abs/2305.14267)   |

_Abstract_
> A potent class of generative models known as Diffusion Probabilistic Models (DPMs) has become prominent. A forward diffusion process adds gradually noise to data, while a model learns to gradually denoise. Sampling from pre-trained DPMs is obtained by solving differential equations (DE) defined by the learnt model, a process which has shown to be prohibitively slow. Numerous efforts on speeding-up this process have consisted on crafting powerful ODE solvers. Despite being quick, such solvers do not usually reach the optimal quality achieved by available slow SDE solvers. Our goal is to propose SDE solvers that reach optimal quality without requiring several hundreds or thousands of NFEs to achieve that goal. We propose Stochastic Explicit Exponential Derivative-free Solvers (SEEDS), improving and generalizing Exponential Integrator approaches to the stochastic case on several frameworks. After carefully analyzing the formulation of exact solutions of diffusion SDEs, we craft SEEDS to analytically compute the linear part of such solutions. Inspired by the Exponential Time-Differencing method, SEEDS use a novel treatment of the stochastic components of solutions, enabling the analytical computation of their variance, and contains high-order terms allowing to reach optimal quality sampling ∼3-5× faster than previous SDE methods. We validate our approach on several image generation benchmarks, showing that SEEDS outperform or are competitive with previous SDE solvers. Contrary to the latter, SEEDS are derivative and training free, and we fully prove strong convergence guarantees for them.

**Further Reading**
- [Fast Sampling of Diffusion Models with Exponential Integrator](https://arxiv.org/abs/2204.13902)


# Week 5

TBD

## Titel

| **Titel**                 |     |
| ------------------------- | --- |
| **Date**                  |     |
| **Authors**               |     |
| **Responsible Recap**     |     |
| **Responsible Questions** |     |
| **Link**                  |     |

_Abstract_
> Contents

# Week 6
## Titel


| **Titel**                 |     |
| ------------------------- | --- |
| **Date**                  |     |
| **Authors**               |     |
| **Responsible Recap**     |     |
| **Responsible Questions** |     |
| **Link**                  |     |

_Abstract_
> Contents



# Week 7
## Flow Matching

| **Titel**                 |     |
| ------------------------- | --- |
| **Date**                  |     |
| **Authors**               |     |
| **Responsible Recap**     |     |
| **Responsible Questions** |     |
| **Link**                  |     |

_Abstract_
> Contents

# Week 8
## Titel

| **Titel**                 |     |
| ------------------------- | --- |
| **Date**                  |     |
| **Authors**               |     |
| **Responsible Recap**     |     |
| **Responsible Questions** |     |
| **Link**                  |     |

_Abstract_
> Contents


# Week 9

## Titel

| **Titel**                 |     |
| ------------------------- | --- |
| **Date**                  |     |
| **Authors**               |     |
| **Responsible Recap**     |     |
| **Responsible Questions** |     |
| **Link**                  |     |

_Abstract_
> Contents

# Week 10

| **Titel**                 |     |
| ------------------------- | --- |
| **Date**                  |     |
| **Authors**               |     |
| **Responsible Recap**     |     |
| **Responsible Questions** |     |
| **Link**                  |     |

_Abstract_
> Contents
