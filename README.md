# diffusion_journal_club_24
Repository of the journal club "Diffusion Models and Generative Modeling"


# Overview

| Date       | Theme                                   | Title                                                                                | Link                                      | Section-link                          | Responsible for Recap | Responsible for Questions |     |
| :--------- | --------------------------------------- | :----------------------------------------------------------------------------------- | :---------------------------------------- | :------------------------------------ | :-------------------- | ------------------------- | --- |
| 02.10.2024 | Introduction 1 : SDE view               | Elucidating the Design Space of Diffusion-Based Generative Models                    | [arxiv](https://arxiv.org/abs/2206.00364) |[Diffusion Models: the stochastic differential equation view](#Diffusion%20Models%20the%20stochastic%20differential%20equation%20view)|                  |                           |     |
| 09.10.2024 | Introduction 2: ELBO view               | Understanding the Diffusion Objective as a Weighted Integral of ELBOs                | [arxiv](https://arxiv.org/abs/2402.04384) |[In Detail](#in-detail)                                     |                       |                           |     |
| 16.10.2024 | Further Properties of DM                | On the Generalization Properties of Diffusion Models                                 | [arxiv](https://arxiv.org/abs/2311.01797) |                                       |                       |                           |     |
| 23.10.2024 | Sampler and Solver for Diffusion Models | SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models  | [arxiv](https://arxiv.org/abs/2305.14267) |                                       |                       |                           |     |
| 30.10.2024 | Bonus 1                                 | TBD                                                                                  | TBD                                       |                                       |                       |                           |     |
| 06.11.2024 | Flow Matching 1                         | Diffusion Schrödinger Bridge Matching                                                | [arxiv](https://arxiv.org/abs/2303.16852) |                                       |                       |                           |     |
| 13.11.2024 | Flow Matching 2                         | Flow Matching for Generative Modeling<br>                                            | [arxiv](https://arxiv.org/abs/2210.02747) |                                       |                       |                           |     |
| 20.11.2024 | Consistency models                      | Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion | [arxiv](https://arxiv.org/abs/2310.02279) |                                       |                       |                           |     |
| 27.11.2024 | Discrete Diffusion                      | Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution        | [arxiv](https://arxiv.org/abs/2310.16834) |                                       |                       |                           |     |
| 04.12.2024 | Bonus 2                                 | TBD                                                                                  | TBD                                       |                                       |                       |                           |     |


# In Detail

## Part 0: Useful Tutorials & Introductory Papers on Diffusion Models


## Part 1: Introduction to Diffusion Models

### Diffusion Models: the stochastic differential equation view

| **Titel**                 | Elucidating the Design Space of Diffusion-Based Generative Models |
| ------------------------- | ----------------------------------------------------------------- |
| **Date**                  | 02.10.2024                                                        |
| **Responsible Recap**     |                                                                   |
| **Responsible Questions** |                                                                   |
| **Link**                  | [arxiv](https://arxiv.org/abs/2206.00364)                         |

_Abstract_
> We argue that the theory and practice of diffusion-based generative models are currently unnecessarily convoluted and seek to remedy the situation by presenting a design space that clearly separates the concrete design choices. This lets us identify several changes to both the sampling and training processes, as well as preconditioning of the score networks. Together, our improvements yield new state-of-the-art FID of 1.79 for CIFAR-10 in a class-conditional setting and 1.97 in an unconditional setting, with much faster sampling (35 network evaluations per image) than prior designs. To further demonstrate their modular nature, we show that our design changes dramatically improve both the efficiency and quality obtainable with pre-trained score networks from previous work, including improving the FID of a previously trained ImageNet-64 model from 2.07 to near-SOTA 1.55, and after re-training with our proposed improvements to a new SOTA of 1.36.

**Further Reading**
- Tutorial: Denoising Diffusion Probabilistic Models in Six Simple Steps [https://arxiv.org/abs/2402.04384](https://arxiv.org/abs/2402.04384)

### Diffusion Models: the ELBO view

| **Titel**                 | Understanding the Diffusion Objective as a Weighted Integral of ELBOs |
| ------------------------- | --------------------------------------------------------------------- |
| **Date**                  | 09.10.2024                                                            |
| **Responsible Recap**     |                                                                       |
| **Responsible Questions** |                                                                       |
| **Link**                  | [arxiv](https://arxiv.org/abs/2402.04384)                             |

_Abstract_
> To achieve the highest perceptual quality, state-of-the-art diffusion models are optimized with objectives that typically look very different from the maximum likelihood and the Evidence Lower Bound (ELBO) objectives. In this work, we reveal that diffusion model objectives are actually closely related to the ELBO.  
Specifically, we show that all commonly used diffusion model objectives equate to a weighted integral of ELBOs over different noise levels, where the weighting depends on the specific objective used. Under the condition of monotonic weighting, the connection is even closer: the diffusion objective then equals the ELBO, combined with simple data augmentation, namely Gaussian noise perturbation. We show that this condition holds for a number of state-of-the-art diffusion models.  
In experiments, we explore new monotonic weightings and demonstrate their effectiveness, achieving state-of-the-art FID scores on the high-resolution ImageNet benchmark.

## Part 2: Further Properties of Diffusion Models



## Part 3: Flow Matching

| **Titel**                 |     |
| ------------------------- | --- |
| **Date**                  |     |
| **Responsible Recap**     |     |
| **Responsible Questions** |     |
| **Link**                  |     |

_Abstract_
> Contents

| **Titel**                 |     |
| ------------------------- | --- |
| **Date**                  |     |
| **Responsible Recap**     |     |
| **Responsible Questions** |     |
| **Link**                  |     |

> [!NOTE] Abstract
> Contents

| **Titel**                 |     |
| ------------------------- | --- |
| **Date**                  |     |
| **Responsible Recap**     |     |
| **Responsible Questions** |     |
| **Link**                  |     |

> [!NOTE] Abstract
> Contents

## Template
| **Titel**                 |     |
| ------------------------- | --- |
| **Date**                  |     |
| **Responsible Recap**     |     |
| **Responsible Questions** |     |
| **Link**                  |     |

> [!NOTE] Abstract
> Contents
