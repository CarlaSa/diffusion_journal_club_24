

# Overview


| Date       | Theme                                   | Title                                                                                                                          | Link                                      | Section-link        | Responsible for Recap | Responsible for Questions |
|:---------- | --------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------ |:----------------------------------------- |:------------------- |:--------------------- | ------------------------- |
| 02.10.2024 | Introduction 1 : SDE view               | Elucidating the Design Space of Diffusion-Based Generative Models                                                              | [arxiv](https://arxiv.org/abs/2206.00364) | [Week 1](#week-1)   |                  |                           |
| 09.10.2024 | Introduction 2: ELBO view               | Understanding the Diffusion Objective as a Weighted Integral of ELBOs                                                          | [arxiv](https://arxiv.org/abs/2402.04384) | [Week 2](#week-2)   |                       |                           |
| 16.10.2024 | Further Properties of DM                | On the Generalization Properties of Diffusion Models                                                                           | [arxiv](https://arxiv.org/abs/2311.01797) | [Week 3](#week-3)   |                       |                           |
| 23.10.2024 | Sampler and Solver for Diffusion Models | SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models                                            | [arxiv](https://arxiv.org/abs/2305.14267) | [Week 4](#week-4)   |                       |                           |
| 30.10.2024 | Bonus 1                                 | TBD                                                                                                                            | TBD                                       | [Week 5](#week-5)   |                       |                           |
| 06.11.2024 | Flow Matching 1                         | Diffusion Schrödinger Bridge Matching                                                                                          | [arxiv](https://arxiv.org/abs/2303.16852) | [Week 6](#week-6)   |                       |                           |
| 13.11.2024 | Flow Matching 2                         | Flow Matching for Generative Modeling                                                                                          | [arxiv](https://arxiv.org/abs/2210.02747) | [Week 7](#week-7)   |                       |                           |
|            |                                         | Scaling Rectified Flow Transformers for High-Resolution Image Synthesis              [arxiv](https://arxiv.org/abs/2403.03206) |                                           |                     |                       |                           |
| 20.11.2024 | Consistency models                      | Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion                                           | [arxiv](https://arxiv.org/abs/2310.02279) | [Week 8](#week-8)   |                       |                           |
| 27.11.2024 | Discrete Diffusion                      | Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution                                                  | [arxiv](https://arxiv.org/abs/2310.16834) | [Week 9](#week-9)   |                       |                           |
| 04.12.2024 | Bonus 2                                 | TBD                                                                                                                            | TBD                                       | [Week 10](#week-10) |                       |                           |

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

**Further reading**
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

**Further reading**
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

**Further reading**
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

**Further reading**
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
## Flow Matching Part 1

| **Titel**                 |   Diffusion Schrödinger Bridge Matching     |
| ------------------------- | --- |
| **Date**                  | 06.11.2024    |
| **Authors**               | Yuyang Shi, Valentin De Bortoli, Andrew Campbell, Arnaud Doucet    |
| **Responsible Recap**     |     |
| **Responsible Questions** |     |
| **Link**                  |   [arxiv](https://arxiv.org/abs/2210.02747)  |

_Abstract_
> Solving transport problems, i.e. finding a map transporting one given distribution to another, has numerous applications in machine learning. Novel mass transport methods motivated by generative modeling have recently been proposed, e.g. Denoising Diffusion Models (DDMs) and Flow Matching Models (FMMs) implement such a transport through a Stochastic Differential Equation (SDE) or an Ordinary Differential Equation (ODE). However, while it is desirable in many applications to approximate the deterministic dynamic Optimal Transport (OT) map which admits attractive properties, DDMs and FMMs are not guaranteed to provide transports close to the OT map. In contrast, Schrödinger bridges (SBs) compute stochastic dynamic mappings which recover entropy-regularized versions of OT. Unfortunately, existing numerical methods approximating SBs either scale poorly with dimension or accumulate errors across iterations. In this work, we introduce Iterative Markovian Fitting (IMF), a new methodology for solving SB problems, and Diffusion Schrödinger Bridge Matching (DSBM), a novel numerical algorithm for computing IMF iterates. DSBM significantly improves over previous SB numerics and recovers as special/limiting cases various recent transport methods. We demonstrate the performance of DSBM on a variety of problems. 



# Week 7
## Flow Matching Part 2

After covering the math with a lot of detail last week, this week we will consider two papers more broadly.

| **Titel**                 | Flow Matching for Generative Modeling         |
| ------------------------- | --- |
| **Date**                  |  13.11.2024   |
| **Authors**               |    Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le |
| **Responsible Recap**     |     |
| **Responsible Questions** |     |
| **Link**                  | [arxiv](https://arxiv.org/abs/2210.02747)    |

_Abstract_
> We introduce a new paradigm for generative modeling built on Continuous Normalizing Flows (CNFs), allowing us to train CNFs at unprecedented scale. Specifically, we present the notion of Flow Matching (FM), a simulation-free approach for training CNFs based on regressing vector fields of fixed conditional probability paths. Flow Matching is compatible with a general family of Gaussian probability paths for transforming between noise and data samples -- which subsumes existing diffusion paths as specific instances. Interestingly, we find that employing FM with diffusion paths results in a more robust and stable alternative for training diffusion models. Furthermore, Flow Matching opens the door to training CNFs with other, non-diffusion probability paths. An instance of particular interest is using Optimal Transport (OT) displacement interpolation to define the conditional probability paths. These paths are more efficient than diffusion paths, provide faster training and sampling, and result in better generalization. Training CNFs using Flow Matching on ImageNet leads to consistently better performance than alternative diffusion-based methods in terms of both likelihood and sample quality, and allows fast and reliable sample generation using off-the-shelf numerical ODE solvers.


| **Titel**                 | Scaling Rectified Flow Transformers for High-Resolution Image Synthesis   |
| ------------------------- | --- |
| **Date**                  |  13.11.2024   |
| **Authors**               |   Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, Robin Rombach

|
| **Responsible Recap**     |     |
| **Responsible Questions** |     |
| **Link**                  | [arxiv](https://arxiv.org/abs/2403.03206)    |

_Abstract_
> Diffusion models create data from noise by inverting the forward paths of data towards noise and have emerged as a powerful generative modeling technique for high-dimensional, perceptual data such as images and videos. Rectified flow is a recent generative model formulation that connects data and noise in a straight line. Despite its better theoretical properties and conceptual simplicity, it is not yet decisively established as standard practice. In this work, we improve existing noise sampling techniques for training rectified flow models by biasing them towards perceptually relevant scales. Through a large-scale study, we demonstrate the superior performance of this approach compared to established diffusion formulations for high-resolution text-to-image synthesis. Additionally, we present a novel transformer-based architecture for text-to-image generation that uses separate weights for the two modalities and enables a bidirectional flow of information between image and text tokens, improving text comprehension, typography, and human preference ratings. We demonstrate that this architecture follows predictable scaling trends and correlates lower validation loss to improved text-to-image synthesis as measured by various metrics and human evaluations. Our largest models outperform state-of-the-art models, and we will make our experimental data, code, and model weights publicly available.

# Week 8
## Consistency models

| **Titel**                 | Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion    |
| ------------------------- | --- |
| **Date**                  |  20.11.2024   |
| **Authors**               |  Dongjun Kim, Chieh-Hsin Lai, Wei-Hsiang Liao, Naoki Murata, Yuhta Takida, Toshimitsu Uesaka, Yutong He, Yuki Mitsufuji, Stefano Ermon   |
| **Responsible Recap**     |     |
| **Responsible Questions** |     |
| **Link**                  | [arxiv](https://arxiv.org/abs/2310.02279)     |

_Abstract_
> Consistency Models (CM) (Song et al., 2023) accelerate score-based diffusion model sampling at the cost of sample quality but lack a natural way to trade-off quality for speed. To address this limitation, we propose Consistency Trajectory Model (CTM), a generalization encompassing CM and score-based models as special cases. CTM trains a single neural network that can -- in a single forward pass -- output scores (i.e., gradients of log-density) and enables unrestricted traversal between any initial and final time along the Probability Flow Ordinary Differential Equation (ODE) in a diffusion process. CTM enables the efficient combination of adversarial training and denoising score matching loss to enhance performance and achieves new state-of-the-art FIDs for single-step diffusion model sampling on CIFAR-10 (FID 1.73) and ImageNet at 64x64 resolution (FID 1.92). CTM also enables a new family of sampling schemes, both deterministic and stochastic, involving long jumps along the ODE solution trajectories. It consistently improves sample quality as computational budgets increase, avoiding the degradation seen in CM. Furthermore, unlike CM, CTM's access to the score function can streamline the adoption of established controllable/conditional generation methods from the diffusion community. This access also enables the computation of likelihood. The code is available at this https URL. 


# Week 9

## Discrete Diffusion 

| **Titel**                 | Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution |
| ------------------------- | ----------------------------------------------------------------------------- |
| **Date**                  | 27.11.2024                                                                    |
| **Authors**               | Aaron Lou, Chenlin Meng, Stefano Ermon                                        |
| **Responsible Recap**     |                                                                               |
| **Responsible Questions** |                                                                               |
| **Link**                  | [arxiv](https://arxiv.org/abs/2310.16834)                                     |

_Abstract_
> Despite their groundbreaking performance for many generative modeling tasks, diffusion models have fallen short on discrete data domains such as natural language. Crucially, standard diffusion models rely on the well-established theory of score matching, but efforts to generalize this to discrete structures have not yielded the same empirical gains. In this work, we bridge this gap by proposing score entropy, a novel loss that naturally extends score matching to discrete spaces, integrates seamlessly to build discrete diffusion models, and significantly boosts performance. Experimentally, we test our Score Entropy Discrete Diffusion models (SEDD) on standard language modeling tasks. For comparable model sizes, SEDD beats existing language diffusion paradigms (reducing perplexity by 25-75\%) and is competitive with autoregressive models, in particular outperforming GPT-2. Furthermore, compared to autoregressive mdoels, SEDD generates faithful text without requiring distribution annealing techniques like temperature scaling (around 6-8× better generative perplexity than un-annealed GPT-2), can trade compute and quality (similar quality with 32× fewer network evaluations), and enables controllable infilling (matching nucleus sampling quality while enabling other strategies besides left to right prompting). 

# Week 10
## TBD

| **Titel**                 |     |
| ------------------------- | --- |
| **Date**                  | 04.12.2024    |
| **Authors**               |     |
| **Responsible Recap**     |     |
| **Responsible Questions** |     |
| **Link**                  |     |

_Abstract_
> Contents

