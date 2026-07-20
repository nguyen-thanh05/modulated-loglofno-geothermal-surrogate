This plan is organized around one claim: **reliable geothermal surrogate modeling requires alignment between architecture, conditioning, and the training objective**. The proposed method combines a modulated local-global Fourier neural operator (Modulated LOGLO-FNO) with a rollout-aware composite objective.

The terminology in the paper must distinguish:

- **One-step MSE baseline:** one-step weighted MSE only, with pushforward training disabled.
- **Rollout-aware composite objective:** weighted MSE + H1 + mass-balance error (MBE) + radial spectral loss, applied to one-step predictions and pushforward states.
- **Full method:** Modulated LOGLO-FNO trained with the rollout-aware composite objective.

Because the MSE-only configuration changes both loss composition and pushforward training, comparisons against it estimate the effect of the **complete training objective**, not the effect of an individual loss term.

---



# 1. Core paper narrative

The paper should be framed around this central problem:

**A surrogate must represent coupled global and local reservoir dynamics, respond correctly to controls and geology, and remain stable when used autoregressively. A one-step pointwise MSE objective provides only part of the supervision required for this task.**

The answer has two complementary parts:

1. **Architecture:** Modulated LOGLO-FNO combines local-global operator blocks with explicit modulation by actions and static reservoir properties. The central architectural hypothesis is that modulation provides more effective conditioning than concatenation and makes the learned dynamics more robust.
2. **Training objective:** The rollout-aware composite objective supplies complementary signals for pointwise accuracy, spatial derivatives, mass balance, spectral structure, and off-manifold states encountered during rollout.

The main result is the interaction of these parts: **Modulated LOGLO-FNO trained with the rollout-aware composite objective outperforms all evaluated baselines.** The one-step MSE experiments add an important robustness result: in the initial seed, Modulated LOGLO-FNO converges, although to worse performance than the full method, whereas vanilla LOGLO-FNO and U-Net do not train successfully.

This supports a more precise story than “the composite loss improves every model”:

- modulation is useful beyond the composite objective because Modulated LOGLO-FNO remains trainable under one-step MSE;
- the composite objective provides a substantial additional gain for Modulated LOGLO-FNO;
- the objective appears to improve optimization robustness across architectures, but this claim requires multi-seed confirmation;
- architecture and objective are complementary rather than interchangeable.



## 1.1 Evidence status and claim discipline

The manuscript should separate observations from confirmed claims.

**Current preliminary observations**

- Under the rollout-aware composite objective, Modulated LOGLO-FNO performs better than all evaluated baselines.
- Under one-step MSE, converged models attain less optimal performance than their composite-objective counterparts.
- In the first MSE-only seed, U-Net and vanilla LOGLO-FNO do not train successfully, while Modulated LOGLO-FNO does.

**Claims allowed only after replication**

- “consistently outperforms” requires results across the planned seeds;
- “improves convergence” requires a predefined convergence criterion and success rate across seeds;
- “the loss term causes the gain” requires component and pushforward ablations;
- “improves U-Net” should not be used when the MSE baseline did not converge; use “enables successful training” only if replicated.

Use “in the initial seed” for the current MSE-only finding until additional seeds finish.

---



# 2. Suggested paper structure

1. **Introduction**
2. **Problem setting and benchmark**
3. **Method**
  - Modulated LOGLO-FNO
  - Rollout-aware composite objective
4. **Experimental protocol**
  - Models, parameter counts, and training controls
  - One-step and autoregressive metrics
  - Seed and convergence protocol
5. **Results**
  - Full-method comparison under the common composite objective
  - One-step MSE baseline and training robustness
  - Pushforward and loss-component ablations
  - Rollout and qualitative analysis
6. **Discussion and limitations**
7. **Conclusion**

This order makes the argument cumulative. First establish that the full method is best under a common training protocol. Then use the one-step MSE experiment and targeted ablations to explain why. Do not lead with the failed MSE runs; they are supporting evidence about robustness, not the headline result.

---



# 3. Introduction plan

The introduction should move through four steps.

## 3.1 Start from the application need

Open with the need for fast surrogate models in reservoir simulation. Keep it broad but specific enough.

For your geothermal / reservoir setting, you can say that numerical simulators support forecasting, optimization, history matching, uncertainty analysis, and control, but repeated high-fidelity simulation is expensive.

Then transition to neural operators:

Neural operators, including FNO-style models, are attractive because they learn mappings between fields and can predict full spatiotemporal states. However, field prediction in reservoir systems is difficult because the model must represent smooth large-scale evolution and localized high-gradient dynamics at the same time.

## 3.2 Identify the coupled limitation

The limitation is not that FNO or MSE is intrinsically inadequate. It is that the architecture, conditioning mechanism, and objective each expose a different failure mode:

- a purely global or purely local representation may miss cross-scale reservoir dynamics;
- concatenating controls with state fields may not condition every operator block effectively;
- one-step MSE does not directly supervise spatial derivatives, mass balance, frequency content, or states reached after model-generated rollout.

A strong phrasing is:

> Accurate geothermal surrogates must resolve interactions across spatial scales, respond to sparse controls and heterogeneous geology, and remain accurate when their own predictions are fed back as inputs. Conventional one-step MSE training directly rewards average pointwise accuracy, but does not explicitly constrain derivative structure, mass balance, spectral content, or off-manifold rollout behavior. This creates a joint representation-and-optimization problem rather than a loss-only problem.



## 3.3 Introduce your response to the limitation

Introduce the response in the same order:

1. local-global Fourier blocks model interactions at complementary spatial scales;
2. action and static-property embeddings modulate the operator blocks rather than being used only through input concatenation;
3. the rollout-aware composite objective supervises pointwise, derivative, mass-balance, and spectral behavior at one-step and pushforward states.

The architecture builds on prior local-global operator ideas. The novelty claim should therefore focus on the **modulated adaptation, its use for controlled geothermal dynamics, and its integration with the training objective**, not on inventing an entirely new operator family.

## 3.4 State contributions explicitly

Suggested contribution structure:

1. **Modulated local-global operator.**
  We adapt LOGLO-FNO to controlled geothermal dynamics by modulating its operator blocks with actions and static reservoir properties. This directly tests modulation against the concatenation-based vanilla LOGLO-FNO.
2. **Rollout-aware composite training.**
  We combine weighted MSE, H1, mass-balance, and radial spectral objectives and apply them during both one-step and pushforward training, targeting complementary field, structural, physical, and temporal error modes.
3. **Controlled architecture–objective study.**
  We compare Modulated LOGLO-FNO with vanilla LOGLO-FNO, FNO, U-FNO, UNO, and U-Net using common data, budgets, and evaluation, with a documented model-specific optimization policy. We separate the full-package result from pushforward and component-level effects.
4. **Empirical finding.**
  The full method achieves the best preliminary performance among all evaluated baselines. Initial one-step MSE results further indicate that modulation improves robustness to objective simplification, while the composite objective provides an additional performance and optimization benefit. State this as a consistent finding only after multi-seed confirmation.

Possible contribution paragraph:

> We study autoregressive neural surrogates for coupled geothermal reservoir dynamics. Our method adapts a local-global Fourier neural operator by using reservoir controls and static properties to modulate its operator blocks, and trains it with a rollout-aware objective combining pointwise, Sobolev, mass-balance, and spectral errors. Under the common composite-objective protocol, Modulated LOGLO-FNO achieves the best performance among the evaluated neural-operator and convolutional baselines. A controlled one-step MSE comparison and targeted objective ablations show that the architecture and training objective play complementary roles: modulation improves robustness to simplified supervision, while rollout-aware composite training produces the strongest final surrogate.

---



# 4. Problem setting and benchmark section

This should come before the method because the task motivates both the architecture and objective.

## 4.1 Define the prediction task

Define the actual autoregressive task, not a menu of possible formulations.

Let the normalized state be


\mathbf{y}*t =
[T*{m,t}, T_{f,t}, P_{m,t}, P_{f,t}],


containing matrix/formation and fracture temperature and pressure. Let \mathbf{a}_t denote the well control field and mask, and let


\mathbf{m}=[\phi_m,\phi_f,k_m,k_f]


contain the static matrix/fracture porosity and permeability fields in the heterogeneous setting. The model predicts the next state:


\hat{\mathbf{y}}*{t+1}
= f*\theta(\mathbf{y}_t,\mathbf{a}_t,\mathbf{m}).


The implementation uses residual next-state prediction, so the network learns a correction to \mathbf{y}*t. During rollout, \hat{\mathbf{y}}*{t+1} is recursively supplied as the next input. Clearly distinguish this inference-time autoregression from pushforward training.

For Modulated LOGLO-FNO, state and static fields form the spatial input while action and static information also drive block-wise modulation. Concatenation baselines receive the same available information through their declared adapters. Document these paths explicitly so the comparison is reproducible.

## 4.2 Explain why the benchmark is special

The benchmark should motivate the method without being overstated as a novel dataset contribution unless the data and split are themselves being released. Emphasize:

- coupled pressure-temperature dynamics;
- matrix/fracture state variables with different dynamics;
- heterogeneous porosity and permeability;
- localized high-gradient behavior;
- control-dependent trajectories;
- sparse but important well-driven dynamics;
- autoregressive error accumulation.

A good phrasing:

> The evaluation setting tests whether a surrogate reproduces both global reservoir evolution and localized matrix–fracture dynamics under heterogeneous geology and time-varying controls. Because one-step accuracy can conceal compounding deployment error, models are evaluated using channel-wise field metrics, structural and mass-balance diagnostics, and full autoregressive rollouts.



## 4.3 Describe train/test splits carefully

You should design splits that test generalization, not just interpolation.

Possible splits:


| Split                               | Purpose                                                  |
| ----------------------------------- | -------------------------------------------------------- |
| Random trajectory split             | Basic generalization across simulations                  |
| Unseen control schedule split       | Tests response to new injection/production schedules     |
| Unseen geological realization split | Tests generalization across permeability/porosity fields |
| Long-rollout split                  | Tests autoregressive stability                           |
| Stress-test subset                  | Tests high-gradient or near-well cases                   |


Even if you do not use all of these, include at least one split that is harder than random sampling. A benchmark is more convincing when it includes a deliberately challenging generalization setting.

---



# 5. Method section

The method section should have three subsections:

1. Task formulation
2. Architecture
3. Rollout-aware composite objective

This order is clean because the objective only makes sense after the task and output are defined.

---



## 5.1 Architecture section

Be explicit about what is inherited from LOGLO-FNO and what this work changes.

### 5.1.1 Local-global backbone

Each LOGLO block combines three representations:

- a global spectral branch for long-range interactions;
- a patchwise local spectral branch that retains local modes;
- a high-frequency pointwise branch derived from the spatial high-frequency component.

The branches are combined inside each block. The model predicts a residual correction to the current four-channel state.

### 5.1.2 Block-wise modulation

The proposed adaptation encodes the action and static reservoir fields into per-block scale and shift parameters. These parameters modulate normalized global and local features before their transformations. The intended role is not merely to add more input channels, but to let controls and geology alter feature processing throughout the operator.

The principal architecture control is **Modulated LOGLO-FNO versus vanilla LOGLO-FNO**:

- both retain the same local-global-high-frequency backbone;
- vanilla LOGLO-FNO concatenates action/static information into the spatial input;
- Modulated LOGLO-FNO uses a separate modulation encoder and block-wise conditional normalization.

This comparison supports a modulation claim more directly than comparison with FNO or U-Net. Report parameter counts and, if they differ materially, include a capacity-matched control or discuss the difference.

## 5.2 Rollout-aware composite objective

Define a composite discrepancy


\mathcal{D}(\hat{\mathbf y},\mathbf y)
= \lambda_{\mathrm{MSE}}\mathcal{L}_{\mathrm{MSE}}

- \lambda_{\mathrm{H1}}\mathcal{L}_{\mathrm{H1}}
- \lambda_{\mathrm{MBE}}\mathcal{L}_{\mathrm{MBE}}
- \lambda_{\mathrm{spec}}\mathcal{L}_{\mathrm{spec}}.


Explain each implemented term:

- **Weighted MSE:** normalized pointwise state accuracy across the four output channels.
- **H1 loss:** state and spatial-derivative agreement, providing direct supervision for field structure.
- **MBE:** a mass-balance discrepancy computed in physical units from pressure/temperature changes, porosity, and control rate.
- **Radial spectral loss:** agreement of spatial frequency content in the middle and high radial bands; retain the low-band diagnostic for evaluation even though it is not included in the current optimized spectral term.

Do not claim that the objective directly emphasizes near-well regions or temporal smoothness unless an implemented term does so.

The full training loss is


\mathcal{L}*{\mathrm{train}}
= \mathcal{D}(\hat{\mathbf y}*{t+1},\mathbf y_{t+1})

- \mathcal{D}(\hat{\mathbf y}^{\mathrm{pf}}*{t+1},\mathbf y*{t+1}).


For the pushforward term, generate a state by rolling the current model forward without gradient through a sampled history of up to k steps, detach that state, and supervise the next prediction. The curriculum increases the maximum k during training. This exposes the model to states produced by its own dynamics while keeping backpropagation local to the final step.

## 5.3 Claim language for the objective

Use:

> The objective decomposes surrogate error into complementary pointwise, derivative, mass-balance, and spectral discrepancies, and evaluates those discrepancies both on simulator states and model-generated pushforward states.

Avoid:

> The composite loss is responsible for the entire improvement.

The MSE-only baseline also disables pushforward. The full-versus-MSE comparison therefore tests the complete objective package. Attribute effects to H1, MBE, spectral supervision, or pushforward only through the corresponding leave-one-out ablations.

---

s

# 6. Experimental plan

The experiments should be designed to answer specific claims.

## 6.1 Main experimental questions

Organize the experiments around these questions, in this order:

1. **Does the full method outperform neural-operator and convolutional baselines under the rollout-aware composite-objective protocol?**
2. **Does block-wise modulation outperform concatenation within the LOGLO backbone?**
3. **How much performance is lost when Modulated LOGLO-FNO is trained with one-step MSE only?**
4. **How often does each model train successfully under each objective across seeds?**
5. **Which gains come from pushforward, H1, MBE, and spectral supervision?**
6. **Where do the gains appear: pointwise accuracy, spatial structure, mass balance, frequency content, or rollout stability?**

This turns your experiments into an argument.

---



# 7. Core experiment matrix

The central result table should contain every architecture under both training objectives:


| Model                | One-step MSE                         | Rollout-aware composite objective |
| -------------------- | ------------------------------------ | --------------------------------- |
| FNO configuration(s) | report                               | report                            |
| U-FNO                | report                               | report                            |
| UNO                  | report                               | report                            |
| U-Net                | convergence status + metric if valid | report                            |
| Vanilla LOGLO-FNO    | convergence status + metric if valid | report                            |
| Modulated LOGLO-FNO  | report                               | **full method**                   |


For every cell, aggregate only valid runs and separately report successful runs / attempted seeds. Never convert a failed run into an arbitrary large metric or silently omit it.

Interpret comparisons carefully:

- **Modulated LOGLO-FNO + composite versus every composite-trained baseline:** headline full-method result and architecture comparison under the same objective family.
- **Modulated versus vanilla LOGLO-FNO + composite:** primary evidence for block-wise modulation.
- **Modulated LOGLO-FNO + composite versus its one-step MSE run:** benefit of the complete training objective for the proposed architecture.
- **Within-model composite versus MSE for other converged models:** evidence that the objective is reusable beyond the proposed architecture.
- **U-Net/vanilla success rate across objectives:** optimization-robustness evidence, not an accuracy improvement, when the MSE run fails.

The current MSE configurations use model-specific learning rates, including a different learning rate for vanilla and Modulated LOGLO-FNO. Before attributing their different MSE behavior to modulation, run a small predefined learning-rate sweep or a matched-learning-rate control. Report the tuning policy for every model.

---



# 8. Metrics to report

Use one primary ranking metric and a small set of diagnostic metrics. Declare the primary metric before the remaining seeds are evaluated so the headline ranking is not selected post hoc.

## 8.1 Standard field metrics

Report:

- aggregate normalized relative L^2 error as the primary field metric;
- RMSE or MAE in normalized and, where interpretable, physical units;
- separate errors for T_m, T_f, P_m, and P_f;
- mean and standard deviation, or median and interquartile range, across seeds.



## 8.2 Spatial structure metrics

Report diagnostics aligned with implemented terms:

- H1 or spatial-gradient error;
- radial spectral error in low, middle, and high bands;
- error in high-gradient regions;
- near-well error only if the well mask and aggregation are defined independently of the method.



## 8.3 Temporal metrics

For autoregressive prediction:

- channel-wise and aggregate rollout error versus time;
- final-time error;
- area under the rollout-error curve;
- error growth rate.

A rollout curve with seed uncertainty is more informative than a single final-time number.

## 8.4 Physics metric

Report the same unweighted mass-balance discrepancy for every method. Keep it an evaluation metric independent of its weighted contribution to training. Do not call the model physically valid solely because MBE is lower.

Add operational metrics such as well-block error, breakthrough timing, or cumulative heat extraction only if they are computed consistently from available outputs and are central to the application.

## 8.5 Optimization robustness

Predefine a successful-run criterion before completing the seed study. It should include finite training, completion of the budget, and a fixed validation requirement. Report:

- successful runs / attempted seeds;
- best validation score and epoch;
- non-finite-loss or divergence events;
- training and validation curves for unsuccessful configurations.

Do not average failed and successful runs into one accuracy number. A failed baseline is evidence about the tested training configuration, not proof that the architecture can never learn the task.

---



# 9. Ablation studies

Use the ablations to separate the two central mechanisms: modulation and the rollout-aware composite objective.

---



## 9.1 Objective decomposition

The current full-versus-MSE comparison simultaneously changes composition and pushforward. Add a minimal 2\times2 decomposition:


| One-step discrepancy      | Pushforward disabled            | Pushforward enabled |
| ------------------------- | ------------------------------- | ------------------- |
| MSE                       | current MSE-only baseline       | add this experiment |
| MSE + H1 + MBE + spectral | current no-pushforward ablation | full objective      |


This directly answers whether composite discrepancies and pushforward provide separate or interacting gains.

Then use the existing leave-one-out experiments, all with pushforward retained:

- full minus H1;
- full minus MBE;
- full minus spectral.

MSE remains in every composite variant because it anchors pointwise accuracy. Report each ablation using the same primary, rollout, H1, MBE, and spectral metrics. A term is useful when it improves deployment-relevant evaluation, not merely when its own training metric decreases.

## 9.2 Loss-weight sensitivity

Because composite weights can look arbitrary, include a compact sensitivity study after the main ablation.

Test:

- default weights;
- half and double each non-MSE weight one at a time;
- optionally one equal-weight setting.

The goal is to show that the conclusion is not confined to one precise weight vector, not to retune every baseline exhaustively.

## 9.3 Architecture ablation

The required ablation is Modulated LOGLO-FNO versus vanilla concatenation-based LOGLO-FNO under the full objective. Run both across the same seeds and document:

- data split, budget, scheduler, and stopping rule;
- model-specific learning rates and how they were selected;
- parameter counts and training cost;
- accuracy, rollout behavior, and success rate.

For the MSE-only comparison, include a matched-learning-rate control or small common sweep before claiming that modulation causes the observed convergence difference.

Only if resources remain, ablate the local, global, or high-frequency branch. These tests explain the LOGLO backbone but are secondary to the paper’s modulation claim.

## 9.4 Generalization tests

If the data support them, compare seen and unseen geological realizations or operating schedules. Keep this secondary to the complete architecture–objective matrix. One-step versus rollout is an evaluation view, not a benchmark ablation.

---



# 10. Figures and tables to include



## Figure 1: Task and benchmark schematic

Show:

- four-channel matrix/fracture state;
- action and heterogeneous static fields;
- one-step transition;
- autoregressive deployment;
- pushforward training exposure.



## Figure 2: Architecture diagram

Show the global spectral, patchwise local spectral, and high-frequency branches in one LOGLO block. Draw the modulation encoder separately and show its per-block scale/shift outputs. A small inset should contrast modulation with vanilla input concatenation.

## Figure 3: Main result

Use two aligned panels:

- model ranking under the rollout-aware composite objective;
- rollout error versus time with seed uncertainty.

Visually mark Modulated LOGLO-FNO as the full method without suppressing uncertainty or failed-run counts.

## Table 1: Dataset/benchmark details

Include:

- number of simulations;
- state variables;
- grid size;
- number of time steps;
- train/validation/test split;
- control variables;
- static variables;
- prediction target;
- evaluation settings.



## Table 2: Main model comparison

Report the complete model-by-objective matrix, parameter counts, central tendency across seeds, and successful/attempted runs. Use “did not train successfully” for a failed configuration rather than a misleading metric.

## Figure 4: Objective decomposition

Show the 2\times2 MSE/composite × one-step/pushforward comparison for Modulated LOGLO-FNO, followed by H1/MBE/spectral leave-one-out results.

## Figure 5: Optimization robustness

If the U-Net and vanilla LOGLO-FNO failures persist across seeds, show training success rate and representative validation curves. Keep this separate from the accuracy ranking so convergence and predictive quality are not conflated.

## Figure 6: Qualitative field predictions

Show ground truth, prediction, and error slices for T_m,T_f,P_m,P_f, including the full method, its MSE-only counterpart, and the strongest baseline. Select cases by a prespecified rule, such as median and high-error test trajectories, rather than choosing only favorable examples.

## Table 3: Training and ablation details

List objective weights, pushforward curriculum, learning-rate selection, training budget, and the ablation definitions. This prevents the central comparison from depending on ambiguous labels such as “MSE” and “full loss.”

---



# 11. How to present the task, architecture, and objective without losing focus

Use one unifying theme:

**Architecture–objective alignment for robust autoregressive geothermal surrogates.**

Every section should serve this theme:

- the task requires cross-scale dynamics, conditioning, and rollout stability;
- LOGLO supplies local-global representations;
- modulation injects controls and geology throughout the operator;
- the composite pushforward objective supplies task-aligned supervision;
- controlled comparisons show why the combination is stronger than either element alone.

Recommended working title:

> Modulated Local–Global Neural Operators with Rollout-Aware Composite Training for Geothermal Reservoir Dynamics

Shorter alternative:

> Robust Geothermal Surrogates through Modulated Operators and Rollout-Aware Training

Do not foreground a “new benchmark” in the title unless the dataset/split is itself a documented contribution.

---



# 12. Discussion section plan

The discussion should interpret the evidence in the same order as the results.

## 12.1 Why the full method is strongest

Argue that the local-global backbone provides the required representation, modulation makes that representation responsive to controls and geology, and the rollout-aware objective directs it toward deployment-relevant behavior. The full method’s ranking is the headline evidence for their complementarity.

## 12.2 What the MSE experiment adds

The weaker result of MSE-trained Modulated LOGLO-FNO shows that architecture alone does not explain the full performance. Its successful initial training, contrasted with vanilla LOGLO-FNO, suggests that modulation may also improve optimization robustness. However, describe the latter as preliminary until the seed study and learning-rate control are complete.

If U-Net and vanilla LOGLO-FNO repeatedly fail under one-step MSE but train under the full objective, interpret this as an **objective-dependent training-stability result**. It is not a paired accuracy improvement because no valid MSE endpoint exists.

## 12.3 How broadly the objective helps

For architectures that converge under both settings, compare paired seed aggregates and discuss predictive gains. For architectures that do not, compare success rates. This produces a nuanced generality claim:

> The rollout-aware composite objective improves predictive quality for converged models and can improve training robustness for more fragile configurations.

Use that sentence only if the completed experiments support both clauses.

## 12.4 Where the method helps most

Discuss whether gains are largest in:

- matrix or fracture variables;
- pressure or temperature;
- spatial gradients or spectral bands;
- mass-balance behavior;
- late rollout times;
- difficult geological realizations.

This gives the results scientific meaning.

## 12.5 Limitations

Be direct. Possible limitations:

- The benchmark is still simulator-generated.
- Composite-objective weights require selection.
- MBE is a soft training/evaluation constraint and does not guarantee conservation.
- The method is tested on the available reservoir settings and control distributions.
- Long-horizon rollout still accumulates error.
- The LOGLO backbone is adapted from prior work; novelty lies in modulation, geothermal application, and architecture–objective evaluation.
- Any remaining failed MSE runs may depend on optimizer hyperparameters as well as architecture, so conclusions are bounded by the documented tuning protocol.

---



# 13. Conclusion plan

The conclusion should be concise.

It should say:

1. The task requires accurate cross-scale, control-conditioned, autoregressive dynamics.
2. Modulated LOGLO-FNO introduces block-wise conditioning of a local-global operator.
3. The rollout-aware composite objective targets pointwise, H1, mass-balance, spectral, and pushforward behavior.
4. Their combination achieves the strongest performance among the evaluated baselines.
5. Objective and architecture controls show complementary roles, stated at the strength justified by the final multi-seed evidence.

Do not overclaim physical validity unless you directly validate it.

A good final sentence could be:

> These results show that robust geothermal surrogate learning depends on aligning cross-scale operator structure and control conditioning with supervision that reflects both field fidelity and autoregressive deployment.

---



# 14. Recommended experiment priority

If you have limited time, prioritize in this order:

1. **Complete the planned seeds** for the full-objective and MSE matrix.
2. **Predefine and report training success**, then diagnose U-Net and vanilla LOGLO-FNO with matched/swept learning-rate controls.
3. **Complete the full model-by-objective table** with uncertainty and successful/attempted counts.
4. **Evaluate rollout error over time** for all valid runs.
5. **Add MSE + pushforward** to complete the 2\times2 objective decomposition.
6. **Complete no-H1, no-MBE, no-spectral, and no-pushforward ablations**.
7. **Compare Modulated and vanilla LOGLO-FNO** with parameter counts and matched seeds.
8. **Produce prespecified qualitative examples**.
9. **Run compact weight sensitivity and harder generalization tests** if resources remain.

The minimum convincing paper needs items 1–7. Items 8–9 deepen interpretation but do not replace replication or objective decomposition.

---



# 15. The cleanest contribution framing

Frame the paper as:

> We develop a control-modulated local-global neural operator and show that its strongest performance emerges when it is paired with a rollout-aware objective that supervises complementary error modes.

The strongest current empirical statement is:

> Modulated LOGLO-FNO trained with the rollout-aware composite objective outperforms all evaluated baselines in the initial experiments. Modulated LOGLO-FNO remains trainable but is less accurate under one-step MSE, while vanilla LOGLO-FNO and U-Net do not train successfully in the first MSE-only seed.

The intended final claim, contingent on replication, is:

> Across seeds, Modulated LOGLO-FNO with rollout-aware composite training provides the best predictive accuracy and rollout behavior. Controlled architecture and objective ablations show that block-wise modulation and task-aligned pushforward supervision make complementary contributions to performance and training robustness.

This is a single story: **the architecture makes the controlled dynamics representable and robust, the objective makes the learned dynamics accurate under deployment, and their combination produces the best surrogate.**