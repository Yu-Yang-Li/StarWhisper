# Problem-to-Data Routing

Use this file after reading the user question. First identify input, output, and sample unit. Then pick the closest task family before suggesting models.

Assume the user understands the science. The routing job is to preserve the scientific wording while assigning data roles.

## Generalized Workflow

Use the same shape across disciplines:

```text
research goal + dataset + field description
-> task recommendation
-> framework selection
-> executable plan
-> visualization/statistics/model result
-> interpretation report
-> goal check
```

The framework can be statistical analysis, classical machine learning, deep learning, time-series modeling, causal inference, retrieval/text analysis, or visualization-only analysis. Choose the lightest one that can answer the goal.

| Scientific wording | Input | Output | Data question | First baseline | Main metrics |
|---|---|---|---|---|---|
| Does X affect Y? | exposure/intervention, outcome, confounders, time/order | effect estimate or group difference | causal effect or group comparison | regression adjustment, matching, DID, simple test | effect size, CI, p/uncertainty, sensitivity |
| Can we predict Y? | features known before prediction | label/value/risk score | supervised prediction | mean/majority, linear/logistic | MAE/RMSE/R2, AUC/F1, calibration |
| Which cases are high risk? | history/features before decision time | risk ranking or class | risk ranking/classification | logistic/Cox/simple score | AUC, PR-AUC, C-index, calibration |
| What patterns exist? | unlabeled features plus batch/source | clusters, embedding, subgroup map | clustering/exploration | PCA/UMAP + k-means/hierarchical | stability, silhouette, interpretability |
| What changes over time? | entity id, time, outcome history, covariates | forecast or trend | time-series or longitudinal modeling | last value, moving average, ARIMA/mixed model | MAE/RMSE, temporal CV |
| Which item should be retrieved? | query and candidate set | ranked list or relevant item | retrieval/ranking | BM25/TF-IDF | recall@k, nDCG, MRR |
| Is this observation abnormal? | reference-normal data plus candidate sample | anomaly score or flag | anomaly detection | z-score/IQR/isolation forest | precision@k, review yield |
| What mechanism explains it? | variables along process chain, time/order, perturbation | mechanism map or tested path | mechanism modeling plus evidence mapping | descriptive + mediation/path model if justified | effect decomposition, fit, falsification |

## Translation Checklist

Ask or infer:

- What is one sample?
- What is the input?
- What is the expected output?
- What field descriptions or metadata are needed?
- What is the outcome?
- What features are known before the outcome?
- Is the question predictive, causal, descriptive, mechanistic, or exploratory?
- What split prevents leakage?
- What would count as a useful baseline result?
- Which framework is sufficient for a first answer?
- What would make the result scientifically untrustworthy?

## Common Corrections

- "X affects Y" is not automatically prediction.
- "Find biomarkers" is usually feature discovery plus validation, not just classification.
- "Mechanism" needs time/order/perturbation evidence, not only feature importance.
- "Good model" means useful under the scientific split, not only high random-split score.
