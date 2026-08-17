# Framework Selection

Choose the lightest framework that can answer the translated data question. Do not treat this as model shopping.

| Data/task signal | Use first | Avoid at first |
|---|---|---|
| Table + clear label | sanity baseline, linear/logistic model, tree baseline | deep model before leakage/split checks |
| Table + "X affects Y" | statistical comparison, regression adjustment, matching/DID when justified | plain prediction framed as causal evidence |
| Small sample + many variables | visualization, simple model, regularization, uncertainty | high-capacity model with one random split |
| Image + label | data quality check, simple CNN or pretrained classifier | custom architecture before label audit |
| Time/order matters | time-aware split, naive/last-value baseline, ARIMA or simple sequence model | random split across time |
| Repeated patient/material/site/source | grouped split, mixed effects or grouped validation | row-level random split |
| Text/literature/query | keyword/BM25 or TF-IDF baseline, then embeddings/RAG if needed | RAG before relevance labels or retrieval metric |
| Mechanism question | pathway variables, perturbation/time evidence, mediation/path analysis if justified | feature importance as mechanism |
| Exploration only | descriptive statistics, PCA/UMAP, clustering, subgroup plots | supervised model without target |

## Minimal Recommendation Format

Report:

- chosen framework;
- why it is sufficient for the first answer;
- what it cannot prove;
- what stronger framework would require.
