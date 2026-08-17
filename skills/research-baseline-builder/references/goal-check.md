# Goal Check

After analysis or modeling, check the result against the original scientific goal before writing a confident conclusion.

## Questions

- Does the output match the requested scientific output?
- Did the model use only information available at the intended decision time?
- Does the split match the scientific generalization target?
- Are the metrics aligned with the scientific cost of errors?
- Are negative, weak, or inconclusive results stated plainly?

## Common Mismatches

| Result | Does not prove |
|---|---|
| High prediction accuracy | causal effect or mechanism |
| Significant correlation | causation |
| Feature importance | biological/physical mechanism |
| Clean clusters | real scientific classes |
| Good random-split score | cross-site, cross-patient, cross-batch, or future performance |
| Nice visualization | quantitative support |
| Low error on held-out rows | robustness to new instruments, labs, materials, or cohorts |

## Final Sentence Pattern

Use a concise statement:

```text
This baseline answers [part of goal] under [data/split/metric]. It does not yet support [unsupported claim]. The next check is [missing data, stronger split, experiment, or model].
```
