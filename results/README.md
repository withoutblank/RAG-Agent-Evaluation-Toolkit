# Evaluation results

Benchmark runs write these working artifacts to this directory:

- `<configuration>.json` — configuration, provenance, aggregate metrics, and
  complete per-question outcomes;
- `summary.csv` — one comparison row per configuration;
- `report.md` — generated provenance, summary, interpretation, failure analysis,
  per-question tables, latency caveats, and warnings.

Those working files are ignored by Git. Live-provider output, timings, run IDs,
and model behavior can vary, and unreviewed outputs may contain details that do
not belong in a public repository.

The implementation commit carries a truthful placeholder at:

- [`sample_benchmark_report.md`](sample_benchmark_report.md)

After that commit is clean, the benchmark is run and a follow-up commit replaces
the placeholder with an exact byte-for-byte copy of the generated `report.md`.

## Reproduce

After installing the project:

```powershell
rag-eval benchmark --config configs/benchmark.yaml --provider fake
```

This recreates the ignored JSON, CSV, and `report.md` working files. To rebuild
the Markdown report from existing result JSON:

```powershell
rag-eval report --results-dir results
```

Do not edit benchmark numbers by hand. After changes to corpus content, labels,
normalisation, chunking, providers, prompts, retrieval, citations, or metrics:

1. rerun the benchmark;
2. confirm the report records a `clean` Git worktree and the implementation
   commit that produced it;
3. review failures and public-safety concerns;
4. copy the generated report exactly to `sample_benchmark_report.md`;
5. update any README values from the same generated run.

Do not promote a dirty or unverified run as the curated release sample.

OpenAI-backed runs require the user's own key, can cost money, and are not
committed by default. See the
[benchmark methodology](../docs/benchmark-methodology.md) for formulas,
eligibility rules, and reproducibility limits.
