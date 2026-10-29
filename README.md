# Cross-domain resource sharing: supplementary code

Supplementary code for the review article

> **Optimizing Computing Resource Sharing across Administrative Boundaries in the AI and Edge Era:
> Taxonomy, Challenges, and Research Directions** (submitted to *Mathematics*, MDPI).

The repository contains two independent parts:

| Script | Reproduces | Runtime |
| --- | --- | --- |
| `experiments_58.py` (uses `sharing_sim.py`) | Section 5.8, Table 9 | about 10 s |
| `openalex_queries.py` | Section 2.4, Table A1 | a few minutes (network access to the OpenAlex API) |

## Requirements

Python 3.9 or later. The simulation needs NumPy; the literature query uses only the standard library.

    pip install -r requirements.txt

The results in `results/` were produced with Python 3.13 and NumPy 2.5.3.

## Section 5.8: illustrative comparison of execution strategies

`sharing_sim.py` is a discrete-event simulation of deadline-constrained external GPU tasks running on
GPUs that their providers can reclaim. `experiments_58.py` runs every strategy under light and
contended load with 30 seeds each and prints the rows of Table 9 (mean and 95% confidence half-width).

    python experiments_58.py

It writes `results_58.json` to the current directory. The committed outputs are in
`results/table9.txt` (printed table) and `results/results_58.json` (all metrics).

Strategy codes:

| Code | Strategy |
| --- | --- |
| P | Placement only: one copy, restart from scratch after a reclaim |
| R | Replication with the covering constraint (14) and the failure-domain limit (15) |
| C | Recovery with (16)–(19), checkpoints on the provider's local disk |
| Cr | Recovery with (16)–(19), checkpoints also stored outside the provider |
| C+ | Recovery with the deadline-aware conditions (19a)–(19b) |

Default setting (`Params` in `sharing_sim.py`): 24 GPUs in 6 provider domains, Poisson task arrivals
(5 per hour for light load, 15 per hour for contended load), tasks of 20–60 min with 5–40 GB of state,
1 Gbit/s between providers, provider reclaims at 1 per GPU-hour with 2 min of notice, a quarter of
the reclaim intensity domain-wide. All random draws use NumPy generators seeded per run, so the output
is deterministic for a given NumPy version.

The example illustrates the formulation of Section 5; it is not a ranking of strategies or a model of
a particular deployment. Its assumptions are listed in Section 5.8 of the article.

## Section 2.4: documented literature search

`openalex_queries.py` reruns the fifteen queries of Table A1 against the OpenAlex API (titles and
abstracts, publication years 2016–2026, types article and preprint) and writes `records.csv` with one
row per distinct record and the queries that retrieved it.

    python openalex_queries.py

The search reported in the article was run on 28 October 2026 and returned 1105 distinct records.
OpenAlex is updated continuously, so later runs can return slightly different counts.

## Citation

If you use this code, please cite the article above.

## License

MIT License; see `LICENSE`.
