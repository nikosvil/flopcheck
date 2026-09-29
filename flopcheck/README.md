# flopcheck

A reference implementation of **Annex A.2** of the European Commission's
*Guidelines on the scope of the obligations for general-purpose AI models*
(18 July 2025), which govern how providers estimate the training compute
that determines classification under **Article 51(2)** of the EU AI Act.

The Annex permits any estimation method "so long as the estimated amount
is, in the providers' best judgement, accurate within an overall error
margin of 30% of the reported estimate", and describes two:

| | Approach | Formula |
|---|---|---|
| A.2.1 | Hardware-based | `C = N × L × H × U` |
| A.2.2 | Architecture-based | `C = 6 × P × D` for dense transformers |

`flopcheck` implements both, reproduces the Commission's own eight worked
examples, and quantifies what the Annex leaves undefined.

## Status

Research code accompanying the preprint *What does "GPU utilization" mean?
Testing the 30% accuracy margin for training-compute estimates under the EU
AI Act* (https://doi.org/10.5281/zenodo.22881088). Peak-performance
entries marked `unverified` **must** be checked against vendor datasheets
before being cited — `peaks.unverified()` lists them.

## Install and run

```bash
pip install -e ".[dev]"
pytest
python -m flopcheck.report
```

## What it shows

**All eight Annex A.3 figures reproduce.** Six follow from published facts
about the models. Models B and H do not: recovering them requires supplying
a peak performance and a GPU utilisation that the Annex asserts without
derivation or citation — `U = 0.5`, stated once for Model B and then reused
for Model H with the words *"GPU utilisation as previously"*.

**"Peak theoretical performance" is precision-ambiguous.** For an H100 SXM
the candidate values span 494.7 → 3957.8 TFLOP/s: a factor of **8**. The
Annex names neither a precision nor a sparsity basis, while requiring in
the same section that "all operations should be counted equally,
independently of floating-point precision". The numerator of the
utilisation term is therefore precision-independent and its denominator is
not, and the two are never reconciled.

**"GPU utilisation" is undefined**, and at least three incompatible
quantities go by that name — `nvidia-smi` time-occupancy (0.90–0.99),
DCGM tensor-pipe activity (0.30–0.70), and model FLOP utilisation
(0.20–0.50). For a disclosed run of 6,000 H100s over 30 days, every
occupancy reading places the model **above** the Article 51(2) threshold
and every MFU reading places it **below**.

**The disclosure format alone consumes the error budget.** The Model
Documentation Form requires hardware days "recorded with at least one
significant figure" — a ±12.5% quantisation, 42% of the entire permitted
margin, before any estimation assumption is made. The compute figure
disclosed to national competent authorities is required only "up to its
order of magnitude", which is less precise than the margin the provider is
bound to.

## Layout

```
flopcheck/
  peaks.py                Accelerator peaks by precision, dense and sparse,
                          each with a provenance tag.
  estimate.py             Both Annex A.2 approaches, reconciliation, and the
                          margin analyses.
  commission_examples.py  Annex A.3, encoded with its assumptions separated
                          from its published facts.
  report.py               Generates the tables and figure data.
tests/                    Reproduces every published Annex A.3 figure.
```

## The regulatory hook

Each methodology field in the Model Documentation Form opens: *"In the
absence of a delegated act adopted in accordance with Article 53(5) AI Act
to detail measurement and calculation methodologies…"*

Article 53(5) empowers the Commission to adopt delegated acts detailing
measurement and calculation methodologies "with a view to allowing for
comparable and verifiable documentation", covering Annex XI points 2(d)
and (e) — computational resources and energy consumption.

That power exists and has not been exercised. This work is intended as
technical input to it.

## Sources

- Guidelines on the scope of the obligations for general-purpose AI models,
  C(2025) 5045, 18 July 2025 — Annex, paragraphs 114–141.
- GPAI Code of Practice, Transparency chapter — Model Documentation Form.
- Regulation (EU) 2024/1689 (AI Act), Articles 51 and 53, Annex XI.

## Licence

MIT, see `LICENSE` in the repository root.

*The views expressed are the author's own and do not represent those of any
employing or contracting institution.*
