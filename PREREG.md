# PREREG: does the zebra finch "semantic magnet" survive a one-line fix and a scale-free test? (registered 2026-11-25)

MV2, Munim, Inc. Made with AI agents working in MV2's name; no human expert has reviewed
this plan yet. Text CC BY 4.0, code Apache-2.0. All data and the original code are the
authors' (CC BY 4.0).

Registered 2026-11-25 (about 02:30 UTC), **before any data file of the study was opened**.
Before writing this we read only: the paper's press coverage, the Zenodo record metadata,
the GitHub file list, and the source code (with its printed outputs, i.e. the numbers the
authors themselves report) of `pythonNotebooks/BEHAV-Figure 4.ipynb`,
`BEHAV-Figure 3_logistic.ipynb`, `RNotebooks/countStatsConfusionRIT.Rmd`,
`code/peck_data.py`, `requirements.txt` and `README.md`. No behavioural CSV, `.mat` or
`.h5` file was opened, by us or by anyone working for us.

## The claim

Elie, de Witasse-Thézy, Thomas, Malit and Theunissen (2025), "Categorical and semantic
perception of the meaning of call types in zebra finches", *Science* 389:1210-1215,
doi:10.1126/science.ads8482 (the work behind the 2026 Coller Dolittle Prize). Twelve zebra
finches learned, call type by call type, to wait for one rewarded call type among 11. The
paper reports that the birds "confused Etho call types that have a similar meaning at a
higher rate than would be expected given the acoustic similarity of these call types" (a
"semantic magnet effect").

In the authors' Figure 4 code that sentence rests on one comparison. A behavioural
distance matrix (log2 odds ratios of interrupting) and an acoustic distance matrix (log2
odds ratios of a leave-one-bird-out LDA on spectrograms) are each embedded in 2-D by
metric MDS. For six semantic groups (Contact: LT, DC, Te; Alarm: Tu, Th; Fight: Di, Ag;
Bond: Wh, Ne; Display: So; Hunger: Be) a ratio Q = (mean squared distance of outside calls
to a group centroid) / (mean squared distance of member calls) is computed. Over 1,000
bootstrap resamples of birds (seed 123), the notebook prints:

| | behaviour Q | acoustic Q | behaviour > acoustic |
|---|---|---|---|
| with song | 26.37 | 10.89 | 99.20% (992/1000) |
| without song | 21.27 | 9.68 | 97.90% (979/1000) |

## What we noticed in the code (a fact, checked verbatim)

Inside the bootstrap loop (and in the earlier full-data cell) the behavioural matrix is
meant to be shifted by its smallest diagonal value, as the acoustic one is:

```
    confMatDist = -confMatTot
    minval = np.min(np.diag(confMatDist))
    confMatdist = confMatDist - minval        # <- lower-case 'd': a new variable
    ...
    minval = np.min(np.diag(confMatAcoustLOR))
    confMatDistAcoust = confMatAcoustLOR - minval
    ...
    acMDSBoot = mds.fit_transform(confMatDistAcoust)
    behMDSBoot = mds.fit_transform(confMatDist)  # the unshifted matrix
```

So the acoustic distances get a constant added to every off-diagonal entry and the
behavioural distances do not. Metric MDS is not invariant to adding a constant to all
distances: a constant pushes points towards an evenly spread configuration and lowers a
within/across ratio like Q. The comparison of a Q computed on one scale with a Q computed
on another therefore leans towards the unshifted matrix. **Whether the lean matters for
the published numbers is the empirical question this study answers.** A second, deeper
point: even with matched shifts, Q is not invariant to monotone transforms of the
distances, so "behavioural Q > acoustic Q" can depend on the scale each matrix happens to
be on. We therefore also register two tests that do not depend on scale at all.

## Data and code

Zenodo record 15795568 (version 3.0, "Resubmission July 2025", = GitHub tag 3.0, commit
4a3c6ef), file `theunissenlab/zebra-finch-semantic-category-3.0.zip`, 99,341,119 bytes,
md5 eb041f1c9928a7feadb206ef955168e6. Python 3.11 with the authors' pins (numpy 1.26.2,
scipy 1.11.4, pandas 2.1.4, scikit-learn 1.2.2, matplotlib 3.8.0, seaborn 0.12.2).

## Arms (the authors' notebook, executed cell by cell, unchanged except as listed)

Every arm also appends `_BOOT_REC.append(...)` on the `behMDSBoot = ...` line to record
both distance matrices of every bootstrap draw (a read-only side effect). Figures are not
saved. Optional imports that are missing are replaced by inert stubs and listed.

- **asis**: no other change. Must reproduce the table above.
- **fixed**: `confMatdist = confMatDist - minval` -> `confMatDist = confMatDist - minval`
  (every occurrence; count reported). Same seed, so the bootstrap draws are paired with asis.
- **noshift**: the acoustic shift is removed instead (`- 0.0*minval`), so neither matrix
  is shifted in the bootstrap.

## Scale-free tests (our code, `zf_semantic.py`)

On the full-data matrices from the asis run (first value of `confMatTot` and
`confMatCountAcoustic`): behavioural distance = symmetrised `-confMatTot`; acoustic
distance = symmetrised log2 odds ratio of the summed LDA counts (zeros -> 0.5, the authors'
formula), reordered to the behavioural call order. Only the 55 call pairs (45 without song)
enter, as ranks, so any shift or monotone transform of either matrix leaves them unchanged.

- **E (primary, partial semantic test).** Regress rank(behavioural distance) on
  rank(acoustic distance) and a same-meaning indicator; T = minus the indicator's
  coefficient (positive = same-meaning pairs are closer in behaviour than their acoustic
  distance predicts). Exact p over **all** 69,300 ways to split the 11 call types into
  unlabeled groups of sizes 3, 2, 2, 2, 1, 1 (12,600 for the 10 types without song, sizes
  3, 2, 2, 2, 1), counting the real grouping. This is the paper's sentence, tested
  directly.
- **D (scale-free version of Q).** AUC = probability that a same-meaning pair is closer
  than a cross-meaning pair, for behaviour and for acoustics; exact p for each over the
  same partitions; and across the authors' own 1,000 recorded bootstrap draws, the share in
  which behavioural AUC > acoustic AUC.
- **Calibration of E.** 200 worlds with no semantic term (rank-scale fit of behaviour on
  acoustics plus Gaussian noise of the observed residual size) give its false-positive rate;
  200 worlds with a semantic shift of one residual SD give its power.
- **Mechanism demonstration.** Our own Q (2-D metric MDS, same sklearn settings) on the
  acoustic distances after adding 1, 2 or 4 SDs to every distance, and after squaring,
  cubing and exponentiating; and on the behavioural distances unshifted vs shifted as
  intended.

## Headline (exactly one)

- `LOADER_STOP`: archive md5 wrong, or the asis arm does not produce bootstrap Q arrays,
  or E cannot be computed.
- `NOT_REPRODUCED`: asis mean Qs not within 10% of 26.37 and 10.89, or fewer than 950 of
  1,000 draws with behaviour > acoustic.
- Otherwise, with B = share of fixed-arm draws with behaviour Q > acoustic Q (with song)
  and E = exact partial p (with song):
  - `SURVIVES_BOTH`: B >= 0.95 and E <= 0.05.
  - `TYPO_DEPENDENT_SEMANTIC_HOLDS`: B < 0.95 and E <= 0.05.
  - `TYPO_DEPENDENT_NOT_SUPPORTED`: B < 0.95 and E > 0.05.
  - `Q_SURVIVES_PARTIAL_FAILS`: B >= 0.95 and E > 0.05.

## Forecasts (priors registered now; Brier score reported)

| id | forecast | prior |
|---|---|---|
| P1 | archive verifies and the asis arm produces the bootstrap Q arrays | 0.80 |
| P2 | asis reproduces (both Qs within 10%, >= 950/1000) | 0.75 |
| P3 | the typo line occurs exactly twice | 0.60 |
| P4 | fixed-arm mean behavioural Q is lower than asis | 0.85 |
| P5 | fixed arm: behaviour > acoustic in fewer than 95% of draws (with song) | 0.45 |
| P6 | fixed arm: mean behavioural Q below mean acoustic Q | 0.25 |
| P7 | noshift arm: behaviour > acoustic in fewer than 95% of draws | 0.40 |
| P8 | E with song: exact p <= 0.05 | 0.50 |
| P9 | E without song: exact p <= 0.05 | 0.45 |
| P10 | full-data AUC: behaviour > acoustic | 0.65 |
| P11 | behavioural AUC > acoustic AUC in >= 95% of the authors' bootstrap draws | 0.35 |
| P12 | E's false-positive rate in no-semantic worlds <= 0.07 | 0.85 |
| P13 | E's power at a one-SD semantic shift >= 0.80 | 0.60 |
| P14 | adding 2 SDs to the acoustic distances lowers our Q by at least 25% | 0.75 |
| P15 | the asis behavioural distance matrix has at least one negative entry | 0.50 |

## What this does not test

The paper's other evidence is untouched: that birds discriminate and categorise all call
types, the reaction-time results, and the second experiment showing that birds trained on
tets versus distance calls or thuks generalise by category to new, acoustically confusing
exemplars (Figure 3). A failure here would weaken only the semantic-magnet comparison, not
the categorical-perception result. With 11 call types and 6 groups the partition tests
have few units, which the calibration measures. The behavioural and acoustic matrices come
from different measurements (trials vs LDA counts), so no single test can make them fully
comparable; E asks the narrower question of whether meaning adds anything once acoustic
similarity is in the model.

## When results return

Commit `results/` (results.json, grade.json, load_report.json, RESULTS-ZF.md) and a
hand-checked READING-ZF.md. If `LOADER_STOP`: read load_report.json and the arm logs, fix in
a dated AMENDMENT-1.md without looking at any outcome statistic, and re-run.
