# zebra-finch-semantic-test

A pre-registered re-test of the "semantic magnet" comparison in Elie et al. 2025, *Science*
389:1210 ("Categorical and semantic perception of the meaning of call types in zebra
finches", doi:10.1126/science.ads8482), run on the authors' own public code and data.

**Status: registered 2026-11-25, waiting for compute. No result yet.**

- [PREREG.md](PREREG.md): the claim, the one-letter variable typo in the authors' Figure 4
  bootstrap (the behavioural matrix never receives the shift the acoustic matrix gets),
  three arms of the authors' notebook (as is, typo fixed, neither shifted), two scale-free
  tests (an exact partial test over all 69,300 groupings of the 11 call types, and a
  within/across AUC), calibration worlds, 15 forecasts with priors, and the headline rules.
- `zf_semantic.py`: `fetch` downloads and md5-checks Zenodo record 15795568 (v3.0, commit
  4a3c6ef); `run` executes the arms and tests and writes results.json, grade.json,
  load_report.json and RESULTS-ZF.md.
- `test_zf_semantic.py`: unit tests (numpy only) on synthetic data with known answers.

```
pip install numpy==1.26.2 scipy==1.11.4 pandas==2.1.4 scikit-learn==1.2.2 \
    matplotlib==3.8.0 seaborn==0.12.2 statsmodels==0.14.1 h5py==3.10.0
python -m unittest -v test_zf_semantic
python zf_semantic.py fetch --data-dir DATA --out-dir OUT
python zf_semantic.py run --data-dir DATA --out-dir OUT
```

The study does not test the paper's other evidence (call-type discrimination, reaction
times, or the generalisation experiment of Figure 3).

The data and the original code are the authors' (CC BY 4.0) and are not redistributed
here. This repository's code is Apache-2.0; text CC BY 4.0. MV2, Munim, Inc.,
mv2@munimtech.com. Made with AI agents working in MV2's name; no human expert has checked
it yet.
