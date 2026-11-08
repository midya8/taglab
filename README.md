# TagLab

Linear-chain conditional random fields from scratch: manual log-space inference,
expected-minus-observed likelihood gradients, complete optimization replay,
exact rational path controls and an editable scalar-JavaScript inspector.
Dependency-free Python 3.10+. Standard CRF theory, not a new method.

## The control is allowed to win

The validation-selected chain has lower test conditional NLL, but makes **more
Viterbi token errors** than the separately trained independent-label control.
Both models use the same eight observed features, including left/right context.

| Selected model | Validation NLL/token | Test NLL/token | Viterbi token errors | Marginal token errors | Whole-sequence errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chain CRF, λ 0.3 | 0.675763 | 0.557614 | 66/164 | 56/164 | 15/16 |
| Independent log-linear, λ 0.3 | 0.831299 | 0.813715 | 56/164 | 56/164 | 15/16 |

NLL is a distribution-scoring objective. Viterbi chooses one highest-probability
complete path, while marginal decisions minimize model expected token error.
Neither decoder is guaranteed to win on one observed sequence. These fixture
comparisons are descriptive, not causal isolation of transitions or a universal
ranking. All six candidates remain visible, including lower-penalty chains
that score differently on test but were not selected by it.

[Interactive sequence and gradient inspector](https://midya8.github.io/taglab/) ·
[Equations, sources and limits](METHOD.md)

## Reproduce it

```sh
python -m taglab demo
python -m unittest discover -s tests -v
node --test tests/*.cjs                 # Node 22 and Chrome for the browser test
bash examples/demo.sh
python -m pip wheel . --no-deps -w /tmp/taglab-wheels
python -m pip install --target /tmp/taglab-installed /tmp/taglab-wheels/*.whl
bash examples/verify_wheel.sh /tmp/taglab-installed "$PWD"
```

The six commands are `demo`, `train`, `audit`, `infer`, `export` and
`normalization`. Eight importable modules including the package root work outside
the checkout. No NumPy, PyTorch, autodiff, model API or external dataset is used.

```sh
python -m taglab train --output /tmp/taglab-fresh.json
python -m taglab audit --input /tmp/taglab-fresh.json
python -m taglab infer --symbols '[-2, 0, 2]' --family chain
python -m taglab normalization
python -m taglab export --output /tmp/taglab-web
```

`train` fits all six candidates from zero and freezes selection before reading
test scores. `audit` replays all 327 stored states, analytic gradients, accepted
Armijo movements, rejected larger steps, validation selection, test metrics and
selected-model complete message witnesses. The archive is losslessly gzip/base64
packed into three bounded parts and SHA-256 checked before loading. Export
forwards those checked bytes rather than recompressing them, so installed browser
exports are byte-exact across supported Python versions. Fresh floating-point
fits are compared with finite relative/absolute tolerance, not falsely called
bitwise portable.

## Follow the entire calculation

- 39 chain parameters: 24 emission-feature weights, nine FROM/TO transition
  scores, three start scores and three end scores.
- Three positive L2 penalties (0.03, 0.3, 3) in each family. Both families select
  by minimum validation conditional NLL/token. No test-driven selection or stop.
- 24 training, 12 validation and 16 test sequences from explicit independent seeds.
  Original synthetic persistent states and noisy integer symbols, not real text.
- Eight fixed observed features: bias, current/left/right symbol, absolute current,
  left/right boundary indicators, and left × current. Both neighbors are available,
  so the model is not a causal streaming predictor.
- Full-batch gradient descent with first acceptable Armijo step among powers of
  1/2; 80-update cap, gradient norm tolerance 1e-5. Three fits stop at tolerance;
  three exhaust the budget. The selected chain exhausts its budget with gradient
  norm about 0.000629. Convexity is not a finite-run convergence certificate.
- Every stored state has full parameters, objective, gradient, accepted step and
  backtrack count. Independent browser gradients check all **12,753 components**.
- Node and adjacent-label marginals, forward/backward messages, Viterbi score
  tables/backpointers, path log probability and sequence entropy remain inspectable.
  Editing observations or transition multipliers is explicit, not relabeled as a
  refit or held-out performance. Invalid inputs clear stale state.

## The normalizer has consequences

An original exact three-path example has allowed paths `A/C`, `B/A`, `B/C`, with
raw weights `1/9`, `1`, `1/9`. Global path normalization gives
`[1/11, 9/11, 1/11]`. Equal start probabilities and separately normalized outgoing
branches give `[1/2, 9/20, 1/20]`. The only outgoing A branch cannot give up its
mass, even when its continuation loses weight. Five evidence weights are retained
as exact fractions.

This is a fixed-score label-bias witness, not a trained MEMM benchmark. The
trained independent-label control above is a different model. The general
inference API uses fully connected finite scores; the separate restricted toy
never disguises a very negative finite score as an impossible transition.

## Independent checks

**22 Python tests and seven Node/real-Chrome tests** cover:

- 60 exhaustive exact-integer path-potential systems with Fraction node/edge
  marginals, exact partition sums and maximum path weights; separate BigInt
  enumeration in JavaScript.
- All 39 central finite differences and independently enumerated expected feature
  counts, independent-label factorization, gradient descent, positive directional
  curvature, boundary orientation, gauge shifts, long extreme log scores and ties.
- All six fresh fits, all 327 optimization states, strict mutation rejection,
  original generator replay and test-label poisoning with unchanged fits/selection.
- All six test metrics and 32 complete selected-model sequence witnesses in
  independent scalar JavaScript.
- 1,311 actual browser states: 327 optimizer states and 984 model/sequence/focus
  combinations. Edited observations, explicit transition controls, keyboard,
  invalid clearing, clipboard success/failure, actual Blob download and
  320/390/1280 responsive overflow and touch targets.
- Eight-module/six-command installed-wheel reproduction, six fresh fits, stored
  and fresh audits, and four byte-exact archive/loader exports outside the checkout.

## Scope and attribution

This is educational implementation work on one inspected synthetic fixture. No
real-language task accuracy, external validation, calibrated posterior confidence,
causal mechanism, population significance or deployment readiness is claimed.
The model supplies `p(y | x)`, not `p(x)` or a generative sequence model. CRFs,
forward/backward and Viterbi follow standard published theory; source links and
notation differences are recorded in [METHOD.md](METHOD.md). The synthetic
fixture, code and prose are original to this project under MIT. Referenced papers
and library code are not redistributed or used as runtime dependencies.
