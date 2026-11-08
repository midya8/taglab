# Method, witnesses and limits

## Conditional, not generative

For observed integer symbols x and a length-T label path y with K=3 labels, define

```
s(x,y) = start[y0] + end[y(T-1)]
       + sum_t dot(W[y_t], phi(x,t))
       + sum_(t=0..T-2) A[y_t, y_(t+1)]
log p(y | x) = s(x,y) - log Z(x)
Z(x) = sum over all K^T paths of exp(s(x,y))
```

Transition matrix rows are FROM and columns are TO. Scores are arbitrary finite
real values, not row-normalized transition probabilities. The whole observed
sequence is fixed when the label distribution is normalized. Nothing here fits
or evaluates the marginal probability of x. This differs from the generative
HMM/unsupervised Baum-Welch work in ChainLab.

The eight observed features, in parameter order, are: 1, current/2, left/2,
right/2, absolute current/2, left-boundary indicator, right-boundary indicator,
left × current/4. A missing neighbor is zero; boundary indicators distinguish it
from an observed zero. Left and right observations can influence a label without
introducing a nonlocal label factor. This is full-sequence, not streaming,
inference. There are no label-derived input features.

Parameter order is three rows of eight W weights, three rows of three A scores,
three start scores, three end scores: 39 total. The independent control trains
the first 24 and fixes the remaining 15 at exactly zero, which factorizes the
conditional label distribution into the same-feature token softmax models. No
feature or training data is withheld from that control.

## Log-space messages

The forward message includes the current emission. The backward message excludes
it and includes the final score:

```
alpha[0,j] = start[j] + emission[0,j]
alpha[t,j] = emission[t,j] + logsumexp_i(alpha[t-1,i] + A[i,j])
logZ = logsumexp_j(alpha[T-1,j] + end[j])
beta[T-1,i] = end[i]
beta[t,i] = logsumexp_j(A[i,j] + emission[t+1,j] + beta[t+1,j])
node[t,j] = exp(alpha[t,j] + beta[t,j] - logZ)
edge[t,i,j] = exp(alpha[t,i] + A[i,j] + emission[t+1,j]
                  + beta[t+1,j] - logZ)
```

Each edge row sums to its left node marginal and each column to its right one.
Length one has start, emission and end scores but no edge. Stable log-sum-exp
subtracts the maximum before exponentiating. Raw path probabilities can underflow
for long sequences; log partition and path log probabilities remain the intended
representation. Finite binary64 computation is not exact symbolic inference.
Scores above magnitude 1e6 and invalid/Boolean/NaN/infinite values reject. The
feature API permits 1-256 integer symbols from -2 to 2. Core inference supports
finite fully connected potentials and is also tested on 1,000-step extremes.
No API represents exact forbidden transitions by pretending `-10000` is infinity.

Viterbi replaces sum with max, retains predecessor indices and traces the best
complete path backward. Ties choose the lowest predecessor in each recursion and
then the lowest final label. This is not a globally lexicographic tie promise.
Marginal decisions independently choose the lowest maximizer of each node
marginal. Viterbi optimizes expected whole-path 0/1 loss under this model;
marginal decisions optimize expected token Hamming loss. Multiplying node
marginals does not yield the joint probability of a path.

Sequence entropy is `logZ - E[s(x,Y)]`; expected start/end, node and edge scores
are all included. The full score witness and path probability are retained.
These are model posterior quantities, not empirically calibrated confidence.

## Supervised objective and analytic gradient

For N complete labeled training sequences and positive lambda:

```
L(theta) = (1/N) sum_n [logZ(x_n) - s(x_n,y_n)]
         + lambda/2 * sum_j theta_j^2
```

This is mean **sequence** conditional NLL plus L2, not mean-token NLL during
training. Validation/test NLL is total unregularized conditional NLL divided by
the number of tokens. The regularization scaling is not changed silently.

The derivative is expected sufficient-feature count minus the observed count,
averaged over sequences, plus `lambda * theta`:

- W[j,f]: sum_t `(node[t,j] - 1[y_t=j]) * phi(x,t,f)`.
- A[i,j]: sum_t `edge[t,i,j] - 1[y_t=i,y_(t+1)=j]`.
- Start/end: corresponding first/last node marginal minus observed indicator.

For the independent control only W is active; sequence parameters remain zero.
No autodiff or external optimizer computes these gradients. Fixed-feature
conditional log partition is convex and positive L2 adds strict curvature on
active coordinates. This mathematical property does not certify a saved finite
iterate as an optimum. Tests check finite differences, independently enumerated
feature moments, a directional curvature bound and actual descent; they are not
an all-input formal proof.

Every fit starts from zero. Up to 80 updates are accepted, with gradient-norm
stopping tolerance 1e-5. Each iteration tries step 1, then halves until

```
L(theta - step*g) <= L(theta) - 0.0001 * step * ||g||^2
```

At most 40 trial steps are allowed. The first acceptable step is used. Histories
retain parameters, full gradients, norm, objective, accepted step and number of
rejected larger steps. Terminal states have step/backtracks zero. The audit
recomputes every gradient and objective, checks rejected steps, movement and
Armijo decrease, terminal status, data digest, validation choice and metrics.
There are six fits and 327 states (including all initial/final states), with
12,753 gradient entries. Three fits stop at tolerance and three exhaust the
budget. The selected chain at lambda 0.3 exhausts 80 updates with gradient norm
0.0006293680943312691. Its final objective is 5.965907053491459 (binary64 value;
inspect the archive rather than relying on rounded prose). The displayed curve
is objective against accepted updates, not time, throughput or held-out progress.

## Original synthetic protocol

Independent Python `random.Random` streams use seeds 2026110801, 2026110802 and
2026110803 for 24 training, 12 validation and 16 test sequences. Each length is
uniformly sampled from 8 through 12. Initial label is uniform over three states;
a state persists with probability 0.78, otherwise moves uniformly to either of
the other two. Conditional symbol probabilities for [-2,-1,0,1,2] are:

```
A: [.48, .28, .14, .07, .03]
B: [.08, .22, .40, .22, .08]
C: [.03, .07, .14, .28, .48]
```

These supplied mechanisms generate observed labeled fixtures. The conditional
model is fitted from labels, not given the generator's probabilities as trained
weights. The generator is not a competing benchmark model. All examples are
retained, so changes in pseudorandom-library internals cannot silently redefine
the archived experiment. There are 124 validation and 164 test tokens.

Both families fit lambda in [0.03, 0.3, 3]. Within each family, minimize validation
NLL/token, using ID as a deterministic secondary key. Test scores and selected
witnesses are evaluated after the IDs are fixed. Tests rotate every test label
and verify unchanged training states, validation scores and selection.

The selected models are `chain-0.3` and `local-0.3`. Test NLL/token is
0.5576136110159814 versus 0.8137151295488614, but Viterbi token errors are 66/164
versus 56/164. Both have 56/164 marginal token errors and 15/16 whole-sequence
errors. The lower-penalty chain has a different test score but cannot replace the
validation-selected choice after seeing test. This one inspected synthetic
fixture is not a pre-registered blind external benchmark, significance test or
real-language performance claim. The family comparison also changes fitted
weights, active parameters and penalty geometry; it is not an isolated causal
transition ablation.

## Exact global/local normalization control

A separate constructed support has only paths A/C, B/A, B/C. Each initial branch
has raw weight 1. Final C receives weight w>0 and final A receives weight 1.
Raw path weights are `[w,1,w]`. Global normalization divides them by `1+2w`.
Local normalization assigns start mass 1/2 to A and B, then normalizes each
outgoing branch: `[1/2, 1/(2*(1+w)), w/(2*(1+w))]`.

At w=1/9, exact global probabilities are `[1/11,9/11,1/11]`, versus local
`[1/2,9/20,1/20]`. A has one successor and always passes on its whole mass;
changes to its only continuation's raw weight cancel locally. The branch's
probability changes under global normalization. Fractions are evaluated for
w=1,1/3,1/9,1/27,1/81, independently in Python Fraction and JavaScript integer
fraction arithmetic. This is an original finite label-bias illustration using
standard theory, not the paper's specific example or a claim of novelty.

Both are fixed-score distributions on the same support, not equally trained
competitors. This toy is not the trained independent-label control and does not
show that local models must always perform worse. Different local features,
structure or refitting change the comparison. The general finite-score chain
API and this exact restricted-support toy deliberately remain separate.

## Independent verification and transport

The original v0.1 suite of 22 Python and seven Node/real-Chrome tests covers 60 independently enumerated
integer-potential systems each in Python Fraction and JavaScript BigInt, all 39
finite differences, full feature moment enumeration, edge consistency, gauges,
long log scores, ties, independent factorization, six fresh fits, all 327 state
audits, test-label poisoning, mutation rejection and installed exports. Exact
oracles sum integer path weights first and then form rational probabilities;
they do not merely compare two copies of the forward recurrence.

Independent JavaScript recomputes all 327 full-data objectives and 12,753 gradient
entries, all six test metrics and 32 complete selected-model sequence witnesses.
Real Chrome visits 327 optimization states and 984 model/sequence/focus states,
plus custom observations, explicit modified transitions, all five normalization
cases, keyboard use, invalid clearing, clipboard failure/success and actual Blob
download. Responsive bounds and 44px targets are checked at 320/390/1280.

The raw report is gzip/base64 packed into three bounded parts with a SHA-256
manifest. Both readers check the uncompressed bytes before parsing. Export
forwards that archive without recompression, making its four browser files
byte-exact. Fresh fits and cross-language floating results use finite
`2e-10 * (1 + abs(reference))` tolerance with exact structure/discrete values;
bitwise floating portability is not claimed. Nine modules and seven commands run
from the installed wheel outside the source tree. No dependency or hidden
network service is required by the model or browser.

## Source ledger and notation differences

1. Lafferty, McCallum and Pereira (2001), *Conditional Random Fields:
   Probabilistic Models for Segmenting and Labeling Sequence Data*.
   [University of Pennsylvania record](https://repository.upenn.edu/entities/publication/c9aea099-b5c8-4fdd-901c-15b6f889e4a7).
   [Author-version mirror](http://web.cs.ucla.edu/~yzsun/classes/2014Spring_CS7280/Papers/Probabilistic_Models/crf_sequence.pdf).
   Read the conditional model, label-bias and finite-observable convexity sections.
   Primary historical theory, not a current performance source. The paper is not
   redistributed; its copyright and personal-use notice are preserved at source.
2. Sutton and McCallum, *An Introduction to Conditional Random Fields*.
   [Tutorial](https://homepages.inf.ed.ac.uk/csutton/publications/crftutv2.pdf).
   Read the linear-chain modeling and observation-feature discussion. The fetched
   extraction covers the early modeling chapters, so it is not used as proof that
   every training chapter was inspected.
3. Laura Ruis (2021-01-25), *Structured Prediction part one*.
   [Derivation](https://lauraruis.github.io/2021/01/25/crfpt1.html).
   Read partition, node/pair moments, likelihood gradients and Viterbi recurrences.
4. Laura Ruis (2021-01-25), *Structured Prediction part two*.
   [Implementation](https://lauraruis.github.io/2021/01/25/crfpt2.html).
   Read log-space message placement, feature sharing and FROM/TO conventions.
   Its batched neural/autodiff implementation is not used as a dependency. Its
   prose calls some scores probabilities; this project explicitly does not.
5. PyTorch official tutorial, *Advanced: Making Dynamic Decisions and the Bi-LSTM CRF*.
   [Tutorial](https://docs.pytorch.org/tutorials/beginner/nlp/advanced_tutorial.md).
   Read the conditional score and forward/Viterbi sections. Its transition rows
   are TO and columns FROM, opposite this project's explicit orientation. Its
   finite negative boundary sentinels are not adopted as exact zero probability.
6. `pytorch-crf` reference documentation.
   [Docs](https://pytorch-crf.readthedocs.io/en/stable/).
   Read separate start/end scores, conditional log likelihood, Viterbi and
   sequence/token reduction semantics. No runtime dependency or claim that this
   package independently verified our outputs.
7. SuPar, *Chain* reference.
   [Docs](https://parser.yzhang.site/structs/chain.html).
   Read log partition, path log probability, argmax and entropy interfaces.
   Its tensor/masking and semi-Markov capabilities are not implemented here.

Sources were fetched and relevant sections read on November 8, 2026. They span
primary research, theoretical/educational derivation, official tutorial and
implementation reference documentation. One search hit labeled as the original
paper was actually Cornell teaching slides; it was not used as the primary
paper. Score orientation and message-inclusion conventions differ between
sources. The formulas above fix this project's convention rather than silently
merging incompatible array layouts. The implementation, synthetic fixture,
tests, browser and prose here are original; cited theory is standard.


## v0.2: exact position-wise label-event conditioning

For fixed observations x and unchanged finite scores s(x,y), let allowed[t]
be a subset of the label indices at position t. The event E is the Cartesian
product of those subsets. Define

```text
Z_E(x) = sum_{y in E} exp(s(x,y))
log P(E | x) = log Z_E(x) - log Z(x)
P(y | x,E) = exp(s(x,y)) / Z_E(x), for y in E, and zero otherwise.
d log P(E | x) / d theta = E[features | x,E] - E[features | x]
```

The masked log-space recurrence excludes forbidden labels exactly, using
negative infinity internally. Alpha includes the current unary; beta excludes
it. The restricted node and adjacent-pair formulas use the same finite scores
and their own partition. Viterbi runs over surviving labels, sorted ascending;
ties select the lowest predecessor and final label, not a globally
lexicographic path. The score API still rejects nonfinite input potentials.
Masks, rather than finite negative sentinels, define support.

The fully connected score model means the number of allowed paths is exactly
the product of per-position subset sizes, serialized as a decimal string. Any
empty subset makes the event impossible. Its conditional distribution and
gradient are undefined, so the JSON response has `possible=false`,
`conditional=null`, `log_event_gradient=null`, `event_probability=0` and
`log_event_probability=null` (representing extended-real negative infinity).
A nonempty event can have a floating-point probability of zero by underflow;
it instead has `possible=true`, a finite log probability and an explicit
underflow flag. Forbidden entries in message/score tables are also null, never
finite approximations to impossible scores. Tiny positive computed log-event
probabilities from roundoff are clamped to zero.

All-free masks recover the original inference. Fully singleton masks give one
path, unit conditional mass and zero entropy up to roundoff. For a singleton
restriction at one position, the event probability equals its unrestricted
node marginal. Disjoint event probabilities add; nested events cannot increase
probability. Entropy given a particular event can increase or decrease. These
identities have independent path-enumeration and numerical controls.

`label_event` evaluates the same eight observed features and returns the full
39-dimensional expected-feature difference. For the independent fitted control,
the last 15 sequence coordinates remain fixed; the displayed 39-vector is the
derivative of the ambient chain score family at that control, not a statement
that all coordinates were trained. No update, retraining or partial-label loss
optimization is performed. In its actual factorized distribution, a restriction
at one position leaves unrestricted positions' node marginals unchanged.

New tests: nine Python tests and five Node/Chrome tests. Sixty Fraction and sixty
separate BigInt allowed-path oracles, all 39 central finite differences,
additivity/nesting, entropy-increase example, pointmass, all-free equality,
long-score boundary, ties, invalid subsets, strict JSON, CLI and underflow checks.
3,336 Python/JavaScript witnesses cover all six existing fitted models, each of
16 test observation sequences, every position's three singleton restrictions
(2,952 total), plus all-free, one-complete-path, multi-subset and impossible
patterns (384 total). They compare probabilities, partitions, nodes, pairs,
complete best paths, entropy and all 39 event derivatives. Existing model
training and held-out scores remain unchanged. A separate 3,336-query actual
Chrome replay checks displayed bars and provenance against that module, not
against a new mathematical oracle. Both inspectors total 4,647 UI states/queries.
Nine modules and seven commands also reproduce from the installed wheel.

### Additional sources and narrower scope

- Papay, Klinger and Padó, [Constraining Linear-chain CRFs to Regular Languages](https://arxiv.org/html/2106.07306v6), sections 2-5: explicitly conditions a CRF distribution on an output event and distinguishes constrained decoding from constrained training. This project implements only position-wise subsets, not its NFA construction, nonlocal regular-language constraints or training results. The paper's corrected-score notice is preserved at source; no experimental score is reused here.
- [TorchCondiRF](https://github.com/MPKonst/torchcondirf), implementation README: restricted partitions and fixed-position label subsets are established functionality. No implementation is imported or copied; no top-k claim here.
- [partial-crfsuite](https://github.com/Oneplus/partial-crfsuite), implementation README: ambiguous position-wise labels are used for partial-annotation training. This project does not perform that training, copy its code, inherit its NLP results or adopt its license.

These supplement the seven-source v0.1 ledger above. Relevant sections were
fetched and read on November 8, 2026. Restricting and renormalizing a fixed
model's output is standard conditioning, not a new method, a causal
intervention, a ground-truth correction or a model-quality improvement.


## v0.3: backward conditional sampling

Keep fixed observations, scores and a nonempty Cartesian label event. Let the restricted forward message include the current unary, as above. The final categorical weights are `exp(alpha[T-1,j] + end[j])`. After drawing label j at position t+1, the predecessor weights at t are `exp(alpha[t,i] + transition[i,j])`. Normalize each row in log space. Future terms, including the next unary, are constant given the successor and cancel. By the chain rule these reverse conditional probabilities multiply to the full conditional path probability. The independent-node control instead multiplies the smoothed one-position marginals, which generally does not equal the joint path law.

An explicit two-label, two-position example uses zero unary/start/end scores and transition potentials `[[9,1],[1,9]]`. Both node marginals are `[1/2,1/2]`. The joint path weights are `[9,1,1,9]/20`, whereas independent nodes put `1/4` on each path. The pair total variation is 2/5. This is a fixed-score identity, not a learned-model advantage or a causal experiment.

The low-level sample API accepts exactly one finite uniform in [0,1) per position, indexed in chronological order and consumed backwards. CDF selection uses the first positive bin whose upper endpoint strictly exceeds u; equality moves to the next positive bin. The last positive bin ends at one. Exact forbidden labels have zero weights. Finite probabilities can underflow, and even positive probabilities may have collapsed CDF bins due to roundoff. Both are listed in each trace. Log probabilities remain visible. No exact real-number sampling or guaranteed reachability of every mathematically positive path is claimed.

`uniforms(seed,count)` uses xorshift32: XOR with left13, right17, left5, truncating to32 bits, then `(state+0.5)/2**32`. Nonzero integer seeds only. This portable deterministic lattice gives reproducible illustrations, not cryptographic or certified independent randomness. Fixed-stream frequencies are not confidence intervals or hypothesis tests. The browser displays64 draws and a selected reverse trace; manual uniforms produce one draw. Both algorithms receive the same uniforms as an explicit coupling, not independent study groups. Stored true labels are not read.

The trace reports `model_path_log_probability` under the restricted model, `proposal_log_probability` as the ideal log-kernel product of the selected sampling method, and `numeric_kernel_log_probability` from normalized finite probabilities. None asserts that a deterministic PRNG has the continuous target distribution. Preparation is O(T K²); each draw is O(T K) after preparation. The browser reuses one prepared result for its64 draws. No speed benchmark or parameter-posterior sampling is claimed.

Six Python tests cover60 independently enumerated Fraction fixtures and all428 supported paths via midpoint uniforms; independent path weights verify the reverse-product identity. Tests also cover exact CDF boundaries, singleton events, factorized agreement, correlated independent-node disagreement, underflow, collapsed CDF bins, long-score limits, strict invalids, CLI and input preservation. Four new Node/Chrome tests cover60 independently enumerated BigInt fixtures, 2,304 Python/JavaScript traces (six models,16 sequences,four events,three seeds,two laws),384 actual-browser queries, manual/seeded inputs, invalid and impossible-event clearing, keyboard focus, actual download, clipboard and320/390/1280 layouts. Browser tests compare rendered evidence to the separately oracle-checked module, not a third mathematical oracle.

### Sampling sources

- Marsaglia, [Xorshift RNGs](https://www.jstatsoft.org/article/view/v008i14/), Journal of Statistical Software8(14),2003, and [Xorshift RNGs for small systems](https://excamera.com/sphinx/article-xorshift.html): standard XOR/shift generator and13/17/5 coefficients. The mid-bin conversion here is explicitly specified. No statistical-quality claim is inherited.

- Douc, Garivier, Moulines and Olsson, [On the Forward Filtering Backward Smoothing Particle Approximations of the Smoothing Distribution in General State Spaces Models](https://arxiv.org/pdf/0904.0316), section2, equations2-3: reverse-kernel factorization background. Only the finite-state kernel identity is used; no particle approximation, convergence theorem or particle implementation is claimed. Relevant sections, not the whole paper, were read.
- MIT6.0411/16.420, [Sampling, Max-Product and Discrete HMMs](https://airr.mit.edu/_static/fall23/lectures/lec12/L12-Sampling-HMM.pdf): sampling and smoothing background. TagLab's conditional-label score model is not a generative HMM.
- Murphy, [HMM toolbox usage](https://www.cs.ubc.ca/~murphyk/Software/HMM/hmm_usage.html): generative sampling, Viterbi and fitting are distinct operations. TagLab does not import the toolbox or sample observations.

These supplement the conditioning and original theory sources above. Fetched November8,2026. Backward conditional sampling is standard theory; original code, tests and visual witnesses do not make it a new method.
