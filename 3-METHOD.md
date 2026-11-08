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

22 Python and seven Node/real-Chrome tests cover 60 independently enumerated
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
bitwise floating portability is not claimed. Eight modules and six commands run
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
