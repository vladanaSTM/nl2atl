# Technical Appendix: Translating Natural Language to Strategic Temporal Specifications via LLMs

*Anonymous AAAI-27 Supplementary Material*

The syntactic trees in this appendix employ X-bar theoretic structures to map natural language sentences to ATL/ATL$^\ast$ formulas. Some trees use representational conventions that prioritize transparent scope and argument structure for translation purposes, rather than strictly derivational syntactic analyses. Therefore, we explicitly flag them as interpretive devices for the syntax-semantics interface. These conventions do not affect the empirical coverage of the test suite, which is designed to probe whether models correctly recover logical form from surface form.

# Annotation Guidelines for NL-to-ATL/ATL$^\ast$ Translation

This section makes explicit the annotation policy used to validate the gold dataset. The dataset is intended to evaluate whether a model can recover the strategic-temporal intent expressed by a natural-language requirement, not whether it can repair the requirement into a different specification that would be preferable from an engineering point of view. The gold formulas therefore follow the linguistic content of the input sentence, while using a fixed set of conventions for temporal scope, coalition attribution, and structurally ambiguous constructions.

## General Translation Target

Each example maps a natural-language requirement to one or more ATL/ATL$^\ast$ formulas. The strategic modality $\langle\!\langle A \rangle\!\rangle$ is used when the sentence attributes an enforceable ability or guarantee to agent or coalition $A$. Atomic propositions are written as lower-case, underscore-separated predicates that preserve the content words of the requirement. Agent names are written as coalition members inside the strategic operator and are separated by commas when several agents form one coalition. Boolean connectives preserve the intended propositional structure of the input: *and* is translated as $\wedge$ or as coalition membership depending on its syntactic role, *or* is translated as inclusive disjunction, *if ... then ...* and related conditional constructions are translated with implication, and negation is attached to the proposition or event that is negated in the natural-language sentence.

When the input is genuinely compatible with multiple intended readings, the dataset may provide multiple gold formulas. In those cases the outputs are not ranked paraphrases: they are the accepted readings for that item. Downstream exact-match and judge-based evaluation should therefore treat multi-output examples as multi-reading items and should check whether the system recovers all required readings, unless an experiment explicitly redefines the task as single-reading generation.

## Temporal Operators and Scope

The temporal operators follow their standard ATL/ATL$^\ast$ interpretation. Phrases such as *at the next step*, *immediately after*, or *at the following state* license $X$. Phrases such as *sooner or later*, *eventually*, *in the future*, or *will ultimately* license $F$. Phrases such as *always*, *every time*, *whenever*, *never*, *remain*, and *will not ever* license $G$, with *never* contributing both temporal persistence and Boolean negation when it modifies an event or state. Phrases such as *until*, *as long as ... before ...*, *keep ... until ...*, and *remain ... until ...* license $U$.

The default annotation policy is surface-driven. If a requirement states that an agent can guarantee an eventuality, the eventuality is translated locally under the strategic operator, for example $\langle\!\langle A \rangle\!\rangle F p$. If the requirement states that an agent can always guarantee a conditional, or uses a habitual universal cue such as *every time* or *whenever*, the conditional is placed under a dominating $G$, for example $\langle\!\langle A \rangle\!\rangle G(p \rightarrow F q)$ or $\langle\!\langle A \rangle\!\rangle G(p \rightarrow X q)$. If the sentence contains a conditional but no explicit habitual or universal cue, the local reading $\langle\!\langle A \rangle\!\rangle(p \rightarrow F q)$ or $\langle\!\langle A \rangle\!\rangle(p \rightarrow X q)$ is allowed as the gold reading unless the item was explicitly adjudicated as persistent by the reviewers.

The annotation also avoids adding redundant temporal operators merely because they are pragmatically plausible. For example, an $U$-construction already encodes temporal progression from the current state, so a dominating $G$ is not inserted unless the input contains an explicit global cue. Conversely, when the consequent itself is persistent, as in *then it will never start drilling*, the $G$ operator is placed in the consequent, yielding formulas of the form $\langle\!\langle A \rangle\!\rangle(p \rightarrow G\neg q)$ or, with an explicit habitual cue, $\langle\!\langle A \rangle\!\rangle G(p \rightarrow G\neg q)$. This policy separates the scope of the rule from the scope of the persistent consequent.

Temporal sequencing expressions are annotated according to the event structure expressed in the input. *After $p$, eventually $q$* is represented as a state or event condition followed by an eventuality, often using $p \wedge X F q$ when the sentence expresses that the relevant postcondition starts after the triggering event. *Before $q$* is represented through an until-style condition when the sentence describes a state that holds up to the occurrence of $q$. These conventions are meant to keep the mapping linguistically transparent rather than to infer a stronger engineering invariant.

## Coalition Attribution

Coalition membership is determined by the grammatical subject of the strategic predicate and by explicit coordination markers. If the input says that *the robot can guarantee ...*, the coalition is $\langle\!\langle Robot \rangle\!\rangle$. If it says that *the robot and the operator together can guarantee ...*, the coalition is $\langle\!\langle Robot,Operator \rangle\!\rangle$. Lexical cues such as *together*, *jointly*, *as a coalition*, and coordinated subjects before the strategic verb license a multi-agent coalition.

Agents mentioned inside antecedents, temporal clauses, or embedded propositions are not automatically inserted into the strategic coalition. For instance, in a requirement of the form *If the user presses cancel, the machine can guarantee that eventually a refund is issued*, the user action is treated as an atomic proposition in the environment, and the strategic ability belongs to the machine: $user\_cancel \rightarrow \langle\!\langle Machine \rangle\!\rangle F\,refund$. The same principle applies when a human confirmation, external event, failure, or environmental condition appears as a trigger. Such entities enter the coalition only when the sentence explicitly attributes the guarantee to them.

The annotation distinguishes coordination inside the coalition from Boolean conjunction in the objective. In *the user and the machine together can guarantee that the system never enters an error state*, *and* links two coalition members. In *the user can guarantee that eventually the ticket is printed and eventually the gate opens*, *and* links two strategic-temporal objectives. The first case yields one coalition modality with several members; the second yields a conjunction of formulas or a conjunction inside the objective, depending on the intended scope.

When an input is pragmatically cooperative but lacks explicit coalition-forming language, the conservative annotation keeps only the agent that grammatically bears the strategic predicate. This prevents the dataset from rewarding models for adding unstated agents to the coalition. The benchmark is a translation task from informal requirements to formal specifications, not a commonsense specification-repair task.

## Ambiguity Families

The dataset contains examples designed to test whether models recover logical form from non-canonical surface syntax. For VP ellipsis, omitted material is reconstructed from the antecedent clause before translation. Thus, in a sentence such as *the arm can always guarantee ..., and the mobile robot can too*, the second conjunct inherits the full strategic-temporal objective of the first conjunct, with the coalition member replaced by the second subject where appropriate.

For right dislocation, left dislocation, and resumptive-pronoun constructions, the displaced noun phrase is interpreted as the semantic argument associated with the pronoun or gap. The translation is therefore based on the resolved argument structure, not on the surface position of the noun phrase. These examples test whether a model can identify the agent or proposition despite non-canonical word order.

For Right Node Raising, the shared constituent is distributed over all coordinated clauses that license it. If one conjunct is negated, the negation is preserved only for that conjunct. The resulting formula therefore reflects the full reconstructed content of each coordinate clause, rather than translating the shared phrase only once.

For quantifier-scope ambiguity, the dataset records the intended readings as separate gold formulas when both readings are part of the example. Surface-scope readings allow each quantified agent to have its own corresponding object or outcome, while inverse-scope readings represent a shared object or outcome across agents. Multi-reading QSA examples therefore intentionally contain more than one output formula, and evaluation scripts should preserve this multi-output structure.

## Gold Validation Procedure

The gold dataset was validated through expert discussion. Disagreements were resolved by adopting the most conservative translation that follows the explicit linguistic evidence in the input, unless the example was specifically designed to encode multiple readings. Reviewer comments in the informal dataset PDF document rejected alternatives, accepted alternatives, and scope decisions, but the canonical JSON file is the authoritative machine-readable gold dataset. The reviewer PDF is retained as an audit trail and should be checked for ID-level alignment with the JSON, while the JSON should be used for training, evaluation, and reproducibility checks.

## Example 1: Quantifier Scope Ambiguity Surface Structure

<div class="example">

Consider the sentence:

> Every rover can guarantee that it will never enter a hazardous area.

The sentence contains two quantificational expressions: the universal DP *every rover* and the indefinite DP *a hazardous area*. This gives rise to two possible semantic readings, (i) a surface-scope reading and (ii) an inverse-scope reading.

</div>

## Example 2: Surface-Scope Interpretation

<div class="example">

Under the surface-scope interpretation, the universal quantifier takes wider scope than the existential quantifier. Informally, the reading is: "for every rover, there is a possibly different hazardous area such that the rover can guarantee that it will never enter it".

</div>

## Example 3: Inverse-Scope Interpretation

<div class="example">

Under the inverse-scope interpretation, the indefinite DP receives wider semantic scope than the universal DP. Informally, the reading is: "there is a hazardous area such that every rover can guarantee that it will never enter that same area".

</div>

The tree in FigureÂ <a href="#fig:qsa-inverse-scope" data-reference-type="ref" data-reference="fig:qsa-inverse-scope">3</a> employs a scope-marking convention rather than a syntactic derivation. The indefinite DP "a hazardous area" is shown TP-adjoined to indicate its wide-scope existential interpretation, not to assert a literal movement operation. This representation is compatible with analyses in which indefinites receive exceptional scope via choice functions or referentiality , without violating island constraints on QR. It should not be interpreted as a claim about syntactic derivation.

## Example 4: VP-Ellipsis under Sentential Coordination

<div class="example">

Consider the sentence:

> The captain can guarantee that at the next step possession will be recovered, and the midfielder can too.

The second conjunct contains an elliptical VP. The phrase *can too* must be interpreted by recovering the full VP from the first conjunct. Consequently, the ATL/ATL$^\ast$ translation must duplicate the strategic-temporal objective for the second agent.

</div>

The corresponding translation reconstructs the elided material before formalization:
```latex
\begin{aligned}
&\langle\!\langle Captain \rangle\!\rangle X\,possession\_recovered \\
&\quad \wedge\; \langle\!\langle Midfielder \rangle\!\rangle X\,possession\_recovered.
\end{aligned}
```
Thus, the example tests whether a model can recover omitted material and duplicate the full strategic-temporal objective, rather than translating only the overt surface string.

## Example 6: Right Dislocation

<div class="example">

Consider the sentence:

> They can eventually ensure the game ends, the players.

The NP *the players* appears at the right periphery of the matrix clause, outside its canonical argument position. In conformity with X-bar theoryâ€™s binary-branching requirement, the dislocated element is represented as a right-adjunct: the outer TP has exactly two daughters, the inner TP (the base clause) and the right-adjoined $\mathrm{DP}_i$, so that no node carries more than two immediate children. The subject pronoun *they* occupies the base-generated subject position and is coreferential with the dislocated NP (co-indexed as $\mathrm{DP}_i$). The adverb *eventually* adjoins to VP, itself producing a binary VP-over-VP adjunction structure.

</div>

## Example 7: Left Dislocation

<div class="example">

Consider the sentence:

> The vending machine, Mary can ensure it will work.

The DP *the vending machine* is displaced to the left periphery of the clause, occupying Spec,TopP in the cartographic analysis of the left periphery . A resumptive pronoun *it* fills the canonical argument position within the embedded complement of *ensure*, co-indexed with the topic $\mathrm{DP}_i$. Binary branching is satisfied throughout: TopP branches into Spec and Top$'$; Top$'$ branches into Top$^\circ$ and TP; and so on down every projection.

</div>

## Example 8: Right Node Raising

<div class="example">

Consider the sentence:

> Bob has, but Albert has not, a strategy to ensure that the vending machine is always operative.

The DP *a strategy to ensure that the vending machine is always operative* is shared between two coordinated TPs. In conformity with X-bar theoryâ€™s binary-branching requirement, the shared element is represented as a right-adjunct to CoordP: the outer CoordP has exactly two daughters, the inner CoordP (the coordinate structure proper) and the right-adjoined $\mathrm{DP}_j$, so that no node carries more than two immediate children. Each conjunct contains a gap in object position, marked by trace $t_j$. To reflect the formal syntax of the second conjunct where the verb precedes negation (*has not*), the verb *has* is raised to T in both conjuncts, leaving a trace $t_v$ heading the VP. The negation *not* heads a NegP below T$'$. The internal structure of the complex shared DP is omitted for readability.

</div>

# Syntactic Tree Figures

The following full-width figures visualize the syntactic structures and interpretive conventions discussed above. They are placed together to keep the two-column text readable while preserving AAAI formatting.

<figure id="fig:qsa-surface" data-latex-placement="t">
<div class="tcolorbox">
<div class="adjustbox">
<p><span>max width=</span></p>
<div class="forest">
<p>xt [CP [C<span class="math inline">â€²</span> [C [<span class="math inline">âŒ€</span>]] [TP [<span style="color: QP1c"><strong>DP<span class="math inline"><sub><em>i</em></sub></span></strong></span> [D<span class="math inline">â€²</span> [D [every]] [NP [N [rover]]]] ] [T<span class="math inline">â€²</span> [T [can]] [VP [V<span class="math inline">â€²</span> [V [guarantee]] [CP [C<span class="math inline">â€²</span> [C [that]] [TP [DP [it]] [T<span class="math inline">â€²</span> [T [will]] [NegP [Neg<span class="math inline">â€²</span> [Neg [never]] [VP [V<span class="math inline">â€²</span> [V [enter]] [<span style="color: QP2c"><strong>DP<span class="math inline"><sub><em>j</em></sub></span></strong></span> [D<span class="math inline">â€²</span> [D [a]] [NP [AP [hazardous]] [N [area]]]] ] ] ] ] ] ] ] ] ] ] ] ] ] ] ]</p>
</div>
</div>
</div>
<figcaption>Surface structure for a quantifier scope ambiguity involving a universal DP and an indefinite DP.</figcaption>
</figure>

<figure id="fig:qsa-surface-scope" data-latex-placement="t">
<div class="tcolorbox">
<div class="adjustbox">
<p><span>max width=</span></p>
<div class="forest">
<p>xt [CP [C<span class="math inline">â€²</span> [C [<span class="math inline">âŒ€</span>]] [TP [<span style="color: QP1c"><strong>DP<span class="math inline"><sub><em>i</em></sub></span></strong></span> [every rover]] [TP [DP [<span class="math inline"><em>t</em><sub><em>i</em></sub></span>]] [T<span class="math inline">â€²</span> [T [can]] [VP [V<span class="math inline">â€²</span> [V [guarantee]] [CP [C<span class="math inline">â€²</span> [C [that]] [TP [DP [it]] [T<span class="math inline">â€²</span> [T [will]] [NegP [Neg<span class="math inline">â€²</span> [Neg [never]] [VP [<span style="color: QP2c"><strong>DP<span class="math inline"><sub><em>j</em></sub></span></strong></span> [a hazardous area]] [VP [V<span class="math inline">â€²</span> [V [enter]] [DP [<span class="math inline"><em>t</em><sub><em>j</em></sub></span>]]] ] ] ] ] ] ] ] ] ] ] ] ] ] ] ]</p>
</div>
</div>
</div>
<figcaption>Surface-scope interpretation, where the universal DP outscopes the indefinite DP.</figcaption>
</figure>

<figure id="fig:qsa-inverse-scope" data-latex-placement="t">
<div class="tcolorbox">
<div class="adjustbox">
<p><span>max width=</span></p>
<div class="forest">
<p>xt [CP [C<span class="math inline">â€²</span> [C [<span class="math inline">âŒ€</span>]] [TP [<span style="color: QP2c"><strong>DP<span class="math inline"><sub><em>j</em></sub></span></strong></span> [a hazardous area]] [TP [<span style="color: QP1c"><strong>DP<span class="math inline"><sub><em>i</em></sub></span></strong></span> [every rover]] [TP [DP [<span class="math inline"><em>t</em><sub><em>i</em></sub></span>]] [T<span class="math inline">â€²</span> [T [can]] [VP [V<span class="math inline">â€²</span> [V [guarantee]] [CP [C<span class="math inline">â€²</span> [C [that]] [TP [DP [it]] [T<span class="math inline">â€²</span> [T [will]] [NegP [Neg<span class="math inline">â€²</span> [Neg [never]] [VP [V<span class="math inline">â€²</span> [V [enter]] [DP [<span class="math inline"><em>t</em><sub><em>j</em></sub></span>]]] ] ] ] ] ] ] ] ] ] ] ] ] ] ] ]</p>
</div>
</div>
</div>
<figcaption>Inverse-scope interpretation, where the indefinite DP receives wider semantic scope than the universal DP.</figcaption>
</figure>

<figure id="fig:vp-ellipsis" data-latex-placement="t">
<div class="tcolorbox">
<div class="adjustbox">
<p><span>max width=</span></p>
<div class="forest">
<p>xt [CoordP [TP<span class="math inline"><sub>1</sub></span> [DP [The captain]] [T<span class="math inline">â€²</span> [T [can]] [VP<span class="math inline"><sub><em>i</em></sub></span> [V<span class="math inline">â€²</span> [V [guarantee]] [CP [C<span class="math inline">â€²</span> [C [that]] [TP [PP [at the next step]] [TP [DP [possession]] [T<span class="math inline">â€²</span> [T [will]] [VP [be recovered]]] ] ] ] ] ] ] ] ] [Coord<span class="math inline">â€²</span> [Coord [and]] [TP<span class="math inline"><sub>2</sub></span> [DP [the midfielder]] [T<span class="math inline">â€²</span> [T [can]] [VP [<span class="math inline"><em>e</em><sub><em>i</em></sub></span>]] ] ] ] ]</p>
</div>
</div>
</div>
<figcaption>VP-ellipsis under sentential coordination. The elided VP in the second conjunct is represented as an indexed empty category <span class="math inline"><em>e</em><sub><em>i</em></sub></span>, co-indexed with the antecedent VP<span class="math inline"><sub><em>i</em></sub></span> in the first conjunct. This notation indicates identity of interpretation without commitment to the theoretical mechanism (deletion vs. pro-form).</figcaption>
</figure>

<figure id="fig:right-dislocation" data-latex-placement="t">
<div class="tcolorbox">
<div class="adjustbox">
<p><span>max width=</span></p>
<div class="forest">
<p>xt [TP [TP [<span style="color: RDc"><strong>DP<span class="math inline"><sub><em>i</em></sub></span></strong></span> [they]] [T<span class="math inline">â€²</span> [T [can]] [VP [AdvP [Adv [eventually]]] [VP [V<span class="math inline">â€²</span> [V [ensure]] [CP [C<span class="math inline">â€²</span> [C [<span class="math inline">âŒ€</span>]] [TP [DP [D<span class="math inline">â€²</span> [D [the]] [NP [N [game]]]]] [T<span class="math inline">â€²</span> [T [<span class="math inline">âŒ€</span>]] [VP [V<span class="math inline">â€²</span> [V [ends]]]] ] ] ] ] ] ] ] ] ] [<span style="color: RDc"><strong>DP<span class="math inline"><sub><em>i</em></sub></span></strong></span><br />
<span>[right-adjoined]</span> [D<span class="math inline">â€²</span> [D [the]] [NP [N [players]]]] ] ]</p>
</div>
</div>
</div>
<figcaption>Right dislocation via right-adjunction. The outer TP has exactly two daughters: the inner TP (the base clause) and the right-adjoined <span class="math inline">DP<sub><em>i</em></sub></span> <em>the players</em> (in <span style="color: RDc"><strong>green</strong></span>). The resumptive subject <em>they</em> and the dislocated NP share index <span class="math inline"><em>i</em></span>. The adverb <em>eventually</em> adjoins to VP, yielding a VP-over-VP binary structure.</figcaption>
</figure>

<figure id="fig:left-dislocation" data-latex-placement="t">
<div class="tcolorbox">
<div class="adjustbox">
<p><span>max width=</span></p>
<div class="forest">
<p>xt [TopP [<span style="color: LDc"><strong>DP<span class="math inline"><sub><em>i</em></sub></span></strong></span> [D<span class="math inline">â€²</span> [D [the]] [NP [N [vending<br />
machine]]]] ] [Top<span class="math inline">â€²</span> [Top [<span class="math inline">âŒ€</span>]] [TP [DP [Mary]] [T<span class="math inline">â€²</span> [T [can]] [VP [V<span class="math inline">â€²</span> [V [ensure]] [CP [C<span class="math inline">â€²</span> [C [<span class="math inline">âŒ€</span>]] [TP [<span style="color: LDc"><strong>DP<span class="math inline"><sub><em>i</em></sub></span></strong></span> [it]] [T<span class="math inline">â€²</span> [T [will]] [VP [V<span class="math inline">â€²</span> [V [work]]]] ] ] ] ] ] ] ] ] ] ]</p>
</div>
</div>
</div>
<figcaption>Left dislocation: <em>the vending machine</em> (<span class="math inline">DP<sub><em>i</em></sub></span>, in <span style="color: LDc"><strong>amber</strong></span>) occupies Spec,TopP in the left periphery. The resumptive pronoun <em>it</em> (<span class="math inline">DP<sub><em>i</em></sub></span>) within the embedded clause is co-indexed with the topic. Top<span class="math inline"><sup>âˆ˜</sup></span> is phonologically null (<span class="math inline">âŒ€</span>). All nodes branch at most binary.</figcaption>
</figure>

<figure id="fig:rnr" data-latex-placement="t">
<div class="tcolorbox">
<div class="adjustbox">
<p><span>max width=</span></p>
<div class="forest">
<p>xt [CoordP [CoordP [TP<span class="math inline"><sub>1</sub></span> [DP [Bob]] [T<span class="math inline">â€²</span> [T [has]] [VP [V<span class="math inline">â€²</span> [V [<span class="math inline"><em>t</em><sub><em>v</em></sub></span>]] [DP [<span class="math inline"><em>t</em><sub><em>j</em></sub></span>]]]] ] ] [Coord<span class="math inline">â€²</span> [Coord [but]] [TP<span class="math inline"><sub>2</sub></span> [DP [Albert]] [T<span class="math inline">â€²</span> [T [has]] [NegP [Neg<span class="math inline">â€²</span> [Neg [not]] [VP [V<span class="math inline">â€²</span> [V [<span class="math inline"><em>t</em><sub><em>v</em></sub></span>]] [DP [<span class="math inline"><em>t</em><sub><em>j</em></sub></span>]]]] ] ] ] ] ] ] [<span style="color: RNRc"><strong>DP<span class="math inline"><sub><em>j</em></sub></span></strong></span><br />
<span>[right-adjoined / shared]</span> [D<span class="math inline">â€²</span> [D [a]] [NP [N [strategy<br />
to ensure <span class="math inline">â€¦</span>]]]] ] ]</p>
</div>
</div>
</div>
<figcaption>Right Node Raising via right-adjunction. The shared DP is represented as right-adjoined to CoordP to maintain binary branching while ensuring the ATL translation associates the objective with both conjuncts. This is a representational convention for the syntax-semantics interface, not a syntactic derivation. The traces <span class="math inline"><em>t</em><sub><em>j</em></sub></span> in object position mark interpretive gaps; no movement operation is claimed.</figcaption>
</figure>

# Model-Checker Integration Interface

To make the end-to-end workflow concrete, FigureÂ <a href="#fig:vitamin" data-reference-type="ref" data-reference="fig:vitamin">8</a> shows the natural-language-to-ATL$^\ast$ translation surfaced directly inside the <span class="smallcaps">VITAMIN</span> model checker. A user enters a strategic requirement in natural language; the translation service returns a well-formed ATL/ATL$^\ast$ formula that is parsed by the same front-end used for verification, so that only syntactically valid specifications enter the model-checking phase.

<figure id="fig:vitamin" data-latex-placement="h">
<img src="genVITAMIN.png" />
<figcaption>Natural-language to ATL<span class="math inline"><sup>â‹†</sup></span> translation surfaced inside the <span class="smallcaps">genVITAMIN</span> interface.</figcaption>
</figure>

# Relevance to the Dataset

These examples illustrate why ambiguity-aware NL-to-ATL/ATL$^\ast$ translation cannot be reduced to direct keyword substitution. Quantifier scope ambiguity may require multiple admissible formal outputs, while VP-ellipsis requires reconstruction of omitted material before translation. Similar considerations apply to right dislocation, left dislocation, and Right Node Raising. In all cases, the resulting formula must preserve the intended scope relations among strategic modalities, temporal operators, Boolean connectives, and atomic propositions.

# Translation-and-Evaluation Algorithm

AlgorithmÂ <a href="#alg:pipeline" data-reference-type="ref" data-reference="alg:pipeline">[alg:pipeline]</a> details the end-to-end translation-and-evaluation loop that the `nl2atl` architecture (FigureÂ 1 in the main text) realizes: it draws a fixed stratified split, fine-tunes each open-weight model on the training split (skipped for the API baselines), generates a prediction for every test requirement under greedy decoding, and scores each prediction with the two-tier exact-match-then-judge protocol before aggregating the per-prediction verdicts over seeds.

**Algorithm 1. Translation and Evaluation**

**Require:** Dataset $\mathcal{D}$, model config $M$, training seeds $S$  
**Ensure:** Raw predictions $\mathcal{P}$, per-prediction verdicts $\mathcal{V}$, and aggregated accuracy

```text
1.  (D_tr, D_te) <- StratifiedSplit(D)  // fixed canonical split; few-shot exemplars held out
2.  Instantiate the inference backend for M via the Model Abstraction Layer.
3.  P <- empty set; V <- empty set.
4.  For each training seed s in S:
5.      M_s <- FineTune(M, D_tr, s) with LoRA  // skipped for API baselines
6.      For each requirement r in D_te:
7.          prompt <- BuildPrompt(r, M)  // system prompt plus optional few-shot examples
8.          phi_hat <- Infer(prompt, M_s)  // greedy decoding; local or API backend
9.          Add (r, phi_hat, s, metadata) to P.
10. For each (r, phi_hat, s) in P with gold set Phi_ref:
11.     If ExactMatch(phi_hat, Phi_ref):
12.         correct <- true
13.     Else:
14.         correct <- LLMJudge(phi_hat, Phi_ref)
15.     Add (r, s, correct) to V.
16. accuracy <- Aggregate(V, S)  // mean over seeds with dispersion
17. Return P, V, and accuracy.
```

# Experimental Configuration

## Computing Infrastructure and Software

All local fine-tuning and inference were run on Linux compute nodes equipped with NVIDIA A100 GPUs, dispatched as independent jobs through a SLURM-managed cluster. The proprietary baselines and the LLM judges were accessed through the Azure OpenAI service. The framework is implemented in Python ($\geq 3.10$) and released under the MIT license. The experiments build, among others, on PyTorchÂ 2.10, Hugging Face TransformersÂ 5.0, PEFTÂ 0.18 (low-rank adaptation), `bitsandbytes`Â 0.49 ($4$-bit quantization), TRLÂ 0.27 (supervised fine-tuning), AccelerateÂ 1.12, DatasetsÂ 4.5, and scikit-learnÂ 1.7. The released repository pins an exact version for every dependency, and the per-node CPU and memory specifications are recorded in the accompanying environment manifest.

## Model Versions and Endpoints

Every open-weight base checkpoint is pinned to an exact Hugging Face commit revision (TableÂ <a href="#tab:modelversions" data-reference-type="ref" data-reference="tab:modelversions">[tab:modelversions]</a>), and the released configuration records these revisions so that fine-tuning and inference reproduce the same weights. The proprietary baselines and four of the six LLM judges are Azure OpenAI deployments, queried with API version `2024-08-01-preview` at temperatureÂ $0$ during JuneÂ 2026. The generator deployments resolved to the dated snapshots `gpt-4.1-2025-04-14` and `gpt-5.4-2026-03-05`, and these two models also serve as judges; the other two Azure judges resolved to `gpt-5.2-2025-12-11` (<span class="smallcaps">GPT-5.2</span>) and the deployed model version `DeepSeek-V3.2`. The remaining two judges are open-weight and run locally at $4$-bit precision like the open-weight generators: <span class="smallcaps">Gemma-2-27B</span> (`google/gemma-2-27b-it`, revision `aaf20e6b`) and <span class="smallcaps">Llama-3.3-70B</span> (`meta-llama/Llama-3.3-70B-Instruct`, revision `6f6073b4`).

<div class="table*">

| **Model**     | **Hugging Face checkpoint**        | **Revision** |
|:--------------|:-----------------------------------|:-------------|
| qwen-3b       | Qwen/Qwen2.5-3B-Instruct           | `aa8e7253`   |
| phi3          | microsoft/Phi-3.5-mini-instruct    | `2fe19245`   |
| qwen-coder-7b | Qwen/Qwen2.5-Coder-7B-Instruct     | `c03e6d35`   |
| mistral-7b    | mistralai/Mistral-7B-Instruct-v0.3 | `c170c708`   |

</div>

## Data Splits and Seeds

The gold dataset is partitioned into stratified training/validation/test sets in a $70/10/20$ ratio, stratified on formula structure (single- versus multi-reading items). The split seed is decoupled from the training seed and fixed at $42$ for the canonical split used for all headline numbers; each fine-tuned configuration is then repeated over three training seeds ($42$, $43$, $44$), and accuracy is reported as the seed mean with a $95\%$ confidence interval. These intervals quantify sensitivity to the training seed at the *fixed* canonical test split and therefore do not capture test-set sampling variance; for a binomial proportion at the observed accuracies, that component is on the order of $\pm0.06$ over the $218$ test items, so the seed intervals understate total uncertainty. The orchestrator additionally supports stratified $k$-fold cross-validation with shared folds, which estimates split-induced variance directly; we report the canonical single-split numbers as the headline and leave a full $k$-fold sweep across all configurations (which multiplies fine-tuning cost by the number of folds) to future runs. The curated few-shot exemplars are held out of every split so that no prompting example can leak into evaluation. Decoding is deterministic (greedy, with sampling disabled) and emits up to $256$ new tokens for local models; Azure calls are issued at temperatureÂ $0$. Only the training split is augmented: each training instance is duplicated once with a templated paraphrase of its natural-language input - a single synonym substitution drawn from a fixed list of temporal and strategic phrasings (for example, â€œsooner or laterâ€â€†$\rightarrow$â€†â€œeventuallyâ€ or â€œcan guarantee thatâ€â€†$\rightarrow$â€†â€œcan ensure thatâ€) - while its gold formula is left unchanged. With an augmentation factor of two this doubles the training data (the original instance plus one paraphrase); the validation and test splits are used verbatim, so augmentation cannot introduce train/test leakage.

## Training and LoRA Hyperparameters

All open-weight models are fine-tuned with low-rank adapters (LoRA) for $8$ epochs using the paged $8$-bit AdamW optimizer, a cosine schedule with a $0.1$ warmup ratio, peak learning rate $1\times10^{-4}$, weight decay $0.01$, gradient clipping at $0.3$, `bf16` precision, and gradient checkpointing; the maximum sequence length is $1536$ tokens. TableÂ <a href="#tab:hparams" data-reference-type="ref" data-reference="tab:hparams">[tab:hparams]</a> lists the per-model adapter and batching settings. LoRA adapters are applied to the attention projections and the MLP projections of each architecture, and every base checkpoint is pinned to an exact revision in the released configuration. The proprietary baselines (<span class="smallcaps">gpt-4.1</span>, <span class="smallcaps">gpt-5.4</span>) are API-only and are not fine-tuned: their weights are closed, and the modest in-domain data available is insufficient to fine-tune models at their scale reliably, so they are evaluated zero- and few-shot.

<div class="table*">

| **Model** | **4-bit** | **LoRA** $r$ | **LoRA** $\alpha$ | **Dropout** | **Batch**$\,\times\,$**Acc.** |
|:---|:--:|:--:|:--:|:--:|:--:|
| mistral-7b | yes | 32 | 64 | 0.05 | $2\times16$ |
| qwen-3b | no | 64 | 128 | 0.05 | $8\times4$ |
| phi3 | no | 32 | 64 | 0.05 | $6\times6$ |
| qwen-coder-7b | yes | 64 | 128 | 0.05 | $4\times8$ |

</div>

# Additional Quantitative Results

This section reports the robustness check and fine-grained breakdowns referenced from the experiments in the main paper: the conservative six-judge accuracy (TableÂ <a href="#tab:accuracy_sixjudge" data-reference-type="ref" data-reference="tab:accuracy_sixjudge">[tab:accuracy_sixjudge]</a>), the decomposition of judged accuracy into its exact-match floor and judge-recovered fraction (FigureÂ <a href="#fig:decomposition" data-reference-type="ref" data-reference="fig:decomposition">9</a>), accuracy split by ambiguity type (TableÂ <a href="#tab:qsa" data-reference-type="ref" data-reference="tab:qsa">1</a>), and the accuracyâ€“latency trade-off (FigureÂ <a href="#fig:accuracy_cost_tradeoff" data-reference-type="ref" data-reference="fig:accuracy_cost_tradeoff">10</a>).

| Model | Size | Baseline ZS | Baseline FS | Fine-tuned ZS | Fine-tuned FS |
|:--|:--:|--:|--:|--:|--:|
| gpt-4.1 | API | 0.438 | 0.623 | -- | -- |
| gpt-5.4 | API | **0.445** | **0.681** | -- | -- |
| mistral-7b | 7B | 0.115 | 0.223 | 0.629 +/- 0.04 | 0.665 +/- 0.05 |
| qwen-3b | 3B | 0.070 | 0.265 | 0.667 +/- 0.04 | 0.692 +/- 0.02 |
| phi3 | 3.8B | 0.147 | 0.293 | 0.707 +/- 0.02 | 0.728 +/- 0.02 |
| qwen-coder-7b | 7B | 0.184 | 0.356 | **0.716 +/- 0.02** | **0.731 +/- 0.02** |

<figure id="fig:decomposition" data-latex-placement="t">
<img src="decomposition.png" style="width:100.0%" />
<figcaption><strong>Accuracy decomposition (Llama-3.3-70B judge).</strong> Semantic accuracy under the headline judge split into the deterministic exact-match floor (blue) and the additional fraction recovered by the LLM judge (orange), for each headline system (proprietary few-shot baselines; open-weight fine-tuned few-shot). The judge-recovered share (annotated) is largest for the proprietary baselines (up to <span class="math inline">46%</span> for <span class="smallcaps">gpt-4.1</span> few-shot) and smaller for the fine-tuned open-weight systems (<span class="math inline">23</span>â€“<span class="math inline">29%</span>), whose outputs more often match the reference surface form. Exact match alone would substantially understate the quality of every system.</figcaption>
</figure>

|  |  |  |
|:---|:--:|:--:|
| **System** | **Single-reading** | **QSA** |
|  | ($n{=}187$) | ($n{=}31$) |
| <span class="smallcaps">gpt-4.1</span> (fs) | 0.88 | 0.58 |
| <span class="smallcaps">gpt-5.4</span> (fs) | 0.88 | 0.68 |
| <span class="smallcaps">mistral-7b</span> (ft+fs) | 0.85 | 0.27 |
| <span class="smallcaps">qwen-3b</span> (ft+fs) | 0.88 | 0.33 |
| <span class="smallcaps">phi3</span> (ft+fs) | 0.87 | 0.49 |
| <span class="smallcaps">qwen-coder-7b</span> (ft+fs) | 0.88 | 0.63 |

Accuracy on the $187$ single-reading versus the $31$ multi-reading (quantifier-scope ambiguity, QSA) test items, under the headline Llama-3.3-70B judge, for the headline systems (proprietary few-shot baselines; fine-tuned few-shot open-weight models; ft+fsâ€†=â€†fine-tunedâ€†+â€†few-shot). Multi-reading items are the rare, hardest cases that require emitting both the distributive and the collective reading. Single-reading accuracy is uniformly high ($0.85$â€“$0.88$), but every system drops sharply on the QSA slice, where no model class dominates: the strongest few-shot proprietary baseline (<span class="smallcaps">gpt-5.4</span>, $0.68$) and the best fine-tuned open-weight model (<span class="smallcaps">qwen-coder-7b</span>, $0.63$) handle it comparably, while the smaller fine-tuned models lag ($0.27$â€“$0.33$). The QSA slice has only $31$ items (a binomial $95\%$ confidence interval is $\approx\pm0.17$), so it supports this qualitative contrast but not a fine ranking. Under the strictest judges the proprietary baselines instead score near zero on this slice; that collapse is an artifact of those judges rejecting their input-grounded predicate paraphrases (our judge-reliability analysis in the main paper), not a failure to emit both readings. This breakdown reuses the existing judged verdicts; no additional labeling was performed. {#tab:qsa}

<figure id="fig:accuracy_cost_tradeoff" data-latex-placement="t">
<img src="accuracyvslatency.png" style="width:100.0%" />
<figcaption>Semantic accuracy (under the headline Llama-3.3-70B judge) versus mean inference latency. Open-weight models are circles, proprietary API baselines diamonds; the dashed line is the Pareto frontier, which under the human-aligned judge is <em>shared</em>: the few-shot proprietary baselines (<span class="smallcaps">gpt-5.4</span>, <span class="smallcaps">gpt-4.1</span>) occupy the high-accuracy end and the fast fine-tuned open-weight systems (<span class="smallcaps">phi3</span>, <span class="smallcaps">qwen-3b</span>) the low-latency end, while the most accurate open-weight system (<span class="smallcaps">qwen-coder-7b</span>) sits just off the frontier, dominated by <span class="smallcaps">gpt-5.4</span>. API latency is network-timed and only indicative across the API/local boundary; accuracy is comparable throughout.</figcaption>
</figure>

# Prompting and Few-Shot Exemplars

All systems share a fixed prompting setup, identical across models and prompting conditions; the verbatim templates ship with the released code. We summarize their content here.

## Translation Prompt

The system prompt casts the model as an ATL/ATL$^\ast$ expert and requires it to return *only* formula text. It fixes:

- **Syntax.** The coalition modality `<<A>>` or `<<A,B>>`; the temporal operators `X`, `F`, `G` (unary) and `U` (binary, written `p U q`); the Boolean operators `!`, `&&`, `||`, `->`; <span class="smallcaps">PascalCase</span> agent and coalition names; and `snake_case` atomic propositions.

- **Scope.** The strategic operator scopes over the whole formula that follows it and is kept separate from the temporal operator it governs, e.g.Â `<<Machine>>G(paid -> ticket_printed)`, not `<<Machine>>(G paid -> ticket_printed)`. Inability is expressed by negating the strategic operator (`!<<Y>>F goal`), not the objective.

- **Targeted ambiguities.** VP ellipsis: repeat the recovered formula for the second agent. Right Node Raising: attach the shared right-peripheral objective to both conjuncts. Quantifier-scope ambiguity: output *all* admissible readings, one per line, never fusing two readings into one formula; a distributive reading ascribes the ability to each agent separately and a collective reading to the coalition jointly.

- **Output discipline.** Emit only the formula(s), one per line, with no explanations, Markdown, labels, or trailing prose.

## Few-Shot Exemplars

In the few-shot condition a fixed pool of seven curated exemplars is prepended to the prompt; these inputs are held out of every train/validation/test split, so they never leak into evaluation. Each exemplar targets a distinct phenomenon:

- **Collective ability (right dislocation).** â€œThey can guarantee that at the next step the alarm will be sent, the surveillance system and the operator.â€\
  $\Rightarrow$ `<<System,Operator>>X alarm_sent`

- **Left dislocation.** â€œThe gate, the machine can guarantee that it will open at the next step.â€\
  $\Rightarrow$ `<<Machine>>X gate_open`

- **VP ellipsis.** â€œRobot number 1 has a strategy to ensure that eventually position 3 holds, and robot number 2 does too.â€\
  $\Rightarrow$ `<<Robot1>>F pos3 && <<Robot2>>F pos3`

- **Quantifier-scope ambiguity (two required readings).** â€œEvery robot can guarantee that it will eventually reach a safe spot.â€\
  $\Rightarrow$ `<<Robot1>>F at_safe_spot_1 && <<Robot2>>F at_safe_spot_2 && <<Robot3>>F at_safe_spot_3`\
  and `<<Robot1,Robot2,Robot3>>F at_safe_spot`

- **Ability asymmetry.** â€œThe diplomatic cable system can, but the encryption gateway cannot, guarantee that classified cables will never be routed publicly.â€\
  $\Rightarrow$ `<<DiplomaticCableSystem>>G !classified_cables_routed_publicly && !<<EncryptionGateway>>G !classified_cables_routed_publicly`

- **Simple eventuality.** â€œThe user can guarantee that sooner or later the ticket will be printed.â€\
  $\Rightarrow$ `<<User>>F ticket_printed`

- **Nested, literary input.** â€œIf we do not wish to fight, we can prevent the enemy from engaging usâ€¦â€ (adapted from Sun Tzu).\
  $\Rightarrow$ `<<We>>(!wish_to_fight -> F (throw_something_odd_in_his_way && G !enemy_engages_us))`

With the count left unset, every exemplar is shown in a fixed order; a smaller count selects a reproducible random subset and is used only for ablations.

## LLM-Judge Prompt

All six judges (DeepSeek-V3.2, GPT-4.1, GPT-5.2, GPT-5.4, Gemma-2-27B, and Llama-3.3-70B) use a single fixed prompt (versionÂ v1.4) that casts the model as an adjudicator of ATL/ATL$^\ast$ *faithfulness* beyond exact string matching. It restates the syntax conventions above and applies an explicit rubric.

- **Accept:** harmless whitespace and redundant parentheses; commutative reordering of `&&` or `||` when the same operands stay under the same strategic, temporal, and Boolean scope; and renamed predicates or agents only when they are clear aliases grounded in the input.

- **Reject:** algebraic rewrites (e.g., `p -> q` as `!p || q`), contraposition, De Morgan, double-negation, idempotence, distributivity, or biconditional rewrites; any change in a temporal operator or its replacement by a temporal-logic equivalent (e.g.Â `F p` as `true U p`, `G p` as `!F !p`); any change of coalition, including distributive versus collective ability; any change of temporal or strategic scope, implication direction, or polarity; turning a conjunction into a disjunction or vice versa; and any omitted or extraneous condition. For ambiguous items it rejects predictions that return only one of the jointly required readings or that collapse them into a single conjunction.

- **Output and robustness.** The judge returns a single machine-parseable JSON object, `{"correct": "yes" | "no", "reasoning": "..."}`, and is calibrated with eight worked accept/reject examples (covering parenthesization, commutativity, operator rewrites, QSA multiplicity, and temporal-operator and coalition errors). To resist prompt injection, it is instructed to treat the input, gold output(s), and prediction as data and to ignore any instructions embedded inside them.

Verdicts are cached per judge identity, so a judge is never queried twice on an identical (input, gold, prediction) triple, while the judges remain independent.

# Human-Annotation Disagreements and Their Adjudication

Of the $599$ audited predictions, the two expert annotators initially diverged on three. In all three the stricter annotator judged the prediction incorrect while the second judged it correct. The annotators then deliberated and reached agreement on two of the three (D2 and D3 below), folding those labels back into the reference set; the third (D1) resisted consensus and is retained here as a genuine unresolved disagreement rather than forced to a label. Both reconciled cases were settled in favour of the *permissive* reading (the annotators agreed the prediction *was* faithful) even though the automated judges had almost all called these two predictions incorrect (unanimously across the six judges for D2, and four-of-six for D3, where <span class="smallcaps">gpt-4.1</span> and the human-aligned <span class="smallcaps">Llama-3.3-70B</span> had sided with the permissive annotator). On these borderline items the human audit therefore overturns an over-strict tendency the judges share, rather than merely confirming them; even the most human-aligned judge, <span class="smallcaps">Llama-3.3-70B</span>, shared this over-strict reading on D2, though it had already sided with the experts on D3. The case left open, D1, is by contrast the item on which all six judges agree the prediction is incorrect, yet the annotators could not settle whether an unresolved pronominal coalition and a predicate that absorbs the agent name still count as a faithful rendering. Each entry below gives the natural-language input, the gold formula, the model prediction, the verdicts (with any annotator note), and the adjudication outcome. With the two reconciled labels added and the single open case excluded, the human-as-reference comparison rests on $598$ consensus labels.

- **D1. Pronominal coalition vs.Â bundled predicate** (<span class="smallcaps">gpt-5.4</span>, zero-shot baseline; item `ex677`).\
  *Input:* â€œIt can guarantee that invalid badges will never open the staff entrance, the badge reader.â€\
  *Gold:*\
  `<<BadgeReader>>G`\
  `!invalid_badges_open_staff_entrance`\
  *Prediction:*\
  `<<It>>G (!invalid_badges_opens)`\
  *AnnotatorÂ 1* (incorrect): â€œprediction uses `<<It>>` instead of `<<BadgeReader>>` and bundles `badge_reader` into the predicate.â€\
  *AnnotatorÂ 2*: correct (no note).*All six judges*: incorrect.\
  The dispute is whether an unresolved pronoun coalition and a proposition that swallows the agent name still count as a faithful rendering. *Adjudication*: **unresolved** - the annotators could not converge after discussion, so this item is kept as a genuine disagreement and excluded from the human-as-reference set.

- **D2. Reversed *until* operands** (<span class="smallcaps">qwen-coder-7b</span>, fine-tuned, zero-shot; item `ex992`).\
  *Input:* â€œThe help desk can guarantee that â€¦the request will remain pending until the missing detail is supplied, the account will remain limited until identity is checked, and the case will remain open until the resolution is confirmed.â€\
  The prediction shares the goldâ€™s prefix `<<HelpDesk>>(X user_notified && G ticket_traceable && â€¦)` and differs only in the three `U` (until) clauses:\
  *Gold:* `(request_pending U missing_detail_supplied) && (account_limited U identity_checked) && (case_open U resolution_confirmed)`\
  *Prediction:* `(missing_detail_supplied U request_resolved) && (identity_checked U account_limited) && (resolution_confirmed U case_open)`\
  *AnnotatorÂ 1* (incorrect): â€œscoping error: `U` operands are wrong.â€*AnnotatorÂ 2*: correct (no note).*All six judges*: incorrect.\
  Every `U` has its two operands transposed (and one predicate altered), which annotatorÂ 1 read as inverting the temporal commitment; the dispute is how strictly operand order must be enforced. *Adjudication*: after discussion the annotators **agreed the prediction is faithful** (*correct*), overturning the stricter initial label and the six judgesâ€™ unanimous incorrect verdict.

- **D3. Predicate abbreviation and line-separated conjuncts** (<span class="smallcaps">mistral-7b</span>, few-shot baseline; item `ex462`).\
  *Input:* â€œThe pharmacy interface can guarantee that the dispense request remains deferred until the interaction screen is cleared, and the medication safety service can too.â€\
  *Gold:* `<<PharmacyInterface>>`\
  `(dispense_request_deferred U`\
  `interaction_screen_cleared)`\
  `&& <<MedicationSafetyService>>`\
  `(dispense_request_deferred U`\
  `interaction_screen_cleared)`\
  *Prediction:* the same two coalitions with `dispense_request_deferred` abbreviated to `request_deferred` and emitted on two separate lines rather than joined by `&&`.\
  *AnnotatorÂ 1*: incorrect (no note).*AnnotatorÂ 2*: correct (no note).*Judges*: incorrect, except <span class="smallcaps">gpt-4.1</span> and the human-aligned <span class="smallcaps">Llama-3.3-70B</span> (correct).\
  The dispute is whether an abbreviated proposition and an implicit (line-break) conjunction preserve the intended meaning. *Adjudication*: after discussion the annotators **agreed the prediction is faithful** (*correct*); the abbreviation and line-break conjunction were accepted as meaning-preserving, a call that, among the judges, only <span class="smallcaps">gpt-4.1</span> and the human-aligned <span class="smallcaps">Llama-3.3-70B</span> had made.

These cases show that the â€œcorrectnessâ€ of a strategicâ€“temporal translation is not always single-valued at the margins: predicate aliasing, implicit coreference, operand order, and the syntactic marking of conjunction all sit on a boundary where expert judgment legitimately differs. Deliberation resolved two of the three, but in both the experts ultimately accepted a prediction that the LLM judges had rejected, and one case remains genuinely open. This is precisely what motivates our two-tier protocol (reporting the deterministic exact-match floor separately, using six independent judges, and validating them against a human audit) rather than treating any single verdict as ground truth.

## References

1. Fodor, J. D.; and Sag, I. A. 1982. Referential and Quantificational Indefinites. *Linguistics and Philosophy*, 5(3): 355â€“398.
2. Reinhart, T. 1997. Quantifier Scope: How Labor Is Divided between QR and Choice Functions. *Linguistics and Philosophy*, 20(4): 335â€“397.
3. Rizzi, L. 1997. The Fine Structure of the Left Periphery. In Haegeman, L., ed., *Elements of Grammar*, 281â€“337. Dordrecht: Kluwer.

