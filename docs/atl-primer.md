# ATL primer for NL2ATL

This document introduces Alternating‑Time Temporal Logic (ATL) as used by NL2ATL. NL2ATL outputs **ATL‑style** formulas with ASCII coalition syntax `<<A,B>>` and temporal operators `G`, `F`, `X`, `U`.

## Why ATL

ATL expresses what **coalitions of agents can guarantee**, regardless of how other agents behave. This is the right formalism for multi‑agent systems with strategic interaction.

In natural language:

- “Agents A and B can ensure the goal eventually holds.”
- “Controller can always avoid error.”

In ATL:

- `<<A,B>>F goal`
- `<<Controller>>G !error`

## Core syntax

### Coalition modality

A coalition is written as `<<A,B>>`. Read it as: *agents A and B have a joint strategy to ensure …*.

### Temporal operators

| Operator | Meaning | Example |
|---|---|---|
| `X` | next | `<<A>>X safe` |
| `F` | eventually | `<<A>>F goal` |
| `G` | always | `<<A>>G !error` |
| `U` | until | `<<A>>(safe U goal)` |

### Logical operators

NL2ATL uses ASCII operators:

- Negation: `!`
- Conjunction: `&&`
- Disjunction: `||`
- Implication: `->`

## Semantics in one page

ATL is interpreted over **Concurrent Game Structures** with agents, states, and joint actions. The coalition modality says that a group of agents has a strategy to force the formula to hold, no matter how the other agents act.

Informal reading rules:

- `<<A>>X p`: A can ensure $p$ holds in the next state.
- `<<A>>F p`: A can ensure $p$ eventually holds.
- `<<A>>G p`: A can ensure $p$ holds forever.
- `<<A>>(p U q)`: A can keep $p$ true until reaching $q$.

## Reading an ATL formula

Formula:

```
<<A,B>>G (request -> F grant)
```

Reading:

> Agents A and B have a joint strategy to always ensure that if a request happens, then a grant eventually follows.

## NL to ATL mapping patterns

| Natural‑language cue | ATL pattern | Example |
|---|---|---|
| “can ensure / can guarantee” | `<<A>>` | “User can ensure success” → `<<User>>` |
| “always / forever” | `G` | “always safe” → `G safe` |
| “eventually / sooner or later” | `F` | “eventually goal” → `F goal` |
| “next step / immediately after” | `X` | “next state safe” → `X safe` |
| “until” | `U` | “safe until goal” → `safe U goal` |
| “if … then …” | `->` | “if request then grant” → `request -> grant` |

## Examples

### Example 1 — safety

Natural language:

> The controller can always avoid errors.

ATL:

```
<<Controller>>G !error
```

### Example 2 — liveness

Natural language:

> Robots A and B can eventually reach the goal.

ATL:

```
<<A,B>>F goal
```

### Example 3 — response

Natural language:

> The server can ensure that every request is eventually granted.

ATL:

```
<<Server>>G (request -> F grant)
```

### Example 4 — until

Natural language:

> The operator can keep the system safe until the recovery state is reached.

ATL:

```
<<Operator>>(safe U recovered)
```

## ATL vs ATL*

ATL restricts temporal operators to appear directly after a coalition modality. ATL* allows arbitrary nesting of temporal operators. NL2ATL targets **ATL‑style** outputs with `<<...>>` coalition syntax.

Valid ATL:

- `<<A>>F p`
- `<<A>>G (p -> F q)`

Not valid ATL but valid ATL*:

- `<<A>>F G p`

## How NL2ATL uses ATL

- Coalitions are extracted from the text and rendered as `<<Agent1,Agent2>>`.
- Temporal phrases map to `G`, `F`, `X`, and `U`.
- Negation and implication use ASCII operators `!` and `->`.

For dataset details and difficulty scoring, see [dataset.md](dataset.md) and
[difficulty_classification.md](difficulty_classification.md).