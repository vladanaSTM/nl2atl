# ATL Primer: Alternating-Time Temporal Logic

This guide introduces Alternating-Time Temporal Logic (ATL), the formal specification language used by NL2ATL. You'll learn ATL syntax, semantics, and how natural language maps to ATL formulas.

## Table of Contents

1. [What is ATL?](#what-is-atl)
2. [Why ATL for Multi-Agent Systems?](#why-atl-for-multi-agent-systems)
3. [ATL Syntax](#atl-syntax)
4. [ATL Semantics](#atl-semantics)
5. [Natural Language to ATL Patterns](#natural-language-to-atl-patterns)
6. [Complete Examples](#complete-examples)
7. [Common Mistakes](#common-mistakes)
8. [ATL vs ATL*](#atl-vs-atl)
9. [Practice Exercises](#practice-exercises)

---

## What is ATL?

**Alternating-Time Temporal Logic (ATL)** is a formal logic for reasoning about what coalitions of agents can achieve in multi-agent systems. Unlike standard temporal logics (like LTL or CTL) that describe what *will* happen, ATL describes what agents *can strategically force* to happen.

### Key Idea

ATL formulas express statements like:

> "Agents A and B, working together, have a strategy to ensure that the goal is eventually reached, regardless of what other agents do."

This is written in ATL as: `<<A,B>>F goal`

### Applications

- **Autonomous systems**: Robot coordination, self-driving cars
- **Security protocols**: Multi-party authentication, Byzantine agreement
- **Game theory**: Equilibrium analysis, coalition formation
- **Distributed systems**: Consensus protocols, fault tolerance
- **Smart contracts**: Multi-signature wallets, decentralized governance

---

## Why ATL for Multi-Agent Systems?

### The Problem

In multi-agent systems, multiple agents interact with potentially conflicting goals. We need to specify properties like:

- "Can the user ensure data is eventually saved?"
- "Can malicious agents prevent the system from reaching a safe state?"
- "Can a coalition of robots coordinate to complete a task?"

Standard temporal logics (LTL, CTL) can't express these strategic capabilities because they don't distinguish between agents.

### The ATL Solution

ATL adds **coalition modalities** that explicitly name which agents have control:

| Logic | What it expresses | Example |
|-------|-------------------|---------|
| **LTL** | "It will eventually happen" | `F goal` |
| **CTL** | "There exists a path where it happens" | `EF goal` |
| **ATL** | "Agents A,B can force it to happen" | `<<A,B>>F goal` |

ATL makes agent capabilities explicit and verifiable.

---

## ATL Syntax

### Coalition Modality

The core of ATL is the coalition modality: `<<A,B,...>>`

**Syntax**: `<<Agent1,Agent2,...>>`

**Reading**: "Agents Agent1, Agent2, ... have a joint strategy to ensure..."

**Examples**:
- `<<User>>` — The User agent alone
- `<<Robot1,Robot2>>` — Robots 1 and 2 cooperating
- `<<Controller,Monitor>>` — Controller and Monitor working together
- `<<>>` — The empty coalition (equivalent to "no matter what all agents do")

### Temporal Operators

ATL uses standard temporal operators:

| Operator | Name | Meaning |
|----------|------|---------|
| **X φ** | Next | φ holds in the next state |
| **F φ** | Eventually (Future) | φ holds at some future state |
| **G φ** | Always (Globally) | φ holds at all future states (including now) |
| **φ U ψ** | Until | φ holds until ψ becomes true |

### Logical Operators

| Operator | Symbol | Meaning |
|----------|--------|---------|
| Negation | `!φ` | Not φ |
| Conjunction | `φ && ψ` | φ and ψ |
| Disjunction | `φ || ψ` | φ or ψ |
| Implication | `φ -> ψ` | If φ then ψ |

### Complete Formula Structure

```
<<Coalition>> TemporalOp LogicalFormula
```

**Examples**:
- `<<A>>F p` — Agent A can eventually make p true
- `<<A>>G !error` — Agent A can always avoid errors
- `<<A,B>>X (p && q)` — A and B can make both p and q true next
- `<<A>>(safe U goal)` — A can keep safe true until goal becomes true

---

## ATL Semantics

### Concurrent Game Structures

ATL is interpreted over **Concurrent Game Structures (CGS)**:

- **States**: Possible system configurations
- **Agents**: Players in the game
- **Actions**: Moves available to each agent at each state
- **Transition function**: Given current state and all agents' actions, determines next state
- **Atomic propositions**: Properties that are true/false at each state

### Coalition Strategies

A **strategy** for a coalition is a plan that assigns actions to coalition members at every state based on the history.

A formula `<<A>>φ` is true if coalition A has a strategy ensuring φ holds **no matter what** the other agents do.

### Informal Reading Rules

Let's build intuition with examples:

**`<<A>>X p`**  
"Agent A can ensure that p holds in the next state"  
→ A has an action that makes p true next, regardless of what others do

**`<<A>>F p`**  
"Agent A can ensure that p eventually holds"  
→ A has a strategy (possibly taking many steps) to reach a state where p is true

**`<<A>>G p`**  
"Agent A can ensure that p always holds"  
→ A has a strategy to maintain p true forever, no matter the opposition

**`<<A>>(p U q)`**  
"Agent A can ensure that p holds continuously until q becomes true"  
→ A has a strategy to keep p true and eventually make q true

### Key Principle: Adversarial Environment

ATL assumes **adversarial completion**: agents outside the coalition act to make the formula false if possible. The coalition must succeed **despite** worst-case behavior from others.

---

## Natural Language to ATL Patterns

This section shows common natural language patterns and their ATL translations.

### Pattern 1: Agent Capability

**NL Cues**: "can ensure", "can guarantee", "has the power to"

| Natural Language | ATL | Notes |
|------------------|-----|-------|
| "User can ensure..." | `<<User>>` | Single agent |
| "Agents A and B can jointly ensure..." | `<<A,B>>` | Coalition |
| "The system can guarantee..." | `<<System>>` | System as agent |

### Pattern 2: Temporal Properties

#### Eventually (Liveness)

**NL Cues**: "eventually", "sooner or later", "at some point", "will reach"

| Natural Language | ATL | Notes |
|------------------|-----|-------|
| "Eventually goal is reached" | `F goal` | (No agent specified) |
| "User can eventually reach goal" | `<<User>>F goal` | User ensures it |
| "Sooner or later ticket is printed" | `F ticket_printed` | Must happen sometime |

#### Always (Safety)

**NL Cues**: "always", "forever", "never", "at all times", "invariably"

| Natural Language | ATL | Notes |
|------------------|-----|-------|
| "Always safe" | `G safe` | Safety property |
| "Never error" | `G !error` | Negation + always |
| "Controller always avoids crash" | `<<Controller>>G !crash` | Safety guarantee |

#### Next

**NL Cues**: "next step", "immediately after", "in the next state"

| Natural Language | ATL | Notes |
|------------------|-----|-------|
| "Next state is safe" | `X safe` | One step |
| "Immediately after request, acknowledge" | `request -> X ack` | Next after condition |

#### Until

**NL Cues**: "until", "while waiting for"

| Natural Language | ATL | Notes |
|------------------|-----|-------|
| "Safe until goal" | `safe U goal` | Keep safe, then goal |
| "Wait until signal" | `wait U signal` | Maintain condition |

### Pattern 3: Conditional Properties

**NL Cues**: "if...then...", "whenever", "after"

| Natural Language | ATL | Notes |
|------------------|-----|-------|
| "If request, then eventually grant" | `request -> F grant` | Response property |
| "Whenever alarm, next state is safe" | `G (alarm -> X safe)` | Always respond |
| "After payment, ticket is printed" | `G (paid -> F ticket_printed)` | Fairness |

### Pattern 4: Coalition Properties

**NL Cues**: "together", "jointly", "cooperating", "both...and..."

| Natural Language | ATL | Notes |
|------------------|-----|-------|
| "A and B together can..." | `<<A,B>>` | Two-agent coalition |
| "All agents jointly ensure..." | `<<Agent1,Agent2,...>>` | Large coalition |
| "Either A alone or B alone can..." | `<<A>>φ || <<B>>φ` | Disjunction |

### Pattern 5: Negation Patterns

**NL Cues**: "avoid", "prevent", "never"

| Natural Language | ATL | Notes |
|------------------|-----|-------|
| "Avoid error" | `G !error` | Never error |
| "Prevent crash" | `<<Agent>>G !crash` | Agent prevents |
| "Never deadlock" | `G !deadlock` | Safety |
| "Eventually not stuck" | `F !stuck` | Escape condition |

---

## Complete Examples

### Example 1: Ticket Machine

**Natural Language**:
> "The user can guarantee that if payment is completed, then sooner or later the ticket will be printed."

**ATL Formula**:
```
<<User>>G (paid -> F ticket_printed)
```

**Explanation**:
- `<<User>>` — User has control
- `G` — Always (at all times)
- `paid -> F ticket_printed` — If paid, then eventually ticket_printed
- Combined: User ensures that every payment eventually results in a ticket

**Why this works**: The User agent can choose actions that, once payment occurs, lead inevitably to printing the ticket.

---

### Example 2: Robot Navigation

**Natural Language**:
> "Robots A and B together can guarantee that they eventually reach the goal while always avoiding obstacles."

**ATL Formula**:
```
<<A,B>>((!obstacle) U goal)
```

**Explanation**:
- `<<A,B>>` — Both robots cooperating
- `(!obstacle) U goal` — Not obstacle holds until goal is reached
- Combined: A and B can navigate to goal while staying clear of obstacles

**Alternative (stronger)**:
```
<<A,B>>F (goal && G !obstacle)
```
This says: reach goal and stay safe forever after.

---

### Example 3: Access Control

**Natural Language**:
> "The system can guarantee that access is granted if and only if authentication succeeds."

**ATL Formula**:
```
<<System>>G (access <-> authenticated)
```

**Explanation**:
- `<<System>>` — System controls access
- `G` — Always maintain this property
- `access <-> authenticated` — Access granted iff authenticated
- Combined: System enforces access control invariant

**Note**: `<->` (iff) is `(access -> authenticated) && (authenticated -> access)`

---

### Example 4: Multi-Agent Coordination

**Natural Language**:
> "Agent A can ensure that if agent B cooperates, then they can jointly reach the goal."

**ATL Formula**:
```
<<A>>(cooperate_B -> <<A,B>>F goal)
```

**Explanation**:
- `<<A>>` — A ensures the implication
- `cooperate_B -> <<A,B>>F goal` — If B cooperates, then A+B can reach goal
- This is a **nested modality** (ATL* feature)

**Simplified ATL version** (if B's cooperation is observable):
```
<<A>>G (cooperate_B -> F goal)
```

---

### Example 5: Fault Tolerance

**Natural Language**:
> "The controller can guarantee that even if one sensor fails, the system remains safe."

**ATL Formula**:
```
<<Controller>>G (sensor_fail -> safe)
```

**Explanation**:
- `<<Controller>>` — Controller handles faults
- `G (sensor_fail -> safe)` — Always: if sensor fails, system stays safe
- Combined: Controller can tolerate single sensor failure

---

### Example 6: Request-Response

**Natural Language**:
> "The server can guarantee that every request is eventually granted and the grant happens within 3 time steps."

**ATL Formula** (bounded response):
```
<<Server>>G (request -> (F grant && X (grant || X (grant || X grant))))
```

**Simplified** (if we don't enforce 3-step bound):
```
<<Server>>G (request -> F grant)
```

**Explanation**:
- For each request, server eventually grants
- This is a **fairness** property

---

## Common Mistakes

### Mistake 1: Forgetting the Coalition

**Wrong**: `F goal`  
**Right**: `<<Agent>>F goal`

**Why**: Without `<<Agent>>`, the formula doesn't specify who can ensure the property. Plain `F goal` is LTL, not ATL.

### Mistake 2: Confusing "Always Eventually" Order

**Wrong**: `<<A>>F G p` (invalid ATL)  
**Right**: `<<A>>G F p`

**Why**: ATL restricts one temporal operator per coalition modality. `G F p` means "always eventually p" (infinitely often).

### Mistake 3: Misplacing Negation

**Ambiguous NL**: "User can avoid the error"

**Wrong**: `<<User>>! error` (syntax error)  
**Right**: `<<User>>G !error`

**Why**: Negation applies to propositions, not modalities.

### Mistake 4: Coalition vs Logical Operators

**Wrong**: `<<A>> && <<B>> F goal`  
**Right**: `<<A,B>>F goal` (both cooperate) or `<<A>>F goal && <<B>>F goal` (each can separately)

**Why**: Coalition modality takes a list of agents, not a logical combination.

### Mistake 5: Until Without Eventuality

**Ambiguous NL**: "Safe until goal"

**Weak**: `<<A>>(safe U goal)` — Requires goal to eventually happen  
**Strong**: `<<A>>G safe || <<A>>(safe U goal)` — Either always safe OR safe until goal

**Why**: `U` requires the second argument to eventually hold. If goal may never occur, use `W` (weak until) if supported, or the disjunction above.

---

## ATL vs ATL*

### ATL (Alternating-Time Temporal Logic)

**Restriction**: Temporal operators must immediately follow coalition modality.

**Valid ATL**:
- `<<A>>F p`
- `<<A>>G (p -> F q)`
- `<<A>>(p U q)`

**Invalid ATL**:
- `<<A>>F G p` — Two temporal operators
- `F <<A>>G p` — Temporal before coalition
- `<<A>>F p && <<B>>G q` — Valid ATL (each is separate formula)

### ATL* (ATL with nested temporal operators)

**Freedom**: Arbitrary nesting of temporal and coalition operators.

**Valid ATL***:
- `<<A>>F G p` — Eventually always
- `<<A>>G F p` — Always eventually (infinitely often)
- `F <<A>>G p` — Eventually, A can maintain p forever
- `<<A>>(F p && G q)` — Eventually p AND always q

**NL2ATL's Target**: ATL-style formulas with `<<...>>` coalition syntax. Most patterns in natural language map to ATL, not ATL*.

---

## Practice Exercises

### Exercise 1

**Natural Language**:  
"The user can eventually log out."

**Your ATL**:  
`_______________________`

<details>
<summary>Answer</summary>

`<<User>>F logout`

**Explanation**: User has control, eventually logout happens.
</details>

---

### Exercise 2

**Natural Language**:  
"The controller can always keep the temperature below 100 degrees."

**Your ATL**:  
`_______________________`

<details>
<summary>Answer</summary>

`<<Controller>>G (temp < 100)`

**Explanation**: Controller maintains invariant forever.
</details>

---

### Exercise 3

**Natural Language**:  
"Agents A and B together can ensure that if a request is made, it is acknowledged in the next step."

**Your ATL**:  
`_______________________`

<details>
<summary>Answer</summary>

`<<A,B>>G (request -> X ack)`

**Explanation**: Coalition A,B ensures at all times: if request, then next state has acknowledgment.
</details>

---

### Exercise 4

**Natural Language**:  
"The system can guarantee it never enters an error state."

**Your ATL**:  
`_______________________`

<details>
<summary>Answer</summary>

`<<System>>G !error`

**Explanation**: System always avoids error.
</details>

---

### Exercise 5

**Natural Language**:  
"Robot can keep moving until it reaches the charging station."

**Your ATL**:  
`_______________________`

<details>
<summary>Answer</summary>

`<<Robot>>(moving U charging_station)`

**Explanation**: Robot maintains moving true until charging_station becomes true.
</details>

---

## Summary

### Key Takeaways

1. **ATL is about strategic capability**: What coalitions *can force* to happen
2. **Coalition modality** `<<A,B>>`: Specifies who has control
3. **Temporal operators** `G, F, X, U`: Specify when properties hold
4. **Adversarial semantics**: Coalition succeeds despite worst-case opponents
5. **ATL ≠ LTL**: ATL makes agent control explicit

### Quick Reference

| Component | Syntax | Meaning |
|-----------|--------|---------|
| Coalition | `<<A,B>>` | Agents A, B cooperate |
| Next | `X φ` | φ in next state |
| Eventually | `F φ` | φ at some future state |
| Always | `G φ` | φ at all future states |
| Until | `φ U ψ` | φ until ψ |
| Not | `!φ` | Negation |
| And | `φ && ψ` | Conjunction |
| Or | `φ || ψ` | Disjunction |
| Implies | `φ -> ψ` | Implication |

### Translation Heuristics

1. Identify **who** can ensure → `<<Agent>>`
2. Identify **when** (always/eventually/next/until) → `G/F/X/U`
3. Identify **what** property → atomic proposition or logical formula
4. Combine: `<<who>> when what`

---

## Further Reading

- **Model Checking**: Clarke, E. M., Grumberg, O., & Peled, D. A. (1999). *Model Checking*. MIT Press.
- **ATL Foundation**: Alur, R., Henzinger, T. A., & Kupferman, O. (2002). "Alternating-time temporal logic." *Journal of the ACM*.
- **genVITAMIN Tool**: Multi-agent model checker supporting ATL verification
- **NL2ATL Dataset**: [dataset.md](dataset.md) for real examples

---

## Next Steps

Now that you understand ATL:

1. **Explore the dataset** → [Dataset Guide](dataset.md)
2. **Run experiments** → [Quick Start](quickstart.md)
3. **Understand evaluation** → [Evaluation Guide](evaluation.md)

---

**Questions?** Check the [full documentation](index.md) or [open an issue](https://github.com/vladanaSTM/nl2atl/issues).
