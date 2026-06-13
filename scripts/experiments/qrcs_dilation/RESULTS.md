# QRCS Experiment 1 — Dilation-Curve Discrimination: RESULTS

Setup: walker on a 12-node cycle GraphBase; T = 10000 global ticks per run; deterministic motion accumulator (no RNG); Float64 (the √ in the Pythagorean rule and the target is irrational, so the exact ring is not applicable); c = 1 hop/tick.

Measured proper-time rate dφ/dt = φ/T; target 1/γ(v) = √(1−v²).

## Self-checks (attribute the result to the rule, not the plumbing)

- Loop mechanics reproduce r(v):  max |φ/T − r(v)| = 2.286e-13  (bound 1e-11: T float additions)  → PASS
- Hop rate matches v:  max |hops/T − v| = 0.0001  (accumulator granularity 1/T = 0.0001)  → PASS

## Rule LINEAR

| v    | measured dφ/dt | target 1/γ | deviation |
|------|----------------|------------|-----------|
| 0.0  | 1.0            | 1.0        | 0.0       |
| 0.1  | 0.9            | 0.994987   | -0.094987 |
| 0.2  | 0.8            | 0.979796   | -0.179796 |
| 0.3  | 0.7            | 0.953939   | -0.253939 |
| 0.4  | 0.6            | 0.916515   | -0.316515 |
| 0.5  | 0.5            | 0.866025   | -0.366025 |
| 0.6  | 0.4            | 0.8        | -0.4      |
| 0.7  | 0.3            | 0.714143   | -0.414143 |
| 0.8  | 0.2            | 0.6        | -0.4      |
| 0.9  | 0.1            | 0.43589    | -0.33589  |
| 0.95 | 0.05           | 0.31225    | -0.26225  |
| 0.99 | 0.01           | 0.141067   | -0.131067 |

Max |deviation| over the sweep: **0.4141428428542492**

## Rule PYTHAGOREAN

| v    | measured dφ/dt | target 1/γ | deviation |
|------|----------------|------------|-----------|
| 0.0  | 1.0            | 1.0        | 0.0       |
| 0.1  | 0.994987       | 0.994987   | 0.0       |
| 0.2  | 0.979796       | 0.979796   | 0.0       |
| 0.3  | 0.953939       | 0.953939   | 0.0       |
| 0.4  | 0.916515       | 0.916515   | 0.0       |
| 0.5  | 0.866025       | 0.866025   | -0.0      |
| 0.6  | 0.8            | 0.8        | 0.0       |
| 0.7  | 0.714143       | 0.714143   | -0.0      |
| 0.8  | 0.6            | 0.6        | 0.0       |
| 0.9  | 0.43589        | 0.43589    | 0.0       |
| 0.95 | 0.31225        | 0.31225    | 0.0       |
| 0.99 | 0.141067       | 0.141067   | -0.0      |

Max |deviation| over the sweep: **2.286e-13**

## Rule QUADRATIC

| v    | measured dφ/dt | target 1/γ | deviation |
|------|----------------|------------|-----------|
| 0.0  | 1.0            | 1.0        | 0.0       |
| 0.1  | 0.99           | 0.994987   | -0.004987 |
| 0.2  | 0.96           | 0.979796   | -0.019796 |
| 0.3  | 0.91           | 0.953939   | -0.043939 |
| 0.4  | 0.84           | 0.916515   | -0.076515 |
| 0.5  | 0.75           | 0.866025   | -0.116025 |
| 0.6  | 0.64           | 0.8        | -0.16     |
| 0.7  | 0.51           | 0.714143   | -0.204143 |
| 0.8  | 0.36           | 0.6        | -0.24     |
| 0.9  | 0.19           | 0.43589    | -0.24589  |
| 0.95 | 0.0975         | 0.31225    | -0.21475  |
| 0.99 | 0.0199         | 0.141067   | -0.121167 |

Max |deviation| over the sweep: **0.245889894354028**

## The three curves against 1/γ (ASCII; '.' = target 1/γ, '@' = PYTHAGOREAN
   sitting on the target, L = LINEAR, Q = QUADRATIC; at v = 0 all curves
   coincide at 1.0 and display as '@')

1.0  |  @    @    @                                               
0.95 |            Q    @                                          
0.9  |       L         Q    @                                     
0.85 |                      Q    @                                
0.8  |            L                   @                           
0.75 |                           Q                                
0.7  |                 L                   @                      
0.65 |                                Q                           
0.6  |                      L                   @                 
0.55 |                                                            
0.5  |                           L         Q                      
0.45 |                                               @            
0.4  |                                L                           
0.35 |                                          Q                 
0.3  |                                     L              @       
0.25 |                                                            
0.2  |                                          L    Q            
0.15 |                                                         @  
0.1  |                                               L    Q       
0.05 |                                                    L       
0.0  |                                                         L  
     +------------------------------------------------------------
      0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  0.95 0.99 
        (v, hops/tick)

Raw curves: curves.csv (v, target, and the three measured rates).

## Pre-registered reading (verbatim, QRCS_Research_Ledger.md §8)

> **Pre-registered reading:** the Pythagorean rule matching γ is
> *consistency-by-construction* (the √ was inserted); the scientific
> content is **discriminative** — the other rules *fail*, showing Lorentz
> dilation uniquely selects the norm-composition law. **Emergence is NOT
> demonstrated by this experiment** and may not be claimed; deriving the
> Pythagorean rule from substrate dynamics is the §2 debt, registered as
> the follow-up. Failure mode pre-stated: if even the Pythagorean rule
> misses γ (discretization effects), that is a finding about the walker
> model, reported as such.

Status of this run against that reading (Ledger §9 sentence shapes):
the discriminative dilation result — Lorentz kinematics selects the norm
law among the tested allocation rules (PYTHAGOREAN max |dev| 2.286e-13, i.e. float roundoff, vs LINEAR 0.414143 and QUADRATIC 0.24589). Dilation is *reproduced under an inserted law*, not derived; no emergence is claimed.
