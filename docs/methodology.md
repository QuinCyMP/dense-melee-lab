# Methodology

How the work was done, so the numbers can be weighed.

## Who did what

One person directed the research: chose the questions, set the pass lines, approved every build and every rented machine, and made every design decision. AI coding agents (Claude Code and OpenAI Codex, in several sessions, sometimes in parallel on the same question) wrote the fixtures, ran them, and wrote the notes these documents are condensed from. Independent review rounds used separate agent sessions that had not seen the work.

That process produced wrong claims. It also produced the fixtures that caught them. Part 1, section 5 lists the claims that did not survive. The rule that made this possible: a number without an executable, a trace and a baseline is an estimate, and is labelled as one.

## Evidence levels

Every number carries one of these levels, and the documents keep them apart:

| Level | Meaning |
|---|---|
| measured | produced by a run whose script hash, machine and raw per-tick data are recorded |
| model | arithmetic on measured inputs; the output was never observed |
| theoretical | an extrapolation or a published figure applied to this workload |
| unmeasured | an attempt that could not be completed (for example a binary blocked by the OS); never reported as a result |

A pass line was written before each experiment. Results are reported against it, including misses.

## Run hygiene

- Every run file records the SHA-256 of the script that produced it and of the simulation it imports.
- Result files refuse to overwrite. A new experiment gets a new name. Failures stay in the record.
- Benchmarks pin a core and never run two at once.
- Each fixture carries a negative control that must fail (a 1 m halo, a dropped ordering rule, a moved micrometre), so a passing check is known to bite.
- Security settings are never changed to make a benchmark run.

## What is withheld, and why

The complete result tables, the raw JSON of several hundred runs, the Unity harness and builds behind Part 2, and the project's measurement log are not published. They belong to a game project that is not announced. The fixtures in this repository regenerate the Part 1 and Part 3 numbers; Part 2 is reported at headline precision only. If you reproduce a figure that disagrees with a headline here, open an issue with your run file.

## Hardware

Intel Core i5-13400F (10 cores, 16 threads, no AVX-512), Windows 11, Python 3.12, numpy 2.5, scipy 1.18. One confirmation run of the Part 2 harness on a rented Linux machine (AMD EPYC, 30 cores, IL2CPP build) gave the same figures within noise.
