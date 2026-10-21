# Contributing

The most useful contribution is a reproduction. Run a fixture on your hardware, keep the JSON it writes, and open an issue with the summary line, your CPU, OS and Python versions. Disagreement with a headline number is a finding, not a problem.

## Ground rules

- Results never overwrite. New runs get new names. Keep failures.
- Every run file keeps the script hash the fixture writes. Do not edit run files by hand.
- Changes to a fixture's simulation (`r7d_replay.py` world, crowd, sweep, trace) change every downstream number; say so in the pull request and rerun the equivalence check.
- Do not add a game engine dependency to Part 3. Its value is that it runs on numpy alone.
- No third-party code without a licence line in the pull request.

## Good first contributions

See the issues labelled `good first issue`: reproduction reports, a plotting script for the summaries, continuous integration for the small equivalence run, a citation file.

## Larger work

A real UDP hop between deciders, a native port of the fixture kernel, a Linux build of the native lab, and runs beyond D = 8 are tracked as issues labelled `help wanted`. Open a short design note in the issue before building, so the pass line is agreed first.

## Style

Documents follow the house rules in `docs/`: a number carries its evidence level, a claim names its run, and a Limits section comes before any conclusion is quoted.
