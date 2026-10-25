# Superseded development records

These are not final experiment results and must not be used in the report.

`strict_parser/` records the first evaluation using a stricter-than-upstream
coordinate parser. Replaying actual responses against upstream exposed 32
numeric-triple lists that upstream accepts by taking their first two values.
The baseline parser was corrected and all evaluations were regenerated from
the same 6,000 saved responses, with no new API calls. Final results are in
`../main/`. These files preserve the investigation history, not a separate
experimental repetition.
