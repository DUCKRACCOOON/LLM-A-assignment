# ACL reproduction report

The English report is an independent `gm` reproduction draft. Team authorship
and the final submission version must be settled by the team. Actual experiments,
negative findings, model/API substitutions, proposal ablations and limitations
are included. The Korean explanation is `../최신/재현_실험_결과.md`.

Regenerate the editable source and figures after final evaluation:

```bash
python -m reproduction.report
```

Compile from this directory using the official ACL style:

```bash
tectonic --outdir ../output/pdf LLM_Astar_Reproduction_Report.tex
```

Alternatively use a LaTeX installation or upload this directory's `.tex`,
`acl.sty` and `figures/` to your usual LaTeX editor. The source is generated from
`report_template.tex`; change the template for durable narrative edits and
regenerate the source. Data-driven numbers come from `results/main`.

Official style source: https://github.com/acl-org/acl-style-files.
The downloaded style is preserved with its header and upstream README.
`TEMPLATE_SOURCE.json` records the source and local compiler version.

`paper_table1.json` is a transcription of the paper's reference numbers, not
measurements from this experiment. The report separates the two.
