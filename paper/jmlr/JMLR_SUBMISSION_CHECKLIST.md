# JMLR Submission Checklist - DLP Grokking

## Required before submission

1. **Corresponding author details**
   - JMLR requires the title page to contain the corresponding author's complete name, postal address, and e-mail address.
   - The public/arXiv version can remain email-free if desired, but the JMLR submission PDF must include these details.

2. **Authorship decision before upload**
   - JMLR states that the initial author list is final during peer review except in rare, justified circumstances.
   - Decide whether any collaborator (e.g. Teddy Lazebnik) will be a co-author before submission.

3. **Funding / compute support disclosure**
   - JMLR requires disclosure of third-party support during the prior 36 months that supported any part of the work.
   - This includes compute resources, cloud/GPU resources, donations, grants, stipends, or programming support.
   - Confirm and disclose the exact LUMI/EuroHPC access mechanism if those resources supported the experiments.

4. **Competing interests / conflicts of interest**
   - List relevant financial relationships and any JMLR Action Editors with whom the author has collaborated in the past three years.
   - Confirm no conflicts with proposed Action Editors and reviewers.

5. **JMLR formatting**
   - Official `jmlr2e.sty` used.
   - PDF only for manuscript submission.
   - Abstract <= 200 words.
   - Running title <= 50 characters: `Grokking the Discrete Logarithm`.
   - Exactly five keywords included.
   - Keep manuscript under 35 pages when practical.
   - Upload file < 5 MB.

6. **Cover letter**
   Must include:
   - prior/overlapping publications;
   - co-author awareness/consent (or sole-author statement);
   - conflicts of interest;
   - 3-5 suggested Action Editors;
   - 3-5 suggested reviewers;
   - keywords.

7. **Preprint**
   - JMLR explicitly permits arXiv/personal-site preprints.
   - If arXiv is posted before journal submission, update the cover letter with the arXiv identifier.
   - Do not submit simultaneously to another journal or conference.

8. **Reproducibility package**
   - Freeze one canonical run manifest.
   - Release deterministic dataset generator + exact split manifests.
   - Release code/configs for all reported experiments.
   - Release figure-source data and analysis scripts.
   - Archive a versioned snapshot (e.g. Zenodo DOI).
   - Include hashes for key checkpoints used in mechanistic analyses.

## Voluntary AI-use disclosure included in the manuscript

> AI-assisted tools were used for programming assistance, code generation, language drafting, and editorial support. The research questions, hypotheses, experimental design, experiment selection, mathematical reasoning, code review, test execution and validation, result interpretation, proofreading, and final editing were carried out or directly verified by the author. All scientific claims and conclusions are the author's responsibility.

The current JMLR author-information page does not state a dedicated LLM-use disclosure policy. This statement is therefore included as a voluntary transparency disclosure, not because the current JMLR page explicitly requires it.

## Proposed Action Editors - verify COI before using

- Joan Bruna
- Samy Bengio
- Mehryar Mohri
- Kilian Weinberger

## Proposed reviewers - verify COI before using

- Neel Nanda
- Ziming Liu
- Vikrant Varma
- Miles Cranmer
- Huu Danh Nguyen
