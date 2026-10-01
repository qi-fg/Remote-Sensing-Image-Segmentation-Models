# Contributing

Thanks for helping keep this remote-sensing segmentation resource accurate and reproducible.

## Adding a paper or method

Please provide:

- method name;
- publication year and venue/status;
- paper URL (prefer arXiv, DOI, or publisher page);
- code URL, if available;
- **code status:** Official / Author-provided / Community implementation / No public code;
- task type (semantic, instance, weakly supervised, reasoning segmentation, change detection, etc.);
- representative dataset(s);
- one short factual description of the main idea.

Do not label a repository as **official** unless it is linked by the authors, paper/project page, or another primary source.

## Adding a benchmark number

Follow [BENCHMARKS.md](BENCHMARKS.md).

Every reported score must include enough protocol information to avoid misleading comparisons. Prefer primary sources and keep paper-reported results separate from results reproduced in this repository.

## Adding a dataset

Please include:

- official/project download page;
- task type;
- image count / pair count when known;
- classes;
- spatial resolution / GSD when relevant;
- official split or split policy;
- license or usage restrictions when known.

## Pull-request checklist

- [ ] Links resolve.
- [ ] Venue/year are verified.
- [ ] Code status is not overstated.
- [ ] Description is factual and concise.
- [ ] Dataset/result claims have a primary source.
- [ ] Existing category counts are updated if needed.
- [ ] Baseline changes pass the smoke-test workflow.

## Scope

The repository focuses on methods that directly perform segmentation, provide a segmentation backbone/adapter, provide grounding that is materially used for segmentation, or orchestrate a segmentation pipeline.

For VLM/LLM/Agent work, please state its role explicitly so the list does not drift into a general remote-sensing multimodal-model index.
