# Contributing

Contributions to ToolWeave are welcome. This project accompanies the paper
*ToolWeave: Structured Synthesis of Complex Multi-Turn Tool-Calling Dialogues*,
so changes that improve the reproducibility or clarity of the synthesis pipeline
are especially useful.

## Contributing in general

To contribute code or documentation, please open a
[pull request](https://github.com/IBM/ToolWeave/pulls).

Before starting on anything substantial, please
[raise an issue](https://github.com/IBM/ToolWeave/issues) so it can be discussed
first. This applies both to new features and to bug fixes, and avoids work that
turns out to duplicate or conflict with something already in progress.

### Reporting bugs

When reporting a bug, include the command you ran, the relevant portion of the
output, and which model and configuration you used. Because the pipeline calls
LLMs, please note whether the behaviour reproduces across runs — generation
variance is expected in some stages.

## Setup

The project targets **Python 3.13+** and pins its dependencies in `uv.lock`.

```bash
uv sync
```

To use watsonx.ai models, create a `.env` file in the repository root as
described in the [README](README.md), and set your model parameters in
`watsonx_llm_config.yml`. For a vLLM endpoint, use `vllm_llm_config.yml`
instead. Never commit `.env` or any credentials.

## Testing

This repository has no automated test suite. Before opening a pull request,
verify your change by running the affected stage end to end on a single domain —
for example, restricting `domains.txt` to one entry and running the pipeline as
documented in the README. Include the command you used and a short summary of
the result in your pull request description.

## Coding style

Match the surrounding code. The existing scripts use type annotations and
docstrings throughout; please keep both for any function you add or modify.

## Merge approval

The maintainers use LGTM (Looks Good To Me) in review comments to indicate
acceptance. For the list of maintainers, see [MAINTAINERS.md](MAINTAINERS.md).

## Legal

Each source file must include a copyright and license header. The SPDX format is
preferred:

```
#
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
#
```

This project uses the
[Developer Certificate of Origin 1.1 (DCO)](https://developercertificate.org/),
the same mechanism the Linux kernel community uses to manage code
contributions. When you submit a patch, include a sign-off line in the commit
message:

```
Signed-off-by: Jane Doe <jane.doe@example.com>
```

Git adds this for you with:

```bash
git commit -s
```

The DCO bot checks this on incoming pull requests.

## Communication

For questions about the paper or the pipeline, open an
[issue](https://github.com/IBM/ToolWeave/issues) or contact the maintainer
listed in [MAINTAINERS.md](MAINTAINERS.md).
