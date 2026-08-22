# Security Policy

## Reporting a Vulnerability

If you believe you have found a security vulnerability in ToolWeave, please
report it privately. **Do not open a public GitHub issue for security reports.**

Email the maintainer at **dikhand1@in.ibm.com** with:

- a description of the issue,
- the steps required to reproduce it,
- the affected version or commit,
- and, if known, any mitigations.

You will receive an acknowledgement within 3 working days. This project follows
a 90 day disclosure timeline.

## Scope

ToolWeave is a research tool for synthesizing tool-calling dialogue data. It is
provided as-is for research use and is not a supported IBM product. Only the
`main` branch is maintained; there are no backported security fixes for earlier
commits.

Note that this project invokes external LLM services and, during API synthesis,
fetches public data from the Wikipedia and Wikidata APIs. Review your own
credential handling and network policy before running it.
