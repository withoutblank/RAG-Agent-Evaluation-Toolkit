# Security Policy

## Supported versions

Security fixes are applied to the latest released version and the current
`main` branch. This portfolio project does not promise long-term support for
older releases.

## Reporting a vulnerability

Do not disclose a suspected vulnerability, secret, or sensitive document in a
public issue. Use GitHub's private vulnerability reporting form:

<https://github.com/withoutblank/RAG-Agent-Evaluation-Toolkit/security/advisories/new>

Include the affected version or commit, impact, reproduction steps, and a
minimal sanitized example. Acknowledgement and remediation are provided on a
best-effort basis; no fixed response deadline is guaranteed.

## Secret and data handling

- API keys are read from the environment and must never be committed.
- `.env`, credentials, authentication headers, and private documents must not
  appear in logs, fixtures, screenshots, reports, or issues.
- Tests and scheduled workflows use deterministic fake providers.
- Revoke and rotate any exposed credential; deleting only the current file is
  not sufficient if the value entered Git history.

This repository is an educational portfolio project, not a production security
product or a service for storing sensitive information.
