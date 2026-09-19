# Local competition-flow tests

Run from the repository root after installing the locked project dependencies:

```sh
poetry install --no-root
poetry run python -m unittest discover -s tests -v
```

No API credentials are needed. Tests suppress `.env` loading, replace the environment with a dummy token, intercept HTTP requests, and block socket connections.

The suite uses the production `SummerTemplateBot2026` and installed `forecasting-tools` SDK. Synthetic question fixtures use the SDK's normalized question schema, not raw Metaculus HTTP response envelopes. They cover binary, multiple-choice, continuous numeric, and discrete numeric questions (a numeric CDF with fewer bins).

The real flow runs from tournament question retrieval through research/forecast prompts, five predictions per question, aggregation, explanation/report generation, disk artifacts, and SDK submission serialization. Inference and structured-output model responses are deterministic substitutes. HTTP submissions are captured in memory, never sent. Assertions check probability bounds, normalized option probabilities, monotonic CDFs and bin counts, question IDs, private comments, dry-run behavior, duplicate skipping, empty rounds, and provider failures.

These are integration/contract checks, not an accuracy benchmark. They do not test real question HTTP parsing, authentication, provider response quality, rate limits, server acceptance, or scheduling. Date and conditional formats are not covered by this suite. Axion and other provider adapters will need their own request/response contract cases when connected.

## Live verification after credentials are configured

```sh
poetry run python main.py --mode test_questions
```

This fetches real testing-area questions and uses paid inference, but does not publish. After reviewing those reports, an explicit `--publish` on the same testing-area command checks actual submission and private comments. Do not use open competition questions to tune a forecast or rerun an answer you dislike.

## Hosting

The prepared GitHub Actions workflow launches temporary Ubuntu runners every 20 minutes, separately for Fall 2026 and MiniBench. Each installs the locked dependencies, runs the bot, uploads reports, and terminates. The model/search APIs run remotely. It is not a permanently running server. The schedule only operates after the workflow is on the default branch and `BOT_ENABLED=true`; credentials and a successful live test are still prerequisites.
