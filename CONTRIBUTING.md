# Contributing to qwen3-tts-server

Bug reports, ideas and pull requests are welcome.

## Getting started

- Open an issue for bugs, ideas or questions.
- Fork, branch, and submit a pull request.
- The whole project is `app.py` and `test_app.py`. The tests stub `torch`, `numpy` and `soundfile` in-process, so `pip install fastapi httpx && python -m unittest discover -p 'test_*.py'` needs no GPU and none of the model dependencies.

## Guidelines

- Keep it small. This is a thin HTTP layer over the [`qwen-tts`](https://pypi.org/project/qwen-tts/) library; synthesis logic belongs upstream, and a new knob is an environment variable with a default that keeps today's behaviour.
- Anything that touches the model goes through `TTSEngine` under its lock. Only one synthesis runs at a time on purpose: the `requests_waiting` gauge is how a client learns it is queued.
- Keep `/v1/audio/speech` and `/v1/audio/voices` compatible with the OpenAI audio API shape; the extra fields are additive.
- Run the tests before submitting a PR; CI runs them on every pull request and builds the image.
- **We use [release-please](https://github.com/googleapis/release-please).** Write [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, …) so versions and the changelog derive from the history.

## AI policy

AI coding tools are used in this project, steered by the maintainer, and nothing is merged without a human having reviewed it. Contributions written with such tools are welcome under the same rule: you are responsible for what you submit, so understand it and make sure it works.
