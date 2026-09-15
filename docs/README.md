# Documentation guide

Start with the question you want to answer:

| Question | Guide |
| --- | --- |
| How do I run the shared agent? | [Agent runbook](AGENT_RUN.md), [first test](first_test.md) |
| Which world should I test in, and what does a pass mean? | [Environments and acceptance progression](environments/README.md) |
| How do I launch tests? | [Testing index](TESTING.md), [simulation smoke battery](simulation_testing_plan.md) |
| How do I freeze an evaluation and collect evidence? | [Evaluation runbook](evaluation.md) |
| What actually happened in an experiment? | [Experiment index](experiments/README.md) |
| How do configuration and robot adapters work? | [Configuration](emet_config.md), [supported robots](robots/supported_robots.md) |
| Which results belong in the paper? | [Paper benchmark mapping](paper_benchmarks.md) |

## Where new documentation belongs

- `environments/`: durable descriptions of worlds, difficulty, acceptance gates,
  and figure conventions. Link to setup instructions; do not duplicate them.
- `experiments/`: dated runs, frozen settings, outcomes, failures and artifact
  provenance. Historical results are not current support guarantees.
- `robots/`: embodiment-specific setup, calibration and hardware limitations.
- `plans/`: proposed work, not completed results.
- Existing top-level runbooks remain the command/reference entry points. This
  organization pass adds navigation without moving them or breaking old links.

When adding results, link both ways between the experiment and its environment
page. Keep performance numbers in the experiment record rather than copying
them into multiple setup guides.
