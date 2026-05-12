---
name: handoff
description: Manage Handoff task management, agent delegation, run monitoring, status reporting, project knowledge, and review workflow through the Handoff CLI. Use when users ask RayClaw to create/list/triage tasks, prepare work for agents, dispatch or monitor coding-agent runs, summarize progress, store implementation artifacts, or handle Review/Done follow-up.
---

# Handoff

Use Handoff as the canonical work tracker and agent-run orchestrator. Do not
connect to Postgres directly, mutate the dashboard, or create markdown task
boards in project repositories.

Requires the `handoff` command to be on `PATH`.

## Command Environment

On `td@100.84.248.34`, always run Handoff with the remote Postgres URL:

```sh
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff health
```

For other commands, keep the same `env ... handoff` prefix. The bare `handoff`
default may try a local Unix socket and fail.

## Inspect Work

Use these first when asked for status, planning context, or progress:

```sh
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff task list [project-id]
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff task show <task-id>
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff run list [task-id]
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff task brief <task-id>
```

Report status with task id, state/column, blockers, latest run id/state,
result branch, log path, and the next recommended action.

## Create And Prepare Tasks

Create project work in Handoff, not in repo-local docs:

```sh
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff project add <project-id> <name>
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff repo add <project-id> <name> <path> [remote-url] [default-branch]
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff task create <project-id> <title> <description> <acceptance>
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff task backlog <task-id>
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff task ready <task-id>
```

If the project or repo is unclear, infer it from the working directory when
safe; otherwise ask one concise question. A task should have a concrete problem
statement and acceptance criteria before moving to Ready.

Use blockers for dependency ordering:

```sh
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff task block <task-id> <blocks-on-task-id>
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff task unblock <task-id> <blocks-on-task-id>
```

## Delegate Agent Work

For coding work that should be tracked, prefer Handoff dispatch over direct
ACP delegation:

```sh
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff run dispatch <task-id>
```

Dispatch only Ready tasks. Handoff creates an Agent Run, launches the remote
worktree runner, records the log path, and moves the Task to Running. Terminal
runs move the Task to Review, not Done.

If dispatch fails because the repo is dirty or unregistered, report the exact
reason and the command that failed. Do not silently use another runner unless
the user explicitly asks for untracked work.

## Review Workflow

Use Review actions only when the user explicitly asks, or after clearly
summarizing what will happen:

```sh
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff review accept <task-id>
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff review retry <task-id>
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff review revise <task-id> <description> <acceptance>
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff review follow-up <task-id> <title> <description> <acceptance>
```

Accept moves Review to Done. Retry creates another Ready run path for the same
task. Revise updates the task specification. Follow-up creates a new task.

## Store Durable Knowledge

Store reusable findings in Handoff knowledge so future agents can retrieve
them through task briefs:

```sh
env HANDOFF_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff HANDOFF_REMOTE_DATABASE_URL=postgresql://postgres@127.0.0.1:55432/handoff handoff knowledge add <project-id> run-note "Short title" "What future agents should know"
```

Use knowledge for implementation notes, investigation findings, decisions, and
handoff artifacts that should survive beyond a single worktree.

## Response Shape

When reporting back, keep it operator-focused:

- `Task`: id, title, state/column
- `Run`: latest run id, state, result branch, log path
- `Blockers`: none or blocking task ids
- `Next`: one concrete next action

Mention command failures exactly. Avoid claiming a run is complete until
`handoff run list` shows `Succeeded`, `Failed`, or `Cancelled`.
