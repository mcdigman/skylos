from pathlib import Path

import yaml

from skylos.rules.config.cicd.github_actions import scan_github_actions_file


WORKFLOW_PATH = Path(".github/workflows/liveness-primer.yml")
COMMENT_WORKFLOW_PATH = Path(".github/workflows/liveness-primer-comment.yml")


def _workflow():
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _comment_workflow():
    return yaml.safe_load(COMMENT_WORKFLOW_PATH.read_text(encoding="utf-8"))


def _comparison_step(workflow):
    return next(
        step
        for step in workflow["jobs"]["blast-radius"]["steps"]
        if step.get("name") == "Compare base with the pull request merge result"
    )


def test_liveness_primer_workflow_is_scoped_read_only_and_advisory():
    workflow = _workflow()
    triggers = workflow.get("on", workflow.get(True))

    assert set(triggers) == {"pull_request"}
    pull_request = triggers["pull_request"]
    assert pull_request["types"] == [
        "opened",
        "synchronize",
        "reopened",
        "ready_for_review",
    ]
    assert set(pull_request["paths"]) == {
        "skylos/**",
        "pyproject.toml",
        "MANIFEST.in",
        ".github/workflows/liveness-primer.yml",
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"] == {
        "group": "liveness-primer-${{ github.event.pull_request.number }}",
        "cancel-in-progress": True,
    }

    job = workflow["jobs"]["blast-radius"]
    assert job["if"] == "github.event.pull_request.draft == false"
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["timeout-minutes"] == 45

    comparison = _comparison_step(workflow)
    assert "--all" in comparison["run"]
    assert "--fail-on" not in comparison["run"]
    assert "continue-on-error" not in comparison
    assert "continue-on-error" not in job


def test_liveness_primer_workflow_pins_actions_and_toolchain():
    workflow = _workflow()
    steps = workflow["jobs"]["blast-radius"]["steps"]
    action_steps = [step for step in steps if "uses" in step]

    assert {step["uses"] for step in action_steps} == {
        "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5",
        "astral-sh/setup-uv@e4db8464a088ece1b920f60402e813ea4de65b8f",
        "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
    }
    for step in action_steps:
        action_ref = step["uses"].split("@", 1)[1]
        assert len(action_ref) == 40
        assert all(character in "0123456789abcdef" for character in action_ref)

    checkout = next(
        step
        for step in steps
        if step.get("name") == "Check out pinned liveness_primer"
    )
    assert checkout["with"] == {
        "repository": "mcdigman/liveness_primer",
        "ref": "d6f3118a2cfc465426500eab449005fe56845c58",
        "path": "_liveness_primer",
        "persist-credentials": False,
    }

    setup_uv = next(step for step in steps if step.get("name") == "Install uv")
    assert setup_uv["with"] == {
        "version": "0.12.5",
        "python-version": "3.13",
        "enable-cache": False,
    }


def test_liveness_primer_workflow_uses_locked_comparison_contract():
    workflow = _workflow()
    comparison = _comparison_step(workflow)
    assert comparison["env"] == {
        # This fork measures its own commits, which upstream does not have.
        "SKYLOS_REPOSITORY": "${{ github.server_url }}/${{ github.repository }}",
        "BASE_SHA": "${{ github.event.pull_request.base.sha }}",
        "MERGE_SHA": "${{ github.sha }}",
        "REPORT_JSON": "liveness-primer-report.json",
        "REPORT_MARKDOWN": "liveness-primer-report.md",
    }
    script = comparison["run"]
    assert "${{" not in script
    assert '[[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]' in script
    assert "uv run --project _liveness_primer --locked liveness-primer run" in script
    assert "--tool skylos" in script
    assert '--repo "$SKYLOS_REPOSITORY"' in script
    assert '--old "$BASE_SHA"' in script
    assert '--new "$MERGE_SHA"' in script
    assert "--output github" in script
    assert '--json-out "$REPORT_JSON"' in script
    assert "--jobs 2" in script
    assert "--timeout 300" in script
    assert "set -euo pipefail" in script
    assert '| tee "$REPORT_MARKDOWN" >> "$GITHUB_STEP_SUMMARY"' in script
    assert 'test -s "$REPORT_JSON"' in script


def test_liveness_primer_workflow_preserves_evidence_without_write_access():
    workflow_source = WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = _workflow()
    steps = workflow["jobs"]["blast-radius"]["steps"]

    artifact = next(
        step
        for step in steps
        if step.get("name") == "Upload blast-radius evidence"
    )
    assert artifact["if"] == "always()"
    assert artifact["with"]["name"] == "liveness-primer-report"
    assert artifact["with"]["if-no-files-found"] == "error"
    assert artifact["with"]["retention-days"] == 14
    assert set(artifact["with"]["path"].splitlines()) == {
        "liveness-primer-report.md",
        "liveness-primer-report.json",
        # Liveness Primer Comment binds its comment target with this.
        "pr-number.txt",
    }

    assert "pull_request_target" not in workflow_source
    assert "secrets." not in workflow_source
    assert "pull-requests: write" not in workflow_source
    assert "gh pr comment" not in workflow_source
    assert scan_github_actions_file(WORKFLOW_PATH, root=".") == []


def test_comment_workflow_consumes_this_workflow_by_name():
    # An upstream sync that renames the measuring workflow silently detaches
    # the comment job, which fails by never running at all.
    comment = _comment_workflow()
    triggers = comment.get("on", comment.get(True))
    assert triggers["workflow_run"]["workflows"] == [_workflow()["name"]]


def test_workflow_hands_the_comment_job_a_pull_request_number():
    workflow = _workflow()
    steps = workflow["jobs"]["blast-radius"]["steps"]

    recorder = next(
        step
        for step in steps
        if step.get("name") == "Record the pull request under measurement"
    )
    assert recorder["env"] == {"PR_NUMBER": "${{ github.event.pull_request.number }}"}
    script = recorder["run"]
    assert "${{" not in script
    assert "set -euo pipefail" in script
    assert "tr -dc '0-9'" in script
    assert "> pr-number.txt" in script
