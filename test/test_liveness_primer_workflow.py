from pathlib import Path

import yaml

from skylos.rules.config.cicd.github_actions import scan_github_actions_file


WORKFLOW_PATH = Path(".github/workflows/liveness-primer.yml")


def _workflow():
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


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
        "actions/cache/restore@27d5ce7f107fe9357f9df03efb73ab90386fccae",
        "actions/cache/save@27d5ce7f107fe9357f9df03efb73ab90386fccae",
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "actions/setup-go@40f1582b2485089dde7abd97c1529aa768e1baff",
        "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
        "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d",
    }
    for step in action_steps:
        action_ref = step["uses"].split("@", 1)[1]
        assert len(action_ref) == 40
        assert all(character in "0123456789abcdef" for character in action_ref)

    assert workflow["env"] == {
        "LIVENESS_PRIMER_REF": "ed343221359560929f78d38ddf081d66d2acfad5"
    }

    trusted_checkout = next(
        step for step in steps if step.get("name") == "Check out trusted Skylos base"
    )
    assert trusted_checkout["with"] == {
        "ref": "${{ github.event.pull_request.base.sha }}",
        "path": "_trusted_skylos",
        "persist-credentials": False,
    }

    primer_checkout = next(
        step for step in steps if step.get("name") == "Check out pinned liveness_primer"
    )
    assert primer_checkout["with"] == {
        "repository": "mcdigman/liveness_primer",
        "ref": "${{ env.LIVENESS_PRIMER_REF }}",
        "path": "_liveness_primer",
        "persist-credentials": False,
    }

    setup_go = next(step for step in steps if step.get("name") == "Install Go")
    assert setup_go["with"] == {"go-version": "1.22", "cache": False}

    setup_uv = next(step for step in steps if step.get("name") == "Install uv")
    assert setup_uv["with"] == {
        "version": "0.12.5",
        "python-version": "3.13",
        "enable-cache": False,
    }


def test_liveness_primer_workflow_builds_trusted_base_go_engine():
    workflow = _workflow()
    steps = workflow["jobs"]["blast-radius"]["steps"]
    build = next(
        step for step in steps if step.get("name") == "Build trusted base Go engine"
    )

    assert build["env"] == {
        "TRUSTED_BASE_SHA": "${{ github.event.pull_request.base.sha }}"
    }
    assert build["shell"] == "bash"
    script = build["run"]
    assert '[[ ! "$TRUSTED_BASE_SHA" =~ ^[0-9a-f]{40}$ ]]' in script
    assert "git -C _trusted_skylos rev-parse HEAD" in script
    assert '[[ "$trusted_checkout_sha" != "$TRUSTED_BASE_SHA" ]]' in script
    assert "cd _trusted_skylos/skylos/engines/go" in script
    assert 'go build -trimpath -o "$engine_dir/skylos-go" ./cmd/skylos-go' in script


def test_liveness_primer_workflow_uses_locked_comparison_contract():
    workflow = _workflow()
    comparison = _comparison_step(workflow)
    assert comparison["env"] == {
        "SKYLOS_REPOSITORY": "${{ github.server_url }}/${{ github.repository }}",
        "SKYLOS_GO_BIN": (
            "${{ format('{0}/skylos-go-engine/skylos-go', runner.temp) }}"
        ),
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
    assert "--container" in script
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
        step for step in steps if step.get("name") == "Upload blast-radius evidence"
    )
    assert artifact["if"] == "always()"
    assert artifact["with"]["name"] == "liveness-primer-report"
    assert artifact["with"]["if-no-files-found"] == "error"
    assert artifact["with"]["retention-days"] == 14
    assert set(artifact["with"]["path"].splitlines()) == {
        "liveness-primer-report.md",
        "liveness-primer-report.json",
    }

    assert "pull_request_target" not in workflow_source
    assert "secrets." not in workflow_source
    assert "pull-requests: write" not in workflow_source
    assert "gh pr comment" not in workflow_source
    assert scan_github_actions_file(WORKFLOW_PATH, root=".") == []
