#!/usr/bin/env python3
import json

with open("latest_status.json", "r") as f:
    data = json.load(f)

runs = data.get("workflow_runs", [])

# Filter runs for our latest commit
latest_commit = "56c3ae88"
latest_runs = [run for run in runs if run["head_sha"].startswith(latest_commit)]

print(f"🔍 Workflow Status for Latest Commit ({latest_commit}):")
print("=" * 60)

if not latest_runs:
    print("⏳ No workflow runs found for latest commit yet...")
else:
    for run in latest_runs:
        status = run["status"]
        conclusion = run.get("conclusion", "N/A")
        workflow_name = run["name"]
        run_number = run["run_number"]

        # Status emoji
        if conclusion == "success":
            emoji = "✅"
        elif conclusion == "failure":
            emoji = "❌"
        elif conclusion == "cancelled":
            emoji = "🚫"
        elif status == "in_progress":
            emoji = "🔄"
        elif status == "queued":
            emoji = "⏳"
        else:
            emoji = "❔"

        print(f"{emoji} Run #{run_number} - {workflow_name}")
        print(f"   Status: {status} | Conclusion: {conclusion}")
        print()

print()
print("📊 Summary of Recent Workflow Runs:")
for i, run in enumerate(runs[:6]):
    commit_sha = run["head_sha"][:8]
    status = run["status"]
    conclusion = run.get("conclusion", "N/A")
    emoji = (
        "✅"
        if conclusion == "success"
        else (
            "❌"
            if conclusion == "failure"
            else "🔄" if status == "in_progress" else "⏳"
        )
    )
    print(
        f'{emoji} {run["name"]} (#{run["run_number"]}) - {commit_sha} - {status}/{conclusion}'
    )
