#!/usr/bin/env python3
"""
Comprehensive workflow monitoring for GenomicLightning CI/CD.
"""

import json
import urllib.request
import time
import sys
from datetime import datetime


def get_workflow_status():
    """Get the current status of GitHub Actions workflows."""
    url = "https://api.github.com/repos/SanjeevaRDodlapati/GenomicLightning/actions/runs?per_page=5"

    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())

        return data.get("workflow_runs", [])
    except Exception as e:
        print(f"❌ Error fetching workflow data: {e}")
        return []


def format_workflow_info(run):
    """Format workflow information for display."""
    name = run.get("name", "Unknown")
    status = run.get("status", "Unknown")
    conclusion = run.get("conclusion", "None")
    created = run.get("created_at", "")
    commit_title = (
        run.get("display_title", "")[:50] + "..."
        if len(run.get("display_title", "")) > 50
        else run.get("display_title", "")
    )
    run_number = run.get("run_number", "N/A")

    # Status icons
    status_icon = {
        "queued": "⏳",
        "in_progress": "🔄",
        "completed": "✅" if conclusion == "success" else "❌",
    }.get(status, "❓")

    conclusion_icon = {
        "success": "✅",
        "failure": "❌",
        "cancelled": "⚠️",
        "skipped": "⏭️",
    }.get(conclusion, "⏳" if conclusion is None else "❓")

    return f"{status_icon} {name} #{run_number}: {status} | {conclusion_icon} {conclusion or 'Running'} | {commit_title}"


def monitor_workflows(duration_minutes=10):
    """Monitor workflows for a specified duration."""
    print("🔍 GitHub Actions Workflow Monitor")
    print("=" * 60)

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    check_interval = 30  # seconds

    last_check_data = {}

    while time.time() < end_time:
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"\n📊 Status at {current_time}:")
        print("-" * 40)

        runs = get_workflow_status()

        if not runs:
            print("❌ No workflow data available")
            time.sleep(check_interval)
            continue

        # Check recent runs (last 2)
        recent_runs = runs[:2]
        any_running = False

        for run in recent_runs:
            run_id = run.get("id")
            status = run.get("status")
            conclusion = run.get("conclusion")

            info = format_workflow_info(run)
            print(f"  {info}")

            # Check if status changed
            if run_id in last_check_data:
                old_status = last_check_data[run_id].get("status")
                old_conclusion = last_check_data[run_id].get("conclusion")

                if status != old_status or conclusion != old_conclusion:
                    print(f"    🔄 Status changed: {old_status} → {status}")
                    if conclusion and conclusion != old_conclusion:
                        print(f"    📝 Conclusion: {old_conclusion} → {conclusion}")

            last_check_data[run_id] = {"status": status, "conclusion": conclusion}

            if status in ["queued", "in_progress"]:
                any_running = True

        if not any_running:
            print("\n✅ All recent workflows completed!")

            # Summary of latest results
            print("\n📋 Final Results:")
            for run in recent_runs:
                if run.get("conclusion"):
                    name = run.get("name", "Unknown")
                    conclusion = run.get("conclusion")
                    icon = "✅" if conclusion == "success" else "❌"
                    print(f"  {icon} {name}: {conclusion.upper()}")

            break

        print(f"\n⏱️  Next check in {check_interval} seconds...")
        time.sleep(check_interval)

    if time.time() >= end_time:
        print(f"\n⏰ Monitoring timeout reached ({duration_minutes} minutes)")
        print(
            "Check manually at: https://github.com/SanjeevaRDodlapati/GenomicLightning/actions"
        )


def get_failure_details(run_id):
    """Get detailed failure information for a specific run."""
    url = f"https://api.github.com/repos/SanjeevaRDodlapati/GenomicLightning/actions/runs/{run_id}/jobs"

    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read())

        failed_jobs = []
        for job in data.get("jobs", []):
            if job.get("conclusion") == "failure":
                failed_jobs.append(
                    {
                        "name": job.get("name"),
                        "steps": [
                            step
                            for step in job.get("steps", [])
                            if step.get("conclusion") == "failure"
                        ],
                    }
                )

        return failed_jobs
    except Exception as e:
        print(f"❌ Error fetching job details: {e}")
        return []


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Quick status check
        print("🔍 Quick workflow status check:")
        runs = get_workflow_status()
        for run in runs[:3]:
            print(f"  {format_workflow_info(run)}")
    else:
        # Full monitoring
        monitor_workflows(10)  # Monitor for 10 minutes
