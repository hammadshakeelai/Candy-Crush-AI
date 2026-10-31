#!/usr/bin/env python3
"""
GitHub Cloud Notebook Runner CLI
================================
A lightweight, Kaggle-style CLI for executing Jupyter notebooks headlessly
on GitHub Actions runners (2 vCPU, 7 GB RAM) using GitHub CLI (`gh`) & Papermill.

Usage:
    # Run a notebook with default settings and stream logs:
    python scripts/cloud_run.py notebooks/smoke_test.ipynb

    # Inject parameters (JSON string or dict):
    python scripts/cloud_run.py notebooks/smoke_test.ipynb -p '{"n_samples": 2000, "noise_level": 0.05}'

    # Check status of the latest run:
    python scripts/cloud_run.py --status

    # List recent cloud runs:
    python scripts/cloud_run.py --list

    # Download artifacts from a specific run:
    python scripts/cloud_run.py --download <run_id>
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
from pathlib import Path

WORKFLOW_FILE = "cloud_notebook_runner.yml"


def run_cmd(cmd, check=True, capture_output=False, text=True):
    """Execute a shell command with real-time streaming or string output."""
    if capture_output:
        res = subprocess.run(cmd, check=check, capture_output=True, text=text)
        return res.stdout.strip()
    return subprocess.run(cmd, check=check)


def check_prerequisites():
    """Verify gh CLI installation and authentication."""
    if not shutil.which("gh"):
        sys.exit(
            "❌ Error: GitHub CLI ('gh') is not installed.\n"
            "   Install it via: https://cli.github.com/ or 'sudo apt install gh' / 'pacman -S github-cli'"
        )

    try:
        auth_res = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
        if auth_res.returncode != 0:
            sys.exit(
                "❌ Error: GitHub CLI is not authenticated.\n"
                f"   {auth_res.stderr.strip()}\n"
                "   Please run 'gh auth login' to authenticate your GitHub account."
            )
    except Exception as e:
        sys.exit(f"❌ Error checking gh authentication: {e}")


def get_current_branch():
    """Return the active git branch name."""
    try:
        return run_cmd(["git", "branch", "--show-current"], check=True, capture_output=True)
    except Exception:
        return "main"


def sync_git_if_needed(notebook_path: Path, auto_push: bool = True):
    """Ensure the notebook and local commits are pushed to remote."""
    branch = get_current_branch()
    
    # Check if notebook exists
    if not notebook_path.exists():
        sys.exit(f"❌ Error: Notebook '{notebook_path}' does not exist.")

    # Check for uncommitted changes in the notebook
    status = run_cmd(["git", "status", "--porcelain", str(notebook_path)], capture_output=True)
    if status:
        print(f"📝 Local changes detected in '{notebook_path}'.")
        if auto_push:
            print("   Auto-staging and committing...")
            run_cmd(["git", "add", str(notebook_path)])
            run_cmd(["git", "commit", "-m", f"Update {notebook_path.name} before cloud execution"])
        else:
            sys.exit("❌ Error: Uncommitted changes in notebook. Commit/push first or pass --auto-push.")

    # Check unpushed commits
    unpushed = run_cmd(["git", "log", f"origin/{branch}..HEAD", "--oneline"], capture_output=True)
    if unpushed:
        print(f"🚀 Pushing local commits to origin/{branch} so GitHub runner has latest code...")
        run_cmd(["git", "push", "origin", branch])
    print(f"✓ Repository synchronized with remote branch '{branch}'.")


def list_recent_runs(limit: int = 5):
    """Display recent cloud runner workflow executions."""
    check_prerequisites()
    cmd = ["gh", "run", "list", f"--workflow={WORKFLOW_FILE}", f"--limit={limit}"]
    subprocess.run(cmd)


def get_latest_run_id(branch: str, wait_seconds: int = 15):
    """Poll for the newly triggered workflow run ID."""
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        try:
            out = run_cmd(
                [
                    "gh", "run", "list",
                    f"--workflow={WORKFLOW_FILE}",
                    f"--branch={branch}",
                    "--limit=1",
                    "--json", "databaseId,status,createdAt,url"
                ],
                capture_output=True
            )
            data = json.loads(out)
            if data and len(data) > 0:
                return data[0]["databaseId"], data[0]["url"]
        except Exception:
            pass
        time.sleep(2)
    return None, None


def download_artifacts(run_id: str, output_dir: Path):
    """Download workflow results artifact into target folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📦 Downloading execution artifacts for Run #{run_id} to '{output_dir}/'...")
    try:
        run_cmd([
            "gh", "run", "download", str(run_id),
            "--name", "notebook-results",
            "--dir", str(output_dir)
        ], check=True)
        print(f"✓ Artifacts successfully downloaded to {output_dir.resolve()}/")
        print("\nDownloaded files:")
        for item in sorted(output_dir.glob("*")):
            if item.is_file():
                size_kb = item.stat().st_size / 1024
                print(f"  • {item.name:32} ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"⚠️ Warning: Could not download artifacts (run might have produced none or failed): {e}")


def show_status(run_id: str = None):
    """Display status of a specific or the latest run."""
    check_prerequisites()
    if run_id:
        subprocess.run(["gh", "run", "view", str(run_id)])
    else:
        branch = get_current_branch()
        latest_id, url = get_latest_run_id(branch, wait_seconds=4)
        if latest_id:
            print(f"Latest Run: #{latest_id} ({url})")
            subprocess.run(["gh", "run", "view", str(latest_id)])
        else:
            print(f"No recent runs found for workflow '{WORKFLOW_FILE}' on branch '{branch}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Headless Jupyter Notebook Cloud Runner via GitHub Actions & Papermill"
    )
    parser.add_argument(
        "notebook_path",
        nargs="?",
        default="notebooks/smoke_test.ipynb",
        help="Path to .ipynb notebook (default: notebooks/smoke_test.ipynb)"
    )
    parser.add_argument(
        "-p", "--parameters",
        default="{}",
        help="JSON string or path to JSON file containing parameters to inject via Papermill"
    )
    parser.add_argument(
        "-b", "--branch",
        default=None,
        help="Git branch to trigger workflow on (defaults to current branch)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Local directory to download artifacts into (default: output/)"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip auto-commit and push of local changes"
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Trigger workflow and return immediately without watching logs"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent workflow runs and exit"
    )
    parser.add_argument(
        "--status",
        nargs="?",
        const="",
        help="View status/logs of latest run or specific run ID"
    )
    parser.add_argument(
        "--download",
        metavar="RUN_ID",
        help="Download artifacts from a specific run ID"
    )

    args = parser.parse_args()

    # Subcommands
    if args.list:
        list_recent_runs()
        return

    if args.download:
        download_artifacts(args.download, Path(args.output_dir))
        return

    if args.status is not None:
        show_status(args.status if args.status != "" else None)
        return

    # Check prerequisites
    check_prerequisites()

    nb_path = Path(args.notebook_path)
    branch = args.branch or get_current_branch()
    out_dir = Path(args.output_dir)

    # Parse parameters
    params_str = args.parameters.strip()
    if params_str.startswith("{") and params_str.endswith("}"):
        pass  # already json string
    elif Path(params_str).exists():
        with open(params_str, "r") as pf:
            params_str = pf.read().strip()
    elif "=" in params_str:
        # Convert key=val,key2=val2 to JSON
        kv_pairs = {}
        for token in params_str.split(","):
            if "=" in token:
                k, v = token.split("=", 1)
                k = k.strip()
                v = v.strip()
                # parse number if possible
                try:
                    if "." in v:
                        v = float(v)
                    else:
                        v = int(v)
                except ValueError:
                    if v.lower() in ("true", "false"):
                        v = v.lower() == "true"
                kv_pairs[k] = v
        params_str = json.dumps(kv_pairs)

    print("=" * 65)
    print("🚀 GITHUB CLOUD NOTEBOOK RUNNER")
    print("=" * 65)
    print(f"• Notebook:   {nb_path}")
    print(f"• Branch:     {branch}")
    print(f"• Parameters: {params_str}")
    print(f"• Output Dir: {out_dir}/")
    print("=" * 65)

    # Sync git
    sync_git_if_needed(nb_path, auto_push=not args.no_push)

    # Dispatch workflow
    print(f"\n📡 Dispatching workflow '{WORKFLOW_FILE}' via GitHub CLI...")
    dispatch_cmd = [
        "gh", "workflow", "run", WORKFLOW_FILE,
        "--ref", branch,
        "-f", f"notebook_path={nb_path}",
        "-f", f"parameters={params_str}"
    ]
    run_cmd(dispatch_cmd, check=True)
    print("✓ Dispatch signal sent.")

    # Locate run ID
    print("⏳ Acquiring active workflow run ID...")
    run_id, run_url = get_latest_run_id(branch)
    if not run_id:
        print("⚠️ Could not resolve new run ID immediately. You can check status with:")
        print(f"   python scripts/cloud_run.py --list")
        return

    print("\n" + "=" * 65)
    print(f"🎉 Cloud Runner Job Active: Run #{run_id}")
    print(f"🌐 Web Console: {run_url}")
    print("=" * 65)

    if args.detach:
        print("\n[Detached mode] Workflow is executing headlessly in the cloud.")
        print(f"Check status anytime:  python scripts/cloud_run.py --status {run_id}")
        print(f"Download results when done: python scripts/cloud_run.py --download {run_id}")
        return

    # Watch execution
    print("\n📺 Streaming live execution logs from GitHub Actions runner...\n")
    watch_res = subprocess.run(["gh", "run", "watch", str(run_id), "--exit-status"])

    if watch_res.returncode == 0:
        print("\n🎉 [SUCCESS] Notebook execution finished successfully!")
        download_artifacts(run_id, out_dir)
    else:
        print(f"\n❌ [FAILURE] Notebook execution exited with code {watch_res.returncode}.")
        print("Fetching error log summary:")
        subprocess.run(["gh", "run", "view", str(run_id), "--log-failed"])
        sys.exit(watch_res.returncode)


if __name__ == "__main__":
    main()
