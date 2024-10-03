import subprocess
import argparse

def run_gitlab_scrapping(path, allow_parsing):
    """Run the GitLab scrapping script."""
    cmd = [
        "python",
        "scripts/gitlab_scrapping.py",
        "--path", path
    ]
    if allow_parsing:
        cmd.append("--allow_parsing")
    try:
        result = subprocess.run(cmd, check=True)
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"GitLab scrapping failed: {e}")
        print(f"Command output: {e.output}")
        print(f"Command stderr: {e.stderr}")

def run_mattermost_scrapping(save_path="./data/mattermost_2.0/"):
    """Run the Mattermost scrapping script."""
    cmd = ["python", "scripts/mattermost_scrapping.py", "--save_path", save_path]
    try:
        result = subprocess.run(cmd, check=True)
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Mattermost scrapping failed: {e}")
        print(f"Command output: {e.output}")
        print(f"Command stderr: {e.stderr}")

def run_happeo_scrapping(happeo_path):
    try:
        cmd = ['python', 'scripts/happeo_scrapping.py', '--directory', happeo_path]
        result = subprocess.run(cmd, check=True)
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Happeo scrapping failed: {e}")
        print(f"Command output: {e.output}")
        print(f"Command stderr: {e.stderr}")

def main():
    parser = argparse.ArgumentParser(description="Launch scrapping for GitLab, Mattermost, and Happeo.")
    parser.add_argument("--gitlab_path", type=str, default="data_test/gitlab", help="Path to save GitLab README files")
    parser.add_argument("--allow_parsing", type=bool, default=False, help="Allow parsing of GitLab README files")
    parser.add_argument("--mattermost_path", type=str, default="data_test/mattermost", help="Path to save Mattermost messages")
    parser.add_argument("--happeo_path", type=str, default="data_test/happeo", help="Path to save Happeo pages")
    args = parser.parse_args()

    print("Starting GitLab scrapping...")
    run_gitlab_scrapping(args.gitlab_path, args.allow_parsing)
    print("GitLab scrapping completed.")
    
    print("Starting Happeo scrapping...")
    run_happeo_scrapping(args.happeo_path)
    print("Happeo scrapping completed.")

    print("Starting Mattermost scrapping...")
    run_mattermost_scrapping(args.mattermost_path)
    print("Mattermost scrapping completed.")

if __name__ == "__main__":
    main()