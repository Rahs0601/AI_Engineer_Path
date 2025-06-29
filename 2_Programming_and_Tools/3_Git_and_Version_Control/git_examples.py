# Git and Version Control Examples (Conceptual - Using subprocess to run Git commands)

import subprocess

def run_git_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, cwd='.') # cwd='.' means current directory
        print(f"Command: git {command}\nStdout:\n{result.stdout}\nStderr:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: git {command}\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}")
    except FileNotFoundError:
        print(f"Error: 'git' command not found. Is Git installed and in your PATH?")

print("--- Git Conceptual Examples ---")
print("Note: These commands will interact with your local Git repository.")
print("They are executed via subprocess, simulating command line usage.")

# Example 1: Check Git version
print("\nChecking Git version...")
run_git_command("--version")

# Example 2: (Conceptual) Initialize a new Git repository
# This would create a .git directory in the current location.
# Be careful if you run this in an existing repo.
print("\n(Conceptual) Initializing a new Git repository:")
print("run_git_command(\"init\")")

# Example 3: (Conceptual) Check Git status
print("\nChecking Git status...")
run_git_command("status")

# Example 4: (Conceptual) Add a file to staging
# This assumes 'example.txt' exists in the current directory.
print("\n(Conceptual) Adding 'example.txt' to staging:")
print("run_git_command(\"add example.txt\")")

# Example 5: (Conceptual) Commit changes
print("\n(Conceptual) Committing changes:")
print("run_git_command(\"commit -m 'Initial commit'\")")

# Example 6: (Conceptual) View commit history
print("\n(Conceptual) Viewing commit history:")
print("run_git_command(\"log --oneline\")")

# Example 7: (Conceptual) Create a new branch
print("\n(Conceptual) Creating a new branch 'feature-branch':")
print("run_git_command(\"branch feature-branch\")")

# Example 8: (Conceptual) Switch to a branch
print("\n(Conceptual) Switching to 'feature-branch':")
print("run_git_command(\"checkout feature-branch\")")

# Example 9: (Conceptual) Pull latest changes from remote
print("\n(Conceptual) Pulling latest changes from remote:")
print("run_git_command(\"pull origin main\")")

# Example 10: (Conceptual) Push changes to remote
print("\n(Conceptual) Pushing changes to remote:")
print("run_git_command(\"push origin feature-branch\")")
