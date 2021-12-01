import subprocess
from subprocess import PIPE
from typing import List, Optional, Union


def run_command(
    command: str,
    return_stdout: bool = False,
    return_exit_status: bool = False,
    fail_on_bad_status: bool = True,
) -> Optional[Union[str, int]]:
    """Run a command using subprocess."""
    print(f"Running command: {command} ({command.split(' ')})")
    process = subprocess.run(command.split(" "), stdout=PIPE, stderr=PIPE)
    # Check exit status if necessary
    if fail_on_bad_status and process.returncode:
        raise AssertionError(
            f"Command '{command}' failed with: {process.stderr.decode('utf8')}"
        )
    # Otherwise see if we should return something from the process
    if return_stdout:
        return process.stdout.decode("utf8")
    elif return_exit_status:
        return process.returncode
    else:
        return None


def get_sorted_tags_list() -> List[str]:
    """Get a list of sorted tags in descending order from the repository."""
    stdout = run_command("git tag -l --sort=-v:refname", return_stdout=True)
    assert isinstance(stdout, str), "Error getting tags"
    return stdout.strip().split("\n")


def get_branch_contents(remote: str, branch: str) -> List[str]:
    """Get the list of directories in a branch."""
    stdout = run_command(
        f"git ls-tree -d --name-only {remote}/{branch}", return_stdout=True
    )
    assert isinstance(
        stdout, str
    ), f"Error getting directory contents of {remote}/{branch}"
    return stdout.strip().split("\n")


def get_docs_url(github_org: str, project: str, version: str):
    """Get the URL for a specific build of the project."""
    return f"https://{github_org}.github.io/{project}/{version}"


def get_version_file_string(
    main_branches: List[str], branches: List[str], releases: List[str],
) -> str:
    """Create the string for the RST file."""
    file_str = ""
    # Add branches first
    for version in main_branches:
        file_str += f"{version}\n"
    for version in branches:
        file_str += f"{version}\n"
    # Add releases
    for version in releases:
        file_str += f"{version}\n"
    return file_str


def push_file_to_GitHub(file_string: str, filename: str, remote: str, branch: str):
    """Push the version file to the GitHub Pages branch."""
    # Check out a local branch tracking the GitHub Pages branch
    run_command(f"git checkout -tq {remote}/{branch}")

    try:
        # Write the file
        with open(filename, "w") as output_file:
            output_file.write(file_string)

        # Check if file is already tracked
        status = run_command(
            f"git ls-files --error-unmatch {filename}",
            return_exit_status=True,
            fail_on_bad_status=False,
        )

        # Status of zero means tracked. So check if file has changed
        if not status:
            status = run_command(
                f"git diff --exit-code {filename}", return_exit_status=True
            )

        # If status is one (either untracked or file has changed) then commit the file
        if status:
            print(f"File {filename} being commited to {remote}/{branch}.")
            # Add the file
            run_command(f"git add {filename}")

            # Commit the file
            run_command("git commit -qm Update-version-file")

            # Push to remote
            run_command("git push -q")
        else:
            print(f"File {filename} has not been updated. Not committing.")

    finally:
        # Revert back to original branch
        run_command("git checkout -q -")

        # Delete local branch
        run_command(f"git branch -q -D {branch}")


def create_gh_pages_versions_file():
    """Generate the file containing the list of all GitHub Pages builds."""
    # General information
    remote = "origin"
    # TODO: replace branch with actual gh-pages branch
    github_pages_branch = "test/gh-pages-file"
    version_filename = "version.txt"

    # Store the builds into different groups
    main_branches: List[str] = []
    releases: List[str] = []
    branches: List[str] = []

    # Get the list of sorted tags
    tags = get_sorted_tags_list()

    # Get the directories (i.e. builds) from the GitHub Pages branch
    all_build_versions = get_branch_contents(remote, github_pages_branch)

    # Parse all builds, storing releases in an unordered dict
    unsorted_releases: List[str] = []
    for version in all_build_versions:
        # Parse the version
        if version[0].isdigit():
            unsorted_releases.append(version)
        elif version in ["master", "main"]:
            main_branches.append(version)
        else:
            branches.append(version)

    # Use output from git tag to sort the releases
    for tag in tags:
        if tag in unsorted_releases:
            releases.append(tag)

    # Get version file string
    version_file_string = get_version_file_string(main_branches, branches, releases)

    # Push version file to GitHub Pages branch
    push_file_to_GitHub(
        version_file_string, version_filename, remote, github_pages_branch
    )


if __name__ == "__main__":
    create_gh_pages_versions_file()
