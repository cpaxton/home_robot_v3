# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from pathlib import Path

import git


def get_git_repo() -> git.Repo | None:
    path = str(Path(__file__).resolve().parent)

    repo_names = ["stretch_ai", "stretchpy", "app"]

    # Find the root of the repo given __file__
    for name in repo_names:
        idx = path.find(name)
        if idx != -1:
            idx += len(name)
            break

    repo_root = path[:idx]
    try:
        return git.Repo(repo_root)
    except git.exc.InvalidGitRepositoryError:
        return None


def get_git_branch() -> str | None:
    repo = get_git_repo()
    if repo is None:
        return None
    return str(repo.active_branch)


def get_git_commit() -> str | None:
    repo = get_git_repo()
    if repo is None:
        return None
    return str(repo.head.commit.hexsha)


def get_git_commit_message() -> str | None:
    repo = get_git_repo()
    if repo is None:
        return None
    return str(repo.head.commit.message)


if __name__ == "__main__":
    print("Repo:", get_git_repo())
    print("Branch:", get_git_branch())
    print("Commit:", get_git_commit())
    print("Message:", get_git_commit_message())
