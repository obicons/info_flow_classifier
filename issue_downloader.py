import json
import requests
from typing import Iterator, Sequence

_GITHUB_API_URL = 'https://api.github.com/repos/{}/{}/issues?state=closed&page={}&per_page=30'


def _download_issues_on_page(repo_owner: str, repo_name: str, page: int) -> Sequence[str]:
    """
    Downloads the issues from the Github repository repo_name, owned by repo_owner.

    Args:
        repo_owner: The owner of the Github repository. For example, servo.
        repo_name: The name of the Github repository. For example, servo.
        page: The page to start downloading from.

    Returns:
        A sequence of issue bodies.
    """
    url = _GITHUB_API_URL.format(repo_owner, repo_name, page)
    response = requests.get(url)
    if response.status_code != requests.codes['ok']:
        return []
    issues = json.loads(response.content)
    return [issue['body'] for issue in issues]


def download_issues(repo_owner: str, repo_name: str) -> Iterator[str]:
    """
    Downloads the issues from the Github repository repo_name, owned by repo_owner.

    Args:
        repo_owner: The owner of the Github repository. For example, servo.
        repo_name: The name of the Github repository. For example, servo.

    Returns:
        A generator that yields the bodies of the Github issues.
    """
    issues = _download_issues_on_page(repo_owner, repo_name, 0)
    current_page = 1
    while issues:
        next_item = issues[0]
        issues = issues[1:]
        if not issues:
            issues = _download_issues_on_page(
                repo_owner,
                repo_name,
                current_page,
            )
            current_page += 1
        yield next_item
