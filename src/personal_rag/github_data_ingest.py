import os
import json
import base64
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

from config import GITHUB_DATA_DIR, GITHUB_USERNAME

load_dotenv()

session = requests.Session()
session.headers.update({
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
})

token = os.getenv("GITHUB_TOKEN")
if token:
    session.headers["Authorization"] = f"Bearer {token}"


def get_json(url, params=None):
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def fetch_github_profile(username: str) -> dict:
    return get_json(f"https://api.github.com/users/{username}")


def fetch_github_repositories(username: str) -> list[dict]:
    url = f"https://api.github.com/users/{username}/repos"
    repositories = []
    page = 1

    while True:
        batch = get_json(
            url,
            params={
                "per_page": 100,
                "page": page,
                "sort": "updated",
                "type": "owner",
            },
        )

        if not batch:
            break

        repositories.extend(batch)
        page += 1

    return repositories


def fetch_repository_content(
    owner: str,
    repo: dict,
    max_files: int = 30,
    max_file_size: int = 100_000,
) -> dict:
    """Fetch repository metadata, README, and selected source files."""
    name = repo["name"]
    base_url = f"https://api.github.com/repos/{owner}/{name}"

    result = {
        "metadata": repo,
        "readme": None,
        "source_files": [],
    }

    # README
    readme_response = session.get(
        f"{base_url}/readme",
        timeout=30,
    )

    if readme_response.status_code == 200:
        readme = readme_response.json()
        result["readme"] = {
            "path": readme["path"],
            "url": readme["html_url"],
            "content": base64.b64decode(
                readme["content"]
            ).decode("utf-8"),
        }
    elif readme_response.status_code != 404:
        readme_response.raise_for_status()

    # Repository file tree
    branch = repo["default_branch"]
    tree_url = (
        f"{base_url}/git/trees/{quote(branch, safe='')}"
    )
    tree_data = get_json(tree_url, {"recursive": "1"})

    if tree_data.get("truncated"):
        print(f"Warning: GitHub returned a truncated tree for {name}")

    extensions = {
        ".py", ".cs", ".java", ".cpp", ".c", ".h", ".hpp",
        ".js", ".ts", ".sql", ".sh",
    }

    excluded_dirs = {
        ".git", ".venv", "venv", "node_modules",
        "__pycache__", "dist", "build",
    }

    excluded_files = {
        ".env", "secrets.json", "credentials.json",
    }

    for item in tree_data.get("tree", []):
        if len(result["source_files"]) >= max_files:
            break

        if item["type"] != "blob":
            continue

        path = item["path"]
        parts = path.split("/")
        filename = parts[-1]

        extension = (
            "." + filename.rsplit(".", 1)[-1].lower()
            if "." in filename else ""
        )

        if any(part.lower() in excluded_dirs for part in parts):
            continue

        if filename.lower() in excluded_files:
            continue

        if extension not in extensions:
            continue

        if item.get("size", 0) > max_file_size:
            continue

        raw_url = (
            f"https://raw.githubusercontent.com/{owner}/{name}/"
            f"{quote(branch, safe='')}/{quote(path, safe='/')}"
        )

        try:
            response = session.get(raw_url, timeout=30)
            response.raise_for_status()
            content = response.content.decode("utf-8")
        except (requests.RequestException, UnicodeDecodeError):
            continue

        if not content.strip():
            continue

        result["source_files"].append({
            "path": path,
            "url": (
                f"{repo['html_url']}/blob/{quote(branch, safe='')}/"
                f"{quote(path, safe='/')}"
            ),
            "content": content,
        })

    return result


def ingest_github(
    username: str = GITHUB_USERNAME,
    output_dir: str = GITHUB_DATA_DIR,
    max_files_per_repo: int = 30,
    force_fetch: bool = False,
) -> Path:
    """Save a GitHub profile and its owned public repositories locally."""
    root = Path(output_dir) / username
    root.mkdir(parents=True, exist_ok=True)

    # Save profile
    profile_path = root / "profile.json"

    if profile_path.exists() and not force_fetch:
        print("Profile already exists; skipping download.")
    else:
        profile = fetch_github_profile(username)
        save_json(profile_path, profile)

    # Retrieve repositories
    repositories = fetch_github_repositories(username)
    repos_dir = root / "repositories"
    repos_dir.mkdir(parents=True, exist_ok=True)

    # Fetch and save each repository
    for index, repo in enumerate(repositories, start=1):
        repo_path = repos_dir / f"{repo['name']}.json"

        if repo_path.exists() and not force_fetch:
            print(f"[{index}/{len(repositories)}] Skipping {repo['name']} (already exists)")
            continue
        try:
            data = fetch_repository_content(
                username,
                repo,
                max_files=max_files_per_repo,
            )

            save_json(
                repos_dir / f"{repo['name']}.json",
                data,
            )

            print(f"[{index}/{len(repositories)}] Saved {repo['name']}")

        except requests.RequestException as error:
            print(f"Failed to fetch {repo['name']}: {error}")

    print(f"GitHub ingestion finished: {root.resolve()}")
    return root
