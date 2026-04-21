#!/usr/bin/env python3
"""
PR Reviewer using Claude API (Anthropic) for vllm-omni project.
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, TypedDict

import requests


class PRDetails(TypedDict):
    title: str
    body: str
    number: int
    state: str
    user: dict[str, Any]


class GitHubComment(TypedDict):
    id: int
    body: str
    created_at: str
    user: dict[str, Any]


TRIGGER_PHRASE: str = "@vllm-omni-reviewer"
DEFAULT_CLAUDE_MODEL: str = "claude-sonnet-4-6"
DEFAULT_COOLDOWN_MINUTES: int = 5
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RETRY_DELAY: float = 1.0
MAX_DIFF_SIZE: int = 100_000


@dataclass
class Config:
    claude_model: str
    cooldown_minutes: int
    max_retries: int
    retry_delay: float
    max_diff_size: int


logging.basicConfig(
    level=logging.INFO,
    format="[PR Reviewer] %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)


def get_config() -> Config:
    return Config(
        claude_model=os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
        cooldown_minutes=int(os.getenv("PR_REVIEWER_COOLDOWN_MINUTES", str(DEFAULT_COOLDOWN_MINUTES))),
        max_retries=int(os.getenv("PR_REVIEWER_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))),
        retry_delay=float(os.getenv("PR_REVIEWER_RETRY_DELAY", str(DEFAULT_RETRY_DELAY))),
        max_diff_size=int(os.getenv("PR_REVIEWER_MAX_DIFF_SIZE", str(MAX_DIFF_SIZE))),
    )


def get_env_var(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        logger.error(f"Environment variable {name} is not set")
        sys.exit(1)
    return value


def check_trigger(comment_body: str) -> bool:
    return TRIGGER_PHRASE in comment_body


def fetch_pr_diff(repo_name: str, pr_number: int, token: str, max_size: int = MAX_DIFF_SIZE) -> str | None:
    url = f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3.diff"}
    logger.info(f"Fetching PR diff from {url}")
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code == 200:
        diff = response.text
        if len(diff) > max_size:
            logger.warning(f"Diff size ({len(diff)} bytes) exceeds maximum ({max_size} bytes), truncating")
            return diff[:max_size] + "\n\n... [Diff truncated due to size] ..."
        logger.info(f"Successfully fetched diff ({len(diff)} bytes)")
        return diff
    logger.error(f"Failed to fetch PR diff: {response.status_code}\n{response.text}")
    return None


def fetch_pr_details(repo_name: str, pr_number: int, token: str) -> PRDetails | None:
    url = f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    logger.info(f"Fetching PR details from {url}")
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code == 200:
        return response.json()
    logger.error(f"Failed to fetch PR details: {response.status_code}")
    return None


def fetch_pr_review_comments(repo_name: str, pr_number: int, token: str) -> list[dict[str, Any]]:
    url = f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}/comments"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code == 200:
        return response.json()
    logger.warning(f"Failed to fetch review comments: {response.status_code}")
    return []


def build_review_prompt(
    pr_title: str,
    pr_description: str,
    diff: str,
    review_comments: list[dict[str, Any]],
) -> str:
    comments_section = ""
    if review_comments:
        comments_text = "\n".join(
            f"- [{c['path']}:{c.get('line', '?')}] {c['user']['login']}: {c['body']}"
            for c in review_comments
        )
        comments_section = f"""
## Existing Review Comments
The following inline review comments have already been posted on this PR.
Please address each one specifically in your response:

{comments_text}
"""

    return f"""You are an expert code reviewer for the vLLM-Omni project — a multi-stage \
heterogeneous inference framework built on vLLM that supports AR (autoregressive) and \
Diffusion (DiT) stages, inter-stage KV cache transfer, and omni-modal generation.

Please review the following pull request:

## Pull Request Details
**Title:** {pr_title}

**Description:**
{pr_description if pr_description else "No description provided."}
{comments_section}
## Code Changes (Diff)
{diff}

## Review Guidelines

Please provide a comprehensive code review with the following sections:

### 1. Overview
- Brief summary of the changes
- Overall assessment (positive, neutral, or concerns)

### 2. Code Quality
- Code style and consistency with the existing codebase
- Potential bugs or edge cases
- Performance considerations
- Error handling

### 3. Architecture & Design
- Integration with the multi-stage pipeline (AR stage, DiT stage, orchestrator)
- Correctness of KV cache layout and inter-stage transfers (if applicable)
- Design patterns and best practices

### 4. Response to Existing Review Comments
- If there are existing review comments above, address each one explicitly.
- State whether each concern is valid, already fixed, or not applicable.

### 5. Specific Suggestions
- Concrete actionable feedback using `file:line` format
- Code examples for improvements where helpful

### 6. Approval Status
- **LGTM** if the PR is ready to merge
- **LGTM with suggestions** if good but has minor suggestions
- **Changes requested** if significant changes are needed

Be constructive, focus on objective technical feedback, and acknowledge good practices.
Format your response in Markdown with clear section headers.
"""


def call_claude_api(prompt: str, api_key: str, config: Config) -> str | None:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": config.claude_model,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
    }
    last_error: str | None = None

    for attempt in range(config.max_retries):
        try:
            logger.info(f"Calling Claude API ({config.claude_model}) - Attempt {attempt + 1}/{config.max_retries}")
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=120,
            )
            if response.status_code == 200:
                data = response.json()
                text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
                if text_blocks:
                    review = "\n".join(text_blocks)
                    logger.info(f"Successfully received review ({len(review)} chars)")
                    return review
                last_error = "Claude API returned no text content"
                logger.error(f"{last_error}: {json.dumps(data, indent=2)}")
            else:
                last_error = f"Claude API request failed: {response.status_code} - {response.text}"
                logger.error(last_error)
        except requests.exceptions.Timeout:
            last_error = f"Claude API request timed out (attempt {attempt + 1})"
            logger.error(last_error)
        except requests.exceptions.RequestException as e:
            last_error = f"Claude API request exception: {e}"
            logger.error(last_error)
        except (json.JSONDecodeError, KeyError) as e:
            last_error = f"Failed to parse Claude API response: {e}"
            logger.error(last_error)

        if attempt < config.max_retries - 1:
            wait_time = config.retry_delay * (2**attempt)
            logger.info(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)

    logger.error(f"All {config.max_retries} attempts failed. Last error: {last_error}")
    return None


def check_cooldown(repo_name: str, pr_number: int, token: str, cooldown_minutes: int) -> bool:
    from datetime import datetime, timedelta

    url = f"https://api.github.com/repos/{repo_name}/issues/{pr_number}/comments"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    logger.info(f"Checking cooldown period ({cooldown_minutes} minutes)")
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code != 200:
        logger.warning(f"Failed to check cooldown: {response.status_code}, proceeding with review")
        return False

    comments: list[dict[str, Any]] = response.json()
    cutoff_time = datetime.utcnow() - timedelta(minutes=cooldown_minutes)

    for comment in reversed(comments):
        body = comment.get("body", "")
        if "VLLM-Omni PR Review" in body or "PR Reviewer Bot" in body:
            created_at_str = comment.get("created_at", "")
            try:
                created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                created_at = created_at.replace(tzinfo=None)
                if created_at > cutoff_time:
                    logger.info(f"PR is within cooldown period (last review: {created_at_str})")
                    return True
            except ValueError:
                logger.warning(f"Failed to parse comment timestamp: {created_at_str}")
                continue

    logger.info("PR is outside cooldown period, proceeding with review")
    return False


def post_review_comment(repo_name: str, pr_number: int, token: str, review: str, model: str) -> bool:
    url = f"https://api.github.com/repos/{repo_name}/issues/{pr_number}/comments"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3+json"}
    comment_body = f"""## 🤖 VLLM-Omni PR Review

{review}

---
*This review was generated automatically by the VLLM-Omni PR Reviewer Bot using {model}.*
"""
    logger.info(f"Posting review comment to PR #{pr_number}")
    response = requests.post(url, headers=headers, json={"body": comment_body}, timeout=30)
    if response.status_code == 201:
        logger.info("Successfully posted review comment")
        return True
    logger.error(f"Failed to post comment: {response.status_code}\n{response.text}")
    return False


def main() -> int:
    logger.info("VLLM-Omni PR Reviewer Bot starting...")
    config = get_config()
    logger.info(f"Configuration: model={config.claude_model}, cooldown={config.cooldown_minutes}min")

    token = get_env_var("GITHUB_TOKEN")
    api_key = get_env_var("ANTHROPIC_API_KEY")
    repo_name = get_env_var("REPO_NAME")
    pr_number_str = get_env_var("PR_NUMBER")
    comment_body = get_env_var("COMMENT_BODY")

    try:
        pr_number = int(pr_number_str)
    except ValueError:
        logger.error(f"Invalid PR number: {pr_number_str}")
        return 1

    logger.info(f"Repository: {repo_name}, PR: #{pr_number}")

    if not check_trigger(comment_body):
        logger.info(f"Comment does not contain trigger phrase '{TRIGGER_PHRASE}', exiting")
        return 0

    logger.info("Trigger phrase detected! Starting review process...")

    if check_cooldown(repo_name, pr_number, token, config.cooldown_minutes):
        logger.info("Skipping review due to cooldown period")
        return 0

    logger.info("Step 1/4: Fetching PR details...")
    pr_details = fetch_pr_details(repo_name, pr_number, token)
    if not pr_details:
        logger.error("Failed to fetch PR details")
        return 1
    pr_title = pr_details.get("title", "Unknown")
    pr_description = pr_details.get("body", "") or ""
    logger.info(f"PR Title: {pr_title}")

    logger.info("Step 2/4: Fetching PR diff and review comments...")
    diff = fetch_pr_diff(repo_name, pr_number, token, config.max_diff_size)
    if diff is None:
        logger.error("Failed to fetch PR diff")
        return 1
    review_comments = fetch_pr_review_comments(repo_name, pr_number, token)
    logger.info(f"Found {len(review_comments)} existing review comments")

    logger.info("Step 3/4: Building review prompt...")
    prompt = build_review_prompt(pr_title, pr_description, diff, review_comments)

    logger.info("Step 4/4: Calling Claude API...")
    review = call_claude_api(prompt, api_key, config)
    if not review:
        logger.error("Failed to get review from Claude API")
        return 1

    logger.info("Posting review comment...")
    if not post_review_comment(repo_name, pr_number, token, review, config.claude_model):
        logger.error("Failed to post review comment")
        return 1

    logger.info("PR review completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
