
"""
web_utils.py

Utility functions to fetch and clean text from web pages for real-time crawling.
"""

import requests
from bs4 import BeautifulSoup


def fetch_legal_webpage(url: str) -> str:
    """
    Fetches and returns visible text from the given URL.
    """
    try:
        response = requests.get(url, timeout=8)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract visible text from <p> tags
        paragraphs = soup.find_all("p")
        text = "\n".join(p.get_text().strip() for p in paragraphs if p.get_text())
        return text[:3000]  # limit length for prompt
    except Exception as e:
        return f"Failed to fetch content: {e}"