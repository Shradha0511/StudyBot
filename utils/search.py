import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import TAVILY_API_KEY


def web_search(query: str) -> str:
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(query=query, max_results=3)

        if not results or not results.get("results"):
            return "No web results found."

        formatted = []
        for i, result in enumerate(results["results"], 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            content = result.get("content", "No content")
            formatted.append(f"[Web Result {i}] {title}\nURL: {url}\n{content}")

        return "\n\n".join(formatted)

    except ImportError:
        return "Web search unavailable: install tavily-python (pip install tavily-python)"
    except Exception as e:
        return f"Web search failed: {str(e)}"