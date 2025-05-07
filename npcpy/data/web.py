# search.py

import requests
import os

from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException

try:
    from googlesearch import search
except:
    pass
from typing import List, Dict, Any, Optional, Union
import numpy as np
import json

try:
    from sentence_transformers import util, SentenceTransformer
except:
    pass


def search_perplexity(
    query: str,
    api_key: str = None,
    model: str = "sonar",
    max_tokens: int = 400,
    temperature: float = 0.2,
    top_p: float = 0.9,
):
    if api_key is None:
        api_key = os.environ["PERPLEXITY_API_KEY"]
    # print("api_key", api_key)
    url = "https://api.perplexity.ai/chat/completions"
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": query},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1,
        "response_format": None,
    }

    # Headers for the request, including the Authorization bearer token
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Make the POST request to the API
    response = requests.post(url, json=payload, headers=headers)
    response = json.loads(response.text)
    #print(response)
    return [response["choices"][0]["message"]["content"], response["citations"]]


def search_web(
    query: str,
    num_results: int = 5,
    provider: str=None,
    api_key=None,
    perplexity_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """
    Function Description:
        This function searches the web for information based on a query.
    Args:
        query: The search query.
    Keyword Args:
        num_results: The number of search results to retrieve.
        provider: The search engine provider to use ('google' or 'duckduckgo').
    Returns:
        A list of dictionaries with 'title', 'link', and 'content' keys.
    """
    if perplexity_kwargs is None:
        perplexity_kwargs = {}
    results = []
    if provider is None:
        provider = 'duckduckgo'

    if provider == "perplexity":
        search_result = search_perplexity(query, api_key=api_key, **perplexity_kwargs)
        # print(search_result, type(search_result))
        return search_result

    if provider == "duckduckgo":
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0"
        }
        ddgs = DDGS(headers=headers)

        try:
            search_results = ddgs.text(query, max_results=num_results)
            urls = [r["href"] for r in search_results]
            results = [
                {"title": r["title"], "link": r["href"], "content": r["body"]}
                for r in search_results
            ]
        except DuckDuckGoSearchException as e:
            print("DuckDuckGo search failed: ", e)
            urls = []
            results = []

    elif provider =='google':  # google
        urls = list(search(query, num_results=num_results))
        # google shit doesnt seem to be working anymore, apparently a lbock they made on browsers without js?
        #print("urls", urls)
        #print(provider)
        for url in urls:
            try:
                # Fetch the webpage content
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                response = requests.get(url, headers=headers, timeout=5)
                response.raise_for_status()

                # Parse with BeautifulSoup
                soup = BeautifulSoup(response.text, "html.parser")

                # Get title and content
                title = soup.title.string if soup.title else url

                # Extract text content and clean it up
                content = " ".join([p.get_text() for p in soup.find_all("p")])
                content = " ".join(content.split())  # Clean up whitespace

                results.append(
                    {
                        "title": title,
                        "link": url,
                        "content": (
                            content[:500] + "..." if len(content) > 500 else content
                        ),
                    }
                )

            except Exception as e:
                print(f"Error fetching {url}: {str(e)}")
                continue

    # except Exception as e:
    #    print(f"Search error: {str(e)}")
    content_str = "\n".join(
        [r["content"] + "\n Citation: " + r["link"] + "\n\n\n" for r in results]
    )
    link_str = "\n".join([r["link"] + "\n" for r in results])
    return [content_str, link_str]

