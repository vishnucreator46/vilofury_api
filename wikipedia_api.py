import requests
from urllib.parse import quote

def search_wikipedia(query):
    """Search Wikipedia for a query and return the most relevant page title."""
    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": 1
        }
        headers = {"User-Agent": "VilofuryAI/1.0 (vishnuprakash@kalasalingam.ac.in)"}
        
        response = requests.get(search_url, params=search_params, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        if data["query"]["search"]:
            return data["query"]["search"][0]["title"]
        return None
    except Exception as e:
        print(f"Search error: {e}")
        return None

def get_wikipedia_summary(query):
    """Get a Wikipedia summary for the given query."""
    try:
        # First, search for the most relevant page
        page_title = search_wikipedia(query)
        if not page_title:
            return None
            
        # Then get the summary using the found page title
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(page_title)}"
        headers = {"User-Agent": "VilofuryAI/1.0 (vishnuprakash@kalasalingam.ac.in)"}
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()

        if 'extract' in data:
            # Check if it's a disambiguation page
            if 'disambiguation' in data.get('type', ''):
                return "This topic has multiple meanings. Could you please be more specific?"
            return data['extract']
        else:
            return None
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return None
        return f"I couldn't access Wikipedia at the moment. Please try again later."
    except Exception as e:
        print(f"Wikipedia error: {e}")  # For debugging
        return None


