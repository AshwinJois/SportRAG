import requests
from langchain.tools import tool
from bs4 import BeautifulSoup

@tool
def get_series_update():
    """Fetch a clean summary of current cricket matches series"""
    print("--- Getting series updates ---")
    try:
        # Use a timeout to avoid hanging on a bad request
        response = requests.get("https://api.cricapi.com/v1/series?apikey=8a1ac602-557f-4801-9bac-e87fd31dc882&offset=0", timeout=10)
        response.raise_for_status()
        raw_data = response.json()
        match_list = raw_data.get("data", [])
        if not match_list:
            return "No match data available right now."
        clean_output = []
        for match in match_list[8:]:  # Limit to 8 matches
            name = match.get('name', 'Unknown')
            start_date = match.get('startDate', 'Unknown')
            end_date = match.get('endDate', 'No status')
            clean_output.append(f"• {name}, starts from {start_date} till {end_date}")
        final_string = "\n".join(clean_output)
        return final_string
    except Exception as e:
        return f"Tool failed to fetch data: {str(e)}"

@tool
def get_match_update():
    """Fetch the recent match results and summary"""
    print("--- Getting Match updates ---")
    try:
        # Use a timeout to avoid hanging on a bad request
        response = requests.get("https://api.cricapi.com/v1/currentMatches?apikey=8a1ac602-557f-4801-9bac-e87fd31dc882&offset=0", timeout=10)
        response.raise_for_status()
        raw_data = response.json()
        match_list = raw_data.get("data", [])
        if not match_list:
            return "No match data available right now."
        # Create a small, clean string. DO NOT return the whole list.
        clean_output = []
        for match in match_list[8:]:  # Limit to 8 matches
            name = match.get('name', 'Unknown')
            status = match.get('status', 'Unknown')
            date = match.get('date', 'No status')
            clean_output.append(f"• {name}, match result: {status}, on the date: {date}")
        final_string = "\n".join(clean_output)
        # This print will show you exactly what is being sent to the LLM
        #print(f"Tool sending to LLM: {final_string}")
        return final_string
    except Exception as e:
        return f"Tool failed to fetch data: {str(e)}"

def get_iconic_cricket_stadiums() -> dict:
    """
    Scrapes a blog page listing the most iconic cricket stadiums in the world.

    This tool retrieves the names and descriptions of famous cricket stadiums
    from the MysteryCricket blog page.

    Use this tool whenever a user asks about:
    - best cricket stadiums
    - iconic cricket stadiums
    - famous cricket grounds
    - cricket venues around the world

    Returns:
        A dictionary containing stadium names and short descriptions.
    """
    print("--- Getting Stadium updates ---")
    url = "https://mysterycricket.com/blogs/the-mystery-cricket-blog/most-iconic-cricket-stadiums-in-the-world"
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    stadiums = []
    # Each stadium title is in an H2 tag
    for tag in soup.find_all("h2"):
        title = tag.get_text(strip=True)
        # get description paragraphs after title
        desc = []
        next_node = tag.find_next_sibling()
        while next_node and next_node.name == "p":
            desc.append(next_node.get_text(strip=True))
            next_node = next_node.find_next_sibling()
        stadiums.append({
            "stadium": title,
            "description": " ".join(desc)
        })
    return {
        "source": url,
        "stadiums": stadiums
    }