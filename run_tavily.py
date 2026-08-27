import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

response = client.search(
    query="Visa chargeback dispute policy for undelivered goods",
    max_results=3,
)
import json
print(json.dumps(response, indent=2))