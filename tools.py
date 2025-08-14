from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    """Save research data to a text file with timestamp."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"
        
        with open(filename, "a", encoding="utf-8") as f:
            f.write(formatted_text)
            
        logger.info(f"Data saved to {filename}")
        return f"Data successfully saved to {filename}"
    except Exception as e:
        logger.error(f"Error saving to file {filename}: {str(e)}")
        return f"Error saving to file: {str(e)}"

# Initialize tools
try:
    save_tool = Tool(
        name="save_text_to_file",
        func=save_to_txt,
        description="Saves structured research data to a text file with timestamp."
    )

    search = DuckDuckGoSearchRun(max_results=5)
    search_tool = Tool(
        name="web_search",
        func=search.run,
        description="Search the web for up-to-date information using DuckDuckGo."
    )

    api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=500)
    wiki_tool = WikipediaQueryRun(
        api_wrapper=api_wrapper,
        description="Query Wikipedia for concise, reliable information."
    )

except Exception as e:
    logger.error(f"Error initializing tools: {str(e)}")
    raise

def export_to_json(data: dict, filename: str = "research_output.json") -> str:
    """Export research data to JSON format."""
    try:
        with open(filename, "a", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        logger.info(f"Data exported to {filename}")
        return f"Data successfully exported to {filename}"
    except Exception as e:
        logger.error(f"Error exporting to JSON {filename}: {str(e)}")
        return f"Error exporting to JSON: {str(e)}"

export_tool = Tool(
    name="export_to_json",
    func=export_to_json,
    description="Exports structured research data to a JSON file."
)