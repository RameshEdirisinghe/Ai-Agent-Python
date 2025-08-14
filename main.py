import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

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

# Load environment variables
load_dotenv()

class ResearchResponse(BaseModel):
    topic: str = Field(description="The main topic of the research")
    summary: str = Field(description="A concise summary of the research findings")
    sources: list[str] = Field(description="List of source URLs or references")
    tools_used: list[str] = Field(description="List of tools used in the research")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

async def create_research_agent() -> AgentExecutor:
    """Create and configure the research agent."""
    try:
        # Initialize LLM (using Anthropic as primary, with OpenAI as fallback)
        primary_llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens=2000
        )
        
        parser = PydanticOutputParser(pydantic_object=ResearchResponse)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an expert research assistant designed to generate high-quality research outputs.
                    Use the provided tools to gather accurate information and format the response according to the specified schema.
                    Ensure all sources are credible and properly cited.
                    Wrap the output in the format: {format_instructions}
                    """
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{query}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        ).partial(format_instructions=parser.get_format_instructions())

        tools = [search_tool, wiki_tool, save_tool]
        agent = create_tool_calling_agent(
            llm=primary_llm,
            prompt=prompt,
            tools=tools
        )

        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise

async def process_research_query(query: str) -> Dict[str, Any]:
    """Process a research query and return structured response."""
    try:
        agent_executor = await create_research_agent()
        logger.info(f"Processing query: {query}")
        
        raw_response = await asyncio.to_thread(
            agent_executor.invoke,
            {"query": query}
        )
        
        parser = PydanticOutputParser(pydantic_object=ResearchResponse)
        structured_response = parser.parse(raw_response.get("output")[0]["text"])
        
        logger.info(f"Successfully processed query: {query}")
        return structured_response.dict()
    
    except Exception as e:
        logger.error(f"Error processing query '{query}': {str(e)}")
        return {
            "error": str(e),
            "raw_response": raw_response if 'raw_response' in locals() else None
        }

async def main():
    """Main entry point for the research assistant."""
    try:
        query = input("What can I help you research? ")
        if not query.strip():
            logger.warning("Empty query provided")
            print("Please provide a valid research query.")
            return
            
        result = await process_research_query(query)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            if result['raw_response']:
                print(f"Raw Response: {result['raw_response']}")
        else:
            print("\nResearch Results:")
            print(f"Topic: {result['topic']}")
            print(f"Summary: {result['summary']}")
            print("Sources:")
            for source in result['sources']:
                print(f"- {source}")
            print(f"Tools Used: {', '.join(result['tools_used'])}")
            print(f"Timestamp: {result['timestamp']}")
            
    except KeyboardInterrupt:
        logger.info("Program terminated by user")
        print("\nProgram terminated.")
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())