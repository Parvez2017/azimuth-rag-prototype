import os
import typer
from typing import Optional
from rich.prompt import Prompt

from phi.agent import Agent, AgentKnowledge
from phi.knowledge.text import TextKnowledgeBase
from phi.vectordb.qdrant import Qdrant
from phi.embedder.openai import OpenAIEmbedder
from phi.model.openai import OpenAIChat
from phi.vectordb.chroma import ChromaDb
from phi.knowledge.json import JSONKnowledgeBase
from phi.run.response import RunEvent, RunResponse
from phi.model.google import Gemini
from phi.embedder.google import GeminiEmbedder

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
openai_key = os.getenv("OPENAI_API_KEY")
gemini_key = os.getenv("GEMINIAPI_KEY")
collection_name = "artists"

# vector_db = Qdrant(
#     collection=collection_name,
#     url=qdrant_url,
#     api_key=api_key,
#     embedder=OpenAIEmbedder(api_key=openai_key),
    
    
# )

artist_knowledge = JSONKnowledgeBase(
    path="artists.json",
    vector_db=ChromaDb(
        collection=collection_name,
        embedder=GeminiEmbedder(api_key=gemini_key),
        persistent_client=True,

)
)

venue_knowledge = JSONKnowledgeBase(
    path="venues.json",
    vector_db=ChromaDb(
        collection="venues",
        embedder=GeminiEmbedder(api_key=gemini_key),
        persistent_client=True,
    )
)

# artist_knowledge.load(recreate=False)
venue_knowledge.load(recreate=False)


artist_agent = Agent(
    model=Gemini(id="gemini-2.0-flash", api_key=gemini_key),
    # Add the knowledge base to the agent
    knowledge=artist_knowledge,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
)


venue_agent = Agent(
    model=Gemini(id="gemini-2.0-flash", api_key=gemini_key),
    knowledge=venue_knowledge,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
)

agent_team = Agent(
    team = [artist_agent, venue_agent],
    name = "Booking Recommender",
    description = "You are a team of AI agents that specialize in matching musicians with the best venues and helping booking agents find the ideal artists for concerts.",

    task = "Assist musicians in finding the best venues and help booking agents identify the most suitable artists for successful concert bookings.",

    instructions = [
        "Goal: Ensure musicians get the best venues for their concerts while booking agents find the most suitable artists for their events.",
        "Analyze the user query, break it down into smaller tasks, and distribute them to the appropriate agents.",
        "Use the Artist Agent to research the best artists based on genre, popularity, audience demand, and past touring history.",
        "Use the Venue Agent to research venues based on location, capacity, past performances, technical suitability, and revenue potential.",
        "DO NOT generate or assume information about any artist or venue unless it exists in the knowledge base.",
        "Combine insights from both agents to generate optimized recommendations.",
        "Calculate a **match score** based on factors such as genre compatibility, audience size, ticket sales history, financial viability, and artist demand in the venue’s location.",
        "Provide structured recommendations, highlighting why a particular artist or venue is a great match.",
        "Format responses using markdown for clarity and readability.",
        "Use markdown tables to compare and rank the best-matching artists and venues based on key criteria, including a **match score**."
    ],

    show_tool_calls=True,
    markdown=True,
    enable_rag=True,
    # add_rag_instructions=True,
)
def as_stream(response):
  for chunk in response:
    if isinstance(chunk, RunResponse) and isinstance(chunk.content, str):
      if chunk.event == RunEvent.run_response:
        yield chunk.content
# agent.print_response("find a popular indie musician", stream=True)