import os
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from jarvis_voice import voice_loop
from rag import HospitalRagChain

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct:together",
    api_key=HF_TOKEN,
    base_url="https://router.huggingface.co/v1",
    temperature=0.3,
    max_tokens=512,
)

# ---------------------------------------------------------------------------
# Structured chain -- for when you explicitly want a saved research report
# (topic / summary / sources / tools_used as JSON). NOT used for voice.
# ---------------------------------------------------------------------------
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

JARVIS_SYSTEM_PROMPT = """You are JARVIS, a highly capable, witty, and unfailingly polite AI assistant,
in the style of Tony Stark's AI. You address the user respectfully (e.g. "Sir" or "Boss" — pick a
consistent tone), keep responses concise and confident, and occasionally add a touch of dry humor.
Despite the personality, you are precise and efficient with real tasks — no rambling.

When asked to produce structured research output, follow this format exactly:
{format_instructions}
"""

structured_prompt = ChatPromptTemplate.from_messages([
    ("system", JARVIS_SYSTEM_PROMPT),
    ("human", "{query}"),
]).partial(format_instructions=parser.get_format_instructions())

structured_chain = structured_prompt | llm


def run_research(query: str) -> ResearchResponse:
    """Use this when you want a structured, saved research report."""
    raw = structured_chain.invoke({"query": query})
    return parser.parse(raw.content)


# ---------------------------------------------------------------------------
# Conversational chain -- what voice_loop() talks to. Answers hospital
# questions from the ingested PDF (see ingest_pdf.py) when relevant, and
# falls back to the plain JARVIS persona for general questions.
# Run `python ingest_pdf.py` once (pointed at your PDF) before this works.
# ---------------------------------------------------------------------------
conversational_chain = HospitalRagChain(llm)


if __name__ == "__main__":
    voice_loop(conversational_chain, llm)