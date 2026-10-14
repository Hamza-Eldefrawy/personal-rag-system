import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.prompts import PromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory

from config import (
    OPENROUTER_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
)

load_dotenv()

DEFAULT_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions about a person.
If you don't know the answer, say you don't know.
Keep your answer within three sentences.Keep the answer as concise as possible.
Use the following pieces of context to answer the question at the end. 
Always say "Any further questions?" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""


def get_llm(
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    api_key: str = None,
    base_url: str = OPENROUTER_BASE_URL,
) -> ChatOpenAI:
    """Initialize ChatOpenAI pointing to OpenRouter."""
    resolved_api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not resolved_api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    return ChatOpenAI(
        api_key=resolved_api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )


def get_prompt_template(template_str: str = None) -> PromptTemplate:
    """Create PromptTemplate for question answering."""
    return PromptTemplate.from_template(template_str or DEFAULT_PROMPT_TEMPLATE)


def get_conversation_memory(
    memory_key: str = "chat_history",
    output_key: str = "answer",
    return_messages: bool = True,
) -> ConversationBufferMemory:
    """Initialize conversation buffer memory with message list format."""
    return ConversationBufferMemory(
        memory_key=memory_key,
        return_messages=return_messages,
        output_key=output_key,
    )


def build_conversational_chain(
    llm: ChatOpenAI,
    retriever,
    prompt: PromptTemplate = None,
    memory: ConversationBufferMemory = None,
    return_source_documents: bool = True,
) -> ConversationalRetrievalChain:
    """Assemble the ConversationalRetrievalChain."""
    if prompt is None:
        prompt = get_prompt_template()

    if memory is None:
        memory = get_conversation_memory()

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=return_source_documents,
        combine_docs_chain_kwargs={"prompt": prompt},
        memory=memory,
    )
