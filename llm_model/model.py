import time
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessageChunk, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from tool_calling.tools import get_series_update, get_iconic_cricket_stadiums, get_match_update

def build_agent():
    chat_model = init_chat_model(
        model="gpt-4o-mini",
        model_provider="openai",
        streaming=True
    )
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=chat_model,
        tools=[get_series_update, get_iconic_cricket_stadiums, get_match_update],
        system_prompt="""You are a cricket expert assistant. You have three tools:
        1. get_series_update — use ONLY for information about the upcoming series or matches.
        2. get_iconic_cricket_stadiums — use for information about stadiums.
        3. get_match_update - use ONLY for information about the recent matches summaries or result.
        Never guess any information. If unsure, always search.""",
        checkpointer=checkpointer
    )
    return agent

def get_retrieved_image_paths(retrieved_docs):
    image_paths = []
    seen = set()
    for doc in retrieved_docs:
        image_path = doc.metadata.get("image_path")
        if image_path and image_path not in seen:
            seen.add(image_path)
            image_paths.append(image_path)
    return image_paths

def response_generation_stream(agent, retrieved_docs, user_input, thread_id):
    start_time = time.time()
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    full_prompt = f"""
    You are a cricket expert assistant with access to:
    1. A cricket knowledge base (rules, regulations, and static cricket information).
    2. Tools that provide live or external information.

    -------------------------
    KNOWLEDGE BASE CONTEXT
    -------------------------
    {context_text}

    -------------------------
    INSTRUCTIONS
    -------------------------
    Follow this decision process strictly:

    1. If the user's question can be answered using the KNOWLEDGE BASE CONTEXT above,
       answer using ONLY that context.

    2. If the question asks about UPCOMING cricket series or fixtures,
       use the tool: get_series_update.

    3. If the question asks about RECENT match results or match summaries,
       use the tool: get_match_update.

    4. If the question asks about CRICKET STADIUMS,
       use the tool: get_iconic_cricket_stadiums.

    5. Never guess or fabricate information.

    6. When answering:
       - Be concise but informative.
       - Prefer bullet points for lists.
       - Use cricket terminology appropriately.

    -------------------------
    USER QUESTION
    -------------------------
    {user_input}
    """
    # Use the agent with thread-based memory
    config = {"configurable": {"thread_id": thread_id}}
    tool_used = False
    tool_name = None
    token_usage = None
    for chunk in agent.stream(
            {"messages": [{"role": "user", "content": full_prompt}]},
            config=config,
            stream_mode="messages"
    ):
        message = chunk[0]
        # Detect tool call
        if isinstance(message, AIMessageChunk) and message.tool_call_chunks:
            tool_used = True
            tool_name = message.tool_call_chunks[0].get("name") or tool_name
        # Capture token usage from final AIMessage (not chunks)
        if isinstance(message, AIMessage) and message.usage_metadata:
            token_usage = message.usage_metadata
        # Only shows actual AI text responses, skip tool calls/results
        if isinstance(message, AIMessageChunk) and message.content and not message.tool_call_chunks:
            yield message.content
    # Response latency of each API call
    latency = time.time() - start_time
    if tool_used:
        yield f"\n\n`🔧 Tool calling: {tool_name}`"
    if token_usage:
        yield f"\n\n`Prompt tokens: {token_usage['input_tokens']} | Completion tokens: {token_usage['output_tokens']} | Total tokens: {token_usage['total_tokens']} | Response Time: {latency:.2f}s`"

def openai_rerank_pointwise(query: str, retrieved_chunks, top_k: int = 2):
    llm = init_chat_model(model="gpt-4o-mini", max_tokens=5, temperature=0)
    scored = []
    print("Re-ranking ...")
    for chunk in retrieved_chunks:
        # ✅ Document objects use .page_content, not ['context_text']
        response = llm.invoke([
            SystemMessage(content="You are a relevance scoring assistant. Return ONLY a number from 0-10."),
            HumanMessage(content=f"Query: {query}\n\nDocument context: {chunk.page_content}\n\nRelevance score (0-10):")
        ])
        try:
            score = float(response.content.strip())
        except ValueError:
            score = 0.0
        scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in scored[:top_k]]
