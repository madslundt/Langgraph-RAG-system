"""LangGraph setup and RAG chat functionality with configurable embeddings and ToolNode."""

import os
import sys
import time
from typing import  Any, TypedDict, Annotated
import operator
import chromadb
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_chroma import Chroma
from chromadb.config import Settings
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.align import Align
from rich.text import Text

from .config import SystemConfig, RAGDomainConfig
from .llm_factory import get_llm
from .embedding_factory import get_embeddings


console = Console()


class GraphState(TypedDict):
    """State of the conversation graph."""
    messages: Annotated[list[BaseMessage], operator.add]
    sources: list[str]


class RAGTool:
    """RAG tool for a specific domain."""

    def __init__(self, domain: RAGDomainConfig, collection):
        self.domain = domain
        self.collection = collection
        self.name = domain.name
        self.description = domain.description

    def search(self, query: str, n_results: int = 5) -> dict[str, Any]:
        """Search for relevant documents in this domain.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary with documents and sources
        """
        try:
            # Use LangChain Chroma's similarity search
            results = self.collection.similarity_search_with_score(
                query=query,
                k=n_results
            )

            return results

        except Exception as e:
            console.print(f"[red]Error searching {self.name}: {e}[/red]")
            return {'documents': [], 'sources': []}


class RAGChatSystem:
    """Main RAG chat system using LangGraph with ToolNode."""

    def __init__(self, config: SystemConfig):
        """Initialize the RAG chat system.

        Args:
            config: System configuration
        """
        self.config = config
        provider = os.environ.get('DEFAULT_LLM_PROVIDER')
        model = os.environ.get('DEFAULT_MODEL')
        self.llm = get_llm(provider, model)

        # Initialize embedding function (same as used in ingestion)
        self.embeddings = get_embeddings()

        self.chroma_client = chromadb.PersistentClient(
            path=config.chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize RAG tools
        self.rag_tools = {}
        self.tools = []

        for domain in config.rag_domains:
            try:
                # Use LangChain's Chroma wrapper with consistent embedding function
                collection = Chroma(
                    client=self.chroma_client,
                    collection_name=domain.name,
                    embedding_function=self.embeddings,
                    persist_directory=config.chroma_path
                )

                # Check if collection has any documents
                try:
                    count = collection._collection.count()
                    if count == 0:
                        console.print(f"[yellow]Warning: Collection '{domain.name}' is empty. Run populate first.[/yellow]")
                        continue
                except Exception:
                    console.print(f"[yellow]Warning: Could not check collection '{domain.name}' status.[/yellow]")
                    continue

                rag_tool = RAGTool(domain, collection)
                self.rag_tools[domain.name] = rag_tool

                # Create LangChain tool
                self._create_langchain_tool(rag_tool)

                console.print(f"[dim]✓ Loaded RAG tool: {domain.name} ({count} chunks)[/dim]")

            except Exception as e:
                console.print(f"[yellow]Warning: Could not load collection '{domain.name}': {e}[/yellow]")

        # Build graph
        self.graph = self._build_graph()

    def _create_langchain_tool(self, rag_tool: RAGTool) -> None:
        """Create a LangChain tool from a RAG tool."""

        @tool(rag_tool.domain.display_name, parse_docstring=False)
        def search_tool(query: str) -> Any:
            """Search tool"""
            results = rag_tool.search(query)

            return results

        search_tool.name = rag_tool.name
        search_tool.description = rag_tool.description

        self.tools.append(search_tool)

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph conversation graph with ToolNode."""

        # Define the graph
        workflow = StateGraph(GraphState)

        # Add nodes
        workflow.add_node("assistant", self._assistant_node)

        if self.tools:
            # Use ToolNode for automatic tool execution
            tool_node = ToolNode(self.tools)
            workflow.add_node("tools", tool_node)

        # Set entry point
        workflow.set_entry_point("assistant")

        # Add edges
        if self.tools:
            workflow.add_conditional_edges(
                "assistant",
                self._should_continue,
                {
                    "continue": "tools",
                    "end": END
                }
            )
            workflow.add_edge("tools", "assistant")
        else:
            workflow.add_edge("assistant", END)

        # Compile graph with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _assistant_node(self, state: GraphState) -> dict[str, Any]:
        """Assistant node that processes messages and decides on tool usage."""

        # Create system prompt
        system_prompt = """You are a helpful AI assistant with access to specialized knowledge bases.

When users ask questions, you can search through different documentation domains to find relevant information.
Use the available tools to search for information when needed.

Available tools:
""" + "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])

        system_prompt += """

Guidelines:
- Use tools when you need specific information from the documentation
- You can use multiple tools if the question spans different domains
- Always provide helpful responses even if no relevant information is found
"""

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}")
        ])

        # Create chain with tool binding
        chain = prompt | self.llm.bind_tools(self.tools)

        # Get response
        response = chain.invoke({"messages": state["messages"]})

        return {"messages": [response]}

    def _should_continue(self, state: GraphState) -> str:
        """Determine if we should continue to tools or end."""
        last_message = state["messages"][-1]

        # If there are tool calls, continue to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"
        else:
            return "end"

    def chat(self, message: str, session_id: str = "default") -> tuple[str, dict[str, list[str]]]:
        """Process a chat message and return response.

        Args:
            message: User message
            session_id: Session identifier for conversation memory

        Returns:
            Assistant response with sources
        """
        if not self.tools:
            return "No RAG tools available. Please run 'populate' command first to ingest documents."

        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=message)],
            "sources": []
        }

        # Run graph
        config = {"configurable": {"thread_id": session_id}}

        try:
            result = self.graph.invoke(initial_state, config)

            # Get final response
            final_message = result["messages"][-1]
            response_content = final_message.content

            # Extract sources from tool calls in the conversation
            sources = self._extract_sources_from_messages(result["messages"])

            return response_content, sources

        except Exception as e:
            console.print(f"[red]Error processing message: {e}[/red]")
            return f"Sorry, I encountered an error processing your request: {e}"

    def _extract_sources_from_messages(self, messages: list[BaseMessage]) -> dict[str, list[str]]:
        """Extract sources from tool calls in the message history."""
        sources: dict[str, list[tuple[Any, float]]] = {}

        for message in messages:
            # Check if this is a tool call message
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get('name', '')
                    query = tool_call.get('args', {}).get('query', '')
                    console.print(f"[dim]Calling tool '{tool_name}' with '{query}'[/dim]")
                    if tool_name in self.rag_tools:
                        # Get the query from tool call args
                        if query:
                            # Perform the search to get sources
                            rag_tool = self.rag_tools[tool_name]
                            search_results = rag_tool.search(query)
                            sources[tool_name] = search_results

        return sources

    def start_interactive_chat(self, session_id: str = "default") -> None:
        """Start an interactive chat session."""
        console.print("[bold blue]🤖 RAG Chat System[/bold blue]")
        console.print("Type your questions and I'll search through the available documentation.")
        console.print(f"[dim]Using LLM: {self.llm.__class__.__name__}[/dim]")
        console.print(f"[dim]Using embeddings: {self.embeddings.__class__.__name__}[/dim]")
        console.print("\nAvailable domains:")

        for domain in self.config.rag_domains:
            if domain.name in self.rag_tools:
                console.print(f"  • [green]{domain.display_name}[/green]: {domain.description}")
            else:
                console.print(f"  • [dim]{domain.display_name}[/dim]: [red]Not available (empty or error)[/red]")

        console.print("\nType 'quit' or 'exit' to end the conversation.\n")

        if not self.tools:
            console.print("[red]⚠️  No RAG tools available. Please run 'populate' command first.[/red]")
            return

        while True:
            try:
                # Get user input
                user_input = console.input("[bold cyan]You:[/bold cyan] ")

                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print("[bold blue]👋 Goodbye![/bold blue]")
                    break

                if not user_input.strip():
                    continue

                sys.stdout.write("\033[F")   # move cursor one line up
                sys.stdout.write("\033[K")   # clear the line
                sys.stdout.flush()

                # Display user message as a bubble (right-aligned)
                user_bubble = Panel(
                    Text(user_input, justify="right"),
                    title="You",
                    title_align="right",
                    border_style="cyan",
                    padding=(0, 2),
                )
                console.print(Align.right(user_bubble))

                # Get response
                console.print("\n[bold dim]Assistant thinking...[/bold dim]")
                start = time.time()
                response, sources = self.chat(user_input, session_id)
                end = time.time()
                console.print(f"[bold dim]Assistant thought for {round(end - start)} seconds[/bold dim]")

                # Display response with markdown formatting
                assistant_bubble = Panel(
                    Markdown(response),
                    title="Assistant",
                    title_align="left",
                    border_style="green",
                    width=80,
                    padding=(-1, 1),
                )
                console.print(assistant_bubble)
                if sources:
                    source_lines = []
                    for source in sources.values():
                        for s in source:
                            citation = s[0].metadata.get("source_citation", "")
                            if citation:
                                source_lines.append(f"       [dim]- {citation}[/dim]")  # indented + dimmed

                    if source_lines:
                        sources_text = "\n".join(source_lines)
                        # Draw a connector line down from the bubble
                        connector = "[dim]   │\n   ▼ Sources:[/dim]"
                        console.print(connector)
                        console.print(sources_text)
                        console.print()

                console.print()

                console.print()

            except KeyboardInterrupt:
                console.print("\n[bold blue]👋 Goodbye![/bold blue]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    def get_available_domains(self) -> list[str]:
        """Get list of available RAG domains."""
        return [domain.name for domain in self.config.rag_domains if domain.name in self.rag_tools]

    def get_system_info(self) -> dict[str, Any]:
        """Get system information for debugging."""
        return {
            'llm_provider': self.llm.__class__.__name__,
            'embedding_provider': self.embeddings.__class__.__name__,
            'available_tools': len(self.tools),
            'available_domains': self.get_available_domains(),
            'total_configured_domains': len(self.config.rag_domains)
        }
