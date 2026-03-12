from typing import TypedDict

from langchain_core.documents import Document
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from cadence_md.app.llm import LLMWrapper
from cadence_md.app.qdrant import QdrantManager
from cadence_md.app.reranker import RerankerWrapper


class RAGState(TypedDict):
    """State for LangGraph"""

    query: str
    retrieved_docs: list[Document]
    context: str
    answer: str


class RAGPipeline:
    """ """

    def __init__(
        self, llm: LLMWrapper, qdrant_manager: QdrantManager, reranker: RerankerWrapper
    ) -> None:
        """ """
        self.llm = llm
        self.reranker = reranker
        self.qdrant_manager = qdrant_manager
        self._graph = None

    def retrieve_node(self, state: RAGState) -> RAGState:
        """ """
        print(f"[RETRIEVE] Query: {state['query']}")

        docs = self.qdrant_manager.retrieve_documents(state["query"])

        print(f"Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs, 1):
            filename = doc.metadata.get("filename", "unknown")
            print(f"  {i}. {filename}")

        state["retrieved_docs"] = docs
        return state

    def reranker_node(self, state: RAGState) -> RAGState:
        """ """
        query = state["query"]
        docs = state["retrieved_docs"]

        print(f"[RERANK] {len(docs)} documents")

        docs = self.reranker.rerank(query, docs)
        state["retrieved_docs"] = [doc[0] for doc in docs]

        print(f"Reranked {len(docs)} documents")
        for i, doc in enumerate(docs, 1):
            filename = doc[0].metadata.get("filename", "unknown")
            print(f"  {i}. {filename}")

        return state

    def context_node(self, state: RAGState) -> RAGState:
        """ """
        print(f"\n[CONTEXT] Preparing context from {len(state['retrieved_docs'])} documents")

        # Combining documents into context
        context_parts = []
        for i, doc in enumerate(state["retrieved_docs"], 1):
            context_parts.append(f"[Document {i} - {doc.metadata['filename']}]\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)
        state["context"] = context

        print(f"Context prepared ({len(context)} chars)")

        return state

    def generate_node(self, state: RAGState) -> RAGState:
        """ """
        print("[GENERATE] Generating answer...")

        system_message = """Вы — отзывчивый медицинский ассистент.
        На основании предоставленных медицинских документов ответьте на вопрос пользователя точно и
        кратко.
        Если информация отсутствует в документах, четко укажите это.
        Приведите ссылки на документы."""

        prompt = f"""{system_message}
        Documents:
        {state["context"]}
        Question: {state["query"]}
        Answer:"""

        state["answer"] = self.llm.invoke(prompt)

        print(f"Generated {len(state['answer'].split())} words")

        return state

    def _build_graph(self):
        """ """
        print("\n[GRAPH] Building LangGraph...")

        workflow = StateGraph(RAGState)

        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("rerank", self.reranker_node)
        workflow.add_node("context", self.context_node)
        workflow.add_node("generate", self.generate_node)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "context")
        workflow.add_edge("context", "generate")
        workflow.add_edge("generate", END)

        graph = workflow.compile()

        print("✓ Graph built successfully")

        return graph

    def run(self, query: str):
        """ """
        if self._graph is None:
            self._graph = self._build_graph()
        graph = self._graph

        print(f"QUERY: {query}")

        graph = self._build_graph()

        initial_state = RAGState(query=query, retrieved_docs=[], context="", answer="")

        return graph.invoke(initial_state)
