from typing import TypedDict

from langchain_core.documents import Document
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from cadence_md.app.llm import LLMWrapper
from cadence_md.app.qdrant import QdrantManager
from cadence_md.app.reranker import RerankerWrapper


class RAGState(TypedDict):
    """State for LangGraph."""

    query: str
    retrieved_docs: list[Document]
    retrieved_scores: list[float]
    reranked_scores: list[float]
    context: str
    context_chars: int
    answer: str
    answer_word_count: int


class RAGPipeline:
    """RAG graph: retrieve → rerank → context → generate."""

    def __init__(
        self, llm: LLMWrapper, qdrant_manager: QdrantManager, reranker: RerankerWrapper
    ) -> None:
        self.llm = llm
        self.reranker = reranker
        self.qdrant_manager = qdrant_manager
        self._graph = None

    def retrieve_node(self, state: RAGState) -> RAGState:
        retrieved = self.qdrant_manager.retrieve(state["query"])
        docs = [doc for doc, _ in retrieved]
        scores = [score for _, score in retrieved]

        state["retrieved_docs"] = docs
        state["retrieved_scores"] = scores
        return state

    def reranker_node(self, state: RAGState) -> RAGState:
        query = state["query"]
        docs = state["retrieved_docs"]

        pairs = self.reranker.rerank(query, docs)
        state["retrieved_docs"] = [doc for doc, _ in pairs]
        state["reranked_scores"] = [float(s) if s is not None else 0.0 for _, s in pairs]

        return state

    def context_node(self, state: RAGState) -> RAGState:
        context_parts = []
        for i, doc in enumerate(state["retrieved_docs"], 1):
            filename = doc.metadata.get("filename", "unknown")
            context_parts.append(f"[Document {i} - {filename}]\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)
        state["context"] = context
        state["context_chars"] = len(context)

        return state

    def generate_node(self, state: RAGState) -> RAGState:
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
        state["answer_word_count"] = len(state["answer"].split())

        return state

    def _build_graph(self):
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

        return workflow.compile()

    def run(self, query: str) -> RAGState:
        if self._graph is None:
            self._graph = self._build_graph()

        initial_state: RAGState = {
            "query": query,
            "retrieved_docs": [],
            "retrieved_scores": [],
            "reranked_scores": [],
            "context": "",
            "context_chars": 0,
            "answer": "",
            "answer_word_count": 0,
        }

        return self._graph.invoke(initial_state)
