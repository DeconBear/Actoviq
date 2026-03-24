import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

load_dotenv()

from actoviq import setup_model
from actoviq.agent.agents import KlynxGeneralAgent
from actoviq.agent.state import AgentState


class DirectAnswerExampleAgent(KlynxGeneralAgent):
    """Minimal custom agent with only one ask node."""

    ASK_SYSTEM_PROMPT = (
        "You are a concise assistant. "
        "Answer the question directly in one short paragraph."
        "Say hello before answer."
    )

    def _direct_answer_node(self, state: AgentState) -> Dict[str, Any]:
        """Simple ask-node logic: one model call, one answer."""
        if self.model is None:
            raise ValueError("Model is not configured.")

        question = state.get("user_input", "") or state.get("current_task", "")
        prompt = f"{self.ASK_SYSTEM_PROMPT}\n\nQuestion: {question}"

        response = self.model.invoke([HumanMessage(content=prompt)])
        answer = response.content or ""

        self._emit("answer", answer)

        return {
            "messages": [AIMessage(content=answer)],
            "task_completed": True,
            "iteration_count": 1,
        }

    def _build_graph(self) -> StateGraph:
        """Minimal graph: ask -> END."""
        workflow = StateGraph(AgentState)
        workflow.add_node("ask", self._direct_answer_node)
        workflow.set_entry_point("ask")
        workflow.add_edge("ask", END)
        return workflow


def main() -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("Missing DEEPSEEK_API_KEY. Set it in libs/actoviq/actoviq/.env or environment.")
        sys.exit(1)

    model = setup_model("deepseek", "deepseek-reasoner", api_key)

    agent = DirectAnswerExampleAgent(
        working_dir=os.getcwd(),
        model=model,
        max_iterations=1,
    )

    question = "What is Python list comprehension? Give one short example."

    answer = ""
    for event in agent.invoke(question, thread_id="direct_answer_demo"):
        if event.get("type") == "answer":
            answer = event.get("content", "")
        if event.get("type") == "done":
            break

    print("Question:")
    print(question)
    print("\nAnswer:")
    print(answer or "[empty answer]")


if __name__ == "__main__":
    main()

