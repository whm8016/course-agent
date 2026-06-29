"""Built-in capabilities for the 课程 Agent."""
from core.capabilities.chat import ChatCapability
from core.capabilities.deep_solve import DeepSolveCapability
from core.capabilities.deep_research import DeepResearchCapability
from core.capabilities.quiz import QuizCapability

__all__ = [
    "ChatCapability",
    "DeepSolveCapability",
    "DeepResearchCapability",
    "QuizCapability",
]
