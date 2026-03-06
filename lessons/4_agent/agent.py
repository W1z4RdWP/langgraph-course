from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from src.settings import settings


# Арифметические инструменты
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

     Args:
         a: first int
         b: second int
     """
    return a * b


def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b




tools = [add, multiply, divide]

# Инициализация модели
llm = ChatOpenAI(
    model="gpt-5.2",
    api_key=settings.OPENAI_API_KEY,
    temperature=0.1,
    max_retries=2,
    base_url="https://api.proxyapi.ru/openai/v1"
)

llm_with_tools = llm.bind_tools(tools)


def assistant(state: MessagesState) -> MessagesState:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


tools_node = ToolNode(tools=tools)

builder = StateGraph(MessagesState)

# Добавление узлов
builder.add_node("assistant", assistant)
builder.add_node("tools", tools_node)

# Определение рёбер
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # Если последнее сообщение - вызов инструмента → переход к tools
    # Если не вызов инструмента → переход к END
    tools_condition
)
builder.add_edge("tools", "assistant")  # Ключевое ребро цикла

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}


react_graph = builder.compile(checkpointer=checkpointer)

def call_msg(msg: str, config: dict = config) -> None:
    message = [HumanMessage(content=msg)]
    result = react_graph.invoke({"messages": message}, config=config)
    for m in result['messages']:
        m.pretty_print()

if __name__ == '__main__':
    call_msg("Сложи 5 и 5")
    call_msg("Умножь результат на 3")
    call_msg("Скажи какое число получилось после выполнения первой операции?")
    call_msg("Раздели его на 2")