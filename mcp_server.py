from mcp.server import MCPServer
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from disputedesk import app_graph

mcp = MCPServer("DisputeDesk")


@mcp.tool()
def resolve_dispute(tenant_id: str, transaction_id: str, customer_message: str) -> dict:
    """Run DisputeDesk's multi-agent graph on a payment dispute: retrieves evidence,
    checks for fraud signals, drafts a response, and fact-checks it. Routine cases
    submit automatically; anomalous or ungrounded ones pause for human approval —
    call approve_dispute with the returned thread_id to complete those."""
    thread_id = f"dispute-{tenant_id}-{transaction_id}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    result = app_graph.invoke({
        "customer_message": customer_message,
        "tenant_id": tenant_id,
        "transaction_id": transaction_id,
        "evidence": None,
        "fraud_flag": None,
        "draft": None,
        "critic_verdict": None,
        "approved": None,
        "submitted": None,
    }, config=config)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return {
            "status": "awaiting_approval",
            "thread_id": thread_id,
            "draft_text": payload["draft_text"],
            "critic_notes": payload["critic_notes"],
        }

    return {
        "status": "submitted" if result["submitted"] else "rejected",
        "thread_id": thread_id,
        "draft_text": result["draft"].draft_text,
    }


@mcp.tool()
def approve_dispute(thread_id: str, approved: bool) -> dict:
    """Resume a dispute paused by resolve_dispute, with a human's approval decision."""
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    result = app_graph.invoke(Command(resume={"approved": approved}), config=config)
    return {
        "status": "submitted" if result["submitted"] else "rejected",
        "draft_text": result["draft"].draft_text,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")