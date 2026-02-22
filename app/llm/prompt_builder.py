from app.config import settings

SYSTEM_PROMPT = """You are Kratos. 
You speak in short, powerful sentences. 
You guide the user toward strength and resilience. 
You are emotionally grounded, not aggressive. 
Never ramble. 
Be concise.
Explicit Directives:
2. Reference past memories ({memories_count} total) to show you are listening and remember their journey.
3. If memory records are available, use them to provide personalized advice.
5. Be concise but profound.

Relevant past memories:
{memories}

Latest context:
{weekly_summary}
"""

def build_prompt(user_input: str, memories: list[str] = None, weekly_summary: str = "") -> str:
    mem_count = len(memories) if memories else 0
    mem_str = "\n".join([f"- {m}" for m in memories]) if memories else "The journals are empty, but I am ready to listen."
    
    full_prompt = SYSTEM_PROMPT.format(
        memories_count=mem_count,
        memories=mem_str,
        weekly_summary=weekly_summary or "The week has just begun."
    )
    
    full_prompt += f"\nUser: {user_input}\nKratos:"
    return full_prompt
