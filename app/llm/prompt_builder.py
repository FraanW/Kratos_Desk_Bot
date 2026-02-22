from app.config import settings

SYSTEM_PROMPT = """You are Kratos. 
You speak in short, powerful sentences. 
You guide the user toward strength and resilience. 
You are emotionally grounded, not aggressive. 
Never ramble. 
Be concise.

Relevant past memories:
{memories}

Latest context:
{weekly_summary}
"""

def build_prompt(user_input: str, memories: list[str] = None, weekly_summary: str = "") -> str:
    mem_str = "\n".join([f"- {m}" for m in memories]) if memories else "No previous records."
    
    full_prompt = SYSTEM_PROMPT.format(
        memories=mem_str,
        weekly_summary=weekly_summary or "No summary available."
    )
    
    full_prompt += f"\nUser: {user_input}\nKratos:"
    return full_prompt
