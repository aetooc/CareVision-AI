import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pipeline.logging_utils import log_action

# LLM
def _get_llm():

    api_key = os.getenv("MISTRAL_API_KEY", "")
    if not api_key:
        return None
    try:
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(
            model="mistral-small-latest",
            mistral_api_key=api_key,
            temperature=0.3,
            max_tokens=1024,
        )
    except ImportError:
        log_action("WARNING", "Run: pip install langchain-mistralai")
        return None
 


_SUMMARY_SYSTEM = """You are a clinical AI assistant helping to structure telemedicine consultations.
Given a transcript and visual observations, extract a concise clinical JSON.
Always respond with ONLY a valid JSON object - no markdown fences, no prose.
Schema:
{{
  "symptoms": ["list", "of", "reported", "symptoms"],
  "observations": "free-text visual/behavioural observations",
  "risk_flags": ["any", "flags", "or", "empty_list"],
  "summary": "2-3 sentence clinical summary",
  "recommended_followup": "suggested next steps"
}}"""
 
_SUMMARY_HUMAN = """Transcript:
{transcript}
 
Visual observations from frame analysis:
{vision_flags}
 
Posture assessment:
{posture}
 
Return the JSON object only."""
 
 
def generate_clinical_summary(transcript: str, vision_analysis: dict) -> dict:
    """LangChain LCEL chain: ChatPromptTemplate | Mistral | JsonOutputParser"""
    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SUMMARY_SYSTEM),
        ("human", _SUMMARY_HUMAN),
    ])
 
    if llm:
        chain = prompt | llm | JsonOutputParser()
        try:
            result = chain.invoke({
                "transcript": transcript,
                "vision_flags": vision_analysis.get("visual_flags", "None"),
                "posture": vision_analysis.get("posture_assessment", "Not assessed"),
            })
            log_action("AI_PROCESSING", "Mistral produced clinical JSON via LangChain LCEL.")
            return result
        except Exception as e:
            log_action("ERROR", f"LangChain chain failed: {e}", status="FAILED")
 
    log_action("AI_PROCESSING", "Demo mode - set MISTRAL_API_KEY for live inference.")
    return {
        "symptoms": ["self-reported concern noted in transcript"],
        "observations": vision_analysis.get("posture_assessment", "Not assessed"),
        "risk_flags": [],
        "summary": "Telemedicine consultation recorded. Transcript: " + transcript[:60] + "...",
        "recommended_followup": "Await clinician review.",
    }
 
 
class ClinicalChatAgent:
    _SYSTEM = (
        "You are a privacy-preserving clinical AI assistant for the CareVision platform.\n"
        "You have access to the patient's processed consultation data below.\n"
        "Answer questions concisely and factually. Never fabricate clinical information.\n\n"
        "Patient context:\n{context}"
    )
 
    def __init__(self, context: str):
        self.context = context
        self.history: list = []
        self.llm = _get_llm()
 
    def chat(self, user_message: str) -> str:
        if not self.llm:
            return (
                "Chat agent offline — MISTRAL_API_KEY not found.\n\n"
                "1. Sign up free at https://console.mistral.ai (no credit card)\n"
                "2. Create an API key\n"
                "3. Add to .streamlit/secrets.toml:\n"
                "   MISTRAL_API_KEY = \"your_key_here\""
            )
 

        msgs = [SystemMessage(content=self._SYSTEM.format(context=self.context))]
        for turn in self.history:
            msgs.append(HumanMessage(content=turn["human"]))
            msgs.append(AIMessage(content=turn["ai"]))
        msgs.append(HumanMessage(content=user_message))
 
        try:
            response = self.llm.invoke(msgs)
            reply = response.content
            self.history.append({"human": user_message, "ai": reply})
            log_action("CHAT_AGENT", f"Turn {len(self.history)} answered.")
            return reply
        except Exception as e:
            full = str(e)
            log_action("ERROR", f"Chat agent error: {full}", status="FAILED")
            if "429" in full or "rate" in full.lower():
                return "Rate limit — wait a moment and try again."
            if "401" in full or "403" in full or "key" in full.lower():
                return "Invalid API key — check MISTRAL_API_KEY in secrets.toml."
            return f"Agent error: {full}"
 