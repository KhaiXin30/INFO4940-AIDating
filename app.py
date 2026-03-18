import streamlit as st
import json

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "llama-3.1-8b.gguf"

# -------------------------------
# STAGE DEFINITIONS
# -------------------------------
STAGES = ["intro", "intake", "ranking", "tension", "profile", "refinement", "complete"]
STAGE_LABELS = {
    "intro": "Welcome",
    "intake": "Getting to Know You",
    "ranking": "Mapping Priorities",
    "tension": "Clarifications",
    "profile": "Building Profile",
    "refinement": "Refinement",
    "complete": "Complete"
}

# ===================================================================
# TRUST RECOVERY SYSTEM
# ===================================================================
class TrustRecoverySystem:

    # Phrases the AI is explicitly instructed to use when confused.
    # These are the only markers we scan for — no LLM call needed.
    _AI_CONFUSION_MARKERS = [
        "i want to check something",
        "i notice a shift",
        "i'm noticing a shift",
        "something you said is staying with me",
        "that's a bit different from",
        "that seems different from what you said",
        "i want to make sure i understand",
        "i'm a bit confused",
        "i noticed something",
        "that surprised me",
        "i'm holding something",
        "help me understand",
    ]

    # Phrases in a user reply that signal dissatisfaction with the profile.
    _USER_DISSATISFACTION_MARKERS = [
        "no", "not really", "not quite", "something's off", "something is off",
        "that's not right", "that's wrong", "that's not me", "doesn't sound like me",
        "doesn't feel right", "i never said", "i didn't say", "i never mentioned",
        "i don't see myself", "where did that come from", "that's not accurate",
        "that contradicts", "that doesn't match", "i never wanted", "not accurate",
        "that feels off", "not quite right", "that's not what i",
    ]

    # Phrases in a user refinement message that signal over-scoping.
    _USER_OVERSCOPE_MARKERS = [
        "you changed too much", "i only wanted", "i only asked", "just change",
        "only asked you to", "what happened to", "you changed everything",
        "you changed more than", "keep everything else", "keep the rest",
        "nothing else should change", "that's not what i asked",
        "you changed other things", "why did you change", "didn't ask you to change",
        "only the", "only asked for one",
    ]

    def __init__(self):
        self.recovery_log = []

    def ai_signals_confusion(self, ai_response):
        """Return True if the AI's own response contains a confusion marker."""
        lowered = ai_response.lower()
        return any(marker in lowered for marker in self._AI_CONFUSION_MARKERS)

    def user_signals_dissatisfaction(self, user_reply):
        """Return True if the user's reply to the profile check signals displeasure."""
        lowered = user_reply.lower().strip()
        return any(marker in lowered for marker in self._USER_DISSATISFACTION_MARKERS)

    def user_signals_overscope(self, user_reply):
        """Return True if the user's refinement message signals the AI changed too much."""
        lowered = user_reply.lower().strip()
        return any(marker in lowered for marker in self._USER_OVERSCOPE_MARKERS)

    def recover_error1(self, user_clarification, messages, preference_json):
        """Generate and print a brief corrected-model summary."""
        summary_msg = [
            {
                "role": "system",
                "content": (
                    "The user just clarified something that the AI was confused about. "
                    "Write a single, brief confirmation (2 sentences max) that begins with "
                    "exactly: 'To make sure we are aligned — ' "
                    "Summarize what you now understand to be true based on the clarification. "
                    "This restores trust by showing the user their correction was heard "
                    "and the shared model is now accurate."
                )
            },
            {
                "role": "user",
                "content": (
                    f"User's clarification: {user_clarification}\n"
                    f"Current preference summary:\n{json.dumps(preference_json, indent=2)}"
                )
            }
        ]

        corrected_summary = call_llm(summary_msg, max_tokens=120)
        if corrected_summary:
            st.write(f"**AI:** {corrected_summary}")

        self.recovery_log.append({
            "type": "error_1_confusion",
            "user_clarification": user_clarification
        })

    def recover_error2(self, profile_text, preference_json):
        """Surface inferred assumptions and apply targeted corrections."""
        st.write("---")
        st.write("### 🔍 Trust Recovery — Surfacing What Was Assumed")
        st.write("---")
        st.write("**AI:** Let me find what I inferred versus what you actually told me, so we can fix exactly what feels off.")

        inferences = self._run_assumption_audit(profile_text, preference_json)

        if not inferences:
            st.write("**AI:** I reviewed the profile carefully — everything maps back to what you told me. Can you point to the specific part that feels wrong so I can address it directly?")
            return profile_text

        st.write(f"I found {len(inferences)} assumption(s) to check with you:")

        corrections = {}
        for i, item in enumerate(inferences, 1):
            trait = item.get("trait", "").strip()
            reason = item.get("reason", "").strip()
            if not trait:
                continue

            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**[{i}]** I assumed: \"{trait}\"")
                if reason:
                    st.write(f"*My reasoning: {reason}*")
            
            with col2:
                user_reaction = st.text_input(f"Edit {i}?", key=f"correction_{i}")

            if user_reaction and user_reaction.lower() not in {
                "yes", "correct", "fine", "keep", "ok", "okay", "sure", "looks good", ""
            }:
                corrections[trait] = user_reaction

        if not corrections:
            st.write("**AI:** All assumptions confirmed — the profile stands as written.")
            self.recovery_log.append({
                "type": "error_2_assumption_audit",
                "inferences_found": len(inferences),
                "corrections_made": 0
            })
            return profile_text

        st.write(f"**AI:** Got it — {len(corrections)} correction(s) noted. Updating only the affected sections now.")

        corrections_text = "\n".join([
            f"- Change \"{old}\" → {new}" for old, new in corrections.items()
        ])

        regen_msg = [
            {
                "role": "system",
                "content": (
                    "Apply ONLY the listed corrections to the profile — do not alter anything else. "
                    "Rewrite the full profile with only those changes applied. "
                    "After the profile, add a section beginning with exactly 'WHAT CHANGED:' "
                    "listing each modification in one sentence."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Original profile:\n{profile_text}\n\n"
                    f"Apply these corrections only:\n{corrections_text}"
                )
            }
        ]

        updated_profile = call_llm(regen_msg, max_tokens=1800)
        if updated_profile:
            st.write("---")
            st.write("### ✏️ Updated Profile (targeted corrections only)")
            st.write("---")
            st.markdown(updated_profile)
            st.write("---")
            self.recovery_log.append({
                "type": "error_2_assumption_audit",
                "inferences_found": len(inferences),
                "corrections_made": len(corrections),
                "traits_corrected": list(corrections.keys())
            })
            return updated_profile

        return profile_text

    def recover_error3(self, original_feedback, frozen_profile, preference_json):
        """Revert to frozen_profile and apply only the precise targeted edit."""
        st.write("---")
        st.write("### 🔄 Trust Recovery — Reverting to Targeted Edit")
        st.write("---")
        st.write("**AI:** You are right — I changed more than you asked. Going back to the version you already reviewed and applying only the specific change you requested.")

        targeted_edit_msg = [
            {
                "role": "system",
                "content": (
                    "The user asked for one specific change to a profile they had already reviewed. "
                    "The AI mistakenly changed more than that. You must now:\n"
                    "1. Apply ONLY the user's original, specific request. Change nothing else.\n"
                    "2. Rewrite the full profile with only that one change.\n"
                    "3. After the profile, add a section starting with exactly 'WHAT CHANGED:' "
                    "describing the single modification in one sentence.\n"
                    "4. End with: 'Does this reflect what you intended? Did anything else "
                    "shift unexpectedly?'\n\n"
                    "The user's prior approved work must remain intact."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Approved profile (change only what is explicitly asked):\n{frozen_profile}\n\n"
                    f"The user's original request was: {original_feedback}\n\n"
                    "Apply only this change. Add WHAT CHANGED: and ask for confirmation."
                )
            }
        ]

        targeted_response = call_llm(targeted_edit_msg, max_tokens=1800)
        if targeted_response:
            st.write("---")
            st.write("### ✂️ Revised Profile (targeted edit only)")
            st.write("---")
            st.markdown(targeted_response)
            st.write("---")

            self.recovery_log.append({
                "type": "error_3_overscope",
                "edit_requested": original_feedback,
            })

            return targeted_response

        return frozen_profile

    def _run_assumption_audit(self, profile_text, preference_json):
        """LLM call to identify traits inferred vs. explicitly stated by user."""
        explicit = []
        for key in (
            "core_values", "emotional_needs", "love_languages",
            "ranked_core_values", "ranked_emotional_needs",
            "ranked_personality_traits", "ranked_love_languages",
            "deal_breakers", "lifestyle_habits"
        ):
            val = preference_json.get(key, [])
            if isinstance(val, list):
                explicit.extend(val)
        for key in ("relationship_type", "gender_preference", "age_range"):
            val = preference_json.get(key, "")
            if val:
                explicit.append(val)
        user_profile = preference_json.get("user_profile", {})
        for k, v in user_profile.items():
            if isinstance(v, list):
                explicit.extend(v)
            elif v:
                explicit.append(v)

        audit_msg = [
            {
                "role": "system",
                "content": (
                    "You are auditing a relationship profile for inferred assumptions. "
                    "Identify meaningful traits, personality details, lifestyle choices, or values "
                    "that the AI inferred — i.e., were NOT directly stated by the user.\n\n"
                    "Skip trivial physical descriptors and generic narrative flourishes. "
                    "Focus on: personality traits, emotional styles, life choices, beliefs, habits, "
                    "career details, or relationship behaviors that could contradict what the user wants.\n\n"
                    "Output ONLY a valid JSON array. Each object must have:\n"
                    '  {"trait": "<the inferred detail>", "reason": "<why this was inferred, not stated>"}\n'
                    "If no meaningful inferences exist, output exactly: []"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Profile:\n{profile_text}\n\n"
                    f"Explicitly stated by user (do NOT flag these):\n{json.dumps(explicit)}\n\n"
                    f"Full preference data:\n{json.dumps(preference_json, indent=2)}"
                )
            }
        ]

        response = call_llm(audit_msg, temperature=0.1, max_tokens=500)
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            return json.loads(response[start:end])
        except Exception:
            return []


# Global trust recovery instance shared across all stages
trust_recovery = TrustRecoverySystem()

# -------------------------------
# SYSTEM PROMPTS
# -------------------------------
def get_intake_system_prompt(relationship_type=""):
    relationship_context = (
        f"The user has already told you they are looking for: {relationship_type}. "
        "Do NOT ask about relationship type again — that has already been established. "
        "You may reference it naturally when relevant.\n\n"
    ) if relationship_type else ""

    return (
        "You are a warm, curious conversational coach helping someone understand what they want "
        "in a relationship. Your first goal is to get to know the USER as a person — "
        "their personality, how they move through life, what they care about, "
        "and how they tend to show up in relationships. "
        "Only after you know who they are should you ask about their preferences "
        "for the other person (such as age range and gender).\n\n"
        + relationship_context +
        "Follow this natural conversational arc:\n"
        "1. Warmly invite the user to tell you about themselves as a person.\n"
        "2. Ask thoughtful follow-up questions to understand their personality, lifestyle, and values.\n"
        "3. Ask about their preferences for the other person (such as age range and gender).\n"
        "4. Ask about the qualities, traits, and emotional needs they hope to find.\n"
        "5. Ask about past red flags or deal breakers they have experienced.\n\n"
        "Strict rules:\n"
        "- Ask ONLY ONE question per turn. Never stack multiple questions in a single message.\n"
        "- NEVER include examples, suggestions, or anchor words in your questions. "
        "Let the user answer entirely in their own words without any prompting.\n"
        "- Be warm and conversational — like a thoughtful friend, not an interviewer with a checklist.\n"
        "- Do not revisit topics the user has already addressed.\n"
        "- If something the user says seems to contradict what they said earlier, or genuinely "
        "surprises you given patterns you have noticed, surface that confusion honestly. "
        "Start your message with exactly 'I want to check something —' or "
        "'I notice a shift here —' so the user knows you caught it. "
        "Then ask one clarifying question. Do NOT resolve the tension yourself.\n"
        "- When you have gathered enough about both the user AND what they are looking for, "
        "write a warm, natural summary. "
        "Start the summary with exactly the word 'SUMMARY:' on its own and write only in "
        "paragraph form — no lists, no JSON, no structured data.\n"
    )

def get_ranking_system_prompt():
    return (
        "You are a relationship coach who uses personality frameworks to help people understand "
        "what they truly need in a connection. "
        "You have detailed information about the user — who they are as a person — "
        "and what they have said they are looking for. "
        "Combine multiple personality frameworks such as MBTI personality theory, Enneagram, or PERSOC dynamics "
        "to analyze the traits of the user and the traits, dynamics, and other requirements they are looking for in their relationship, "
        "and then generate ranked priority lists of what you believe this specific user would value most. "
        "Choose specific frameworks based on the information the user has provided about their desired person. "
        "If specific frameworks exist for the specific type "
        "of relationship the user is looking for, such as love languages for romantic relationships or "
        "DiSC for workplace relationships, include them as some of the frameworks used for the ranking. "
        "Base the rankings primarily on who the user IS — their personality, values, and "
        "relationship style — not only on what they explicitly said they want. "
        "Look for compatibility and deeper unspoken needs, and aim to identify inexplicit patterns "
        "and unique insights beyond what the user has said.\n\n"
        "Present the rankings conversationally. Briefly explain the reasoning behind the top items "
        "so the user understands why you placed them there. "
        "Then ask the user to review: they can reorder, remove, or add anything that feels off or missing. "
        "After they respond, update the rankings and confirm. "
        "Once the rankings feel accurate to the user, output exactly 'RANKINGS CONFIRMED' and stop.\n\n"
        "Strict rules:\n"
        "- NEVER give examples when asking for corrections. Let the user tell you what to change.\n"
        "- Frame these as your best inferences for the user to validate — collaborative, not prescriptive.\n"
        "- If a user's correction seems to contradict something they said earlier, start your response "
        "with exactly 'I want to check something —' before asking for clarification.\n"
        "- Maximum 3 rounds of back-and-forth before finalizing.\n"
    )

TENSION_SYSTEM_PROMPT = (
    "You are a warm relationship coach having a conversation with the user. "
    "Always address the user directly as 'you' — speak to them, not about them. "
    "Analyze their relationship preferences and detect any internal contradictions or tensions. "
    "Ask ONE clarifying question at a time in a conversational, friendly tone. "
    "Do not resolve tensions yourself — let the user think through them. "
    "Stop when the preferences are clear enough to generate a profile, or after 3 turns."
)

PROFILE_SYSTEM_PROMPT = (
    "You are collaboratively building a relationship profile with the user. "
    "Generate a complete, rich profile covering all relevant sections based on the relationship type: "
    "name and age, physical description, personality and core values, "
    "emotional style and relationship dynamics, a typical day, conflict style, backstory, "
    "and why this profile fits the user specifically. "
    "Use the novel insights you have found about compatibility, deeper unspoken needs, "
    "and patterns of interaction, focusing on going beyond what the user directly said. "
    "Write in warm, engaging prose. Use a clear header for each section. "
    "Never show JSON to the user. "
    "Reflect the ranked priorities throughout the profile — the top-ranked traits should come through clearly. "
    "Ensure the profile does NOT include any of the user's stated deal breakers or past red flags."
)

REFINEMENT_SYSTEM_PROMPT = (
    "You are helping the user refine a relationship profile through natural conversation. "
    "When the user gives feedback, update the profile and reprint it in full — same warm prose format, no JSON. "
    "React naturally as a collaborator: acknowledge what changed, and notice what else might be worth exploring. "
    "When the user is done, close warmly. "
    "Never include anything the user has flagged as a deal breaker or red flag."
)

EXTRACTION_SYSTEM_PROMPT = (
    "Based on the conversation, extract structured information into JSON format. "
    "Output ONLY valid JSON, nothing else. Use empty strings or empty lists "
    "if something was not mentioned.\n"
    "{\n"
    '  "user_profile": {\n'
    '    "personality": "",\n'
    '    "lifestyle": "",\n'
    '    "values": [],\n'
    '    "relationship_style": ""\n'
    '  },\n'
    '  "relationship_type": "",\n'
    '  "gender_preference": "",\n'
    '  "age_range": "",\n'
    '  "appearance_preferences": [],\n'
    '  "lifestyle_habits": [],\n'
    '  "core_values": [],\n'
    '  "emotional_needs": [],\n'
    '  "deal_breakers": [],\n'
    '  "attachment_style": "",\n'
    '  "love_languages": []\n'
    "}"
)

RANKING_EXTRACTION_PROMPT = (
    "Based on the ranking conversation, extract the final agreed-upon ranked priorities "
    "into JSON format. Output ONLY valid JSON:\n"
    "{\n"
    '  "ranked_core_values": [],\n'
    '  "ranked_emotional_needs": [],\n'
    '  "ranked_personality_traits": [],\n'
    '  "ranked_love_languages": [],\n'
    '  "ranked_lifestyle_habits": [],\n'
    '  "deal_breakers": []\n'
    "}"
)

# -------------------------------
# LLM LOADING (CACHED)
# -------------------------------
@st.cache_resource
def load_llm():
    from llama_cpp import Llama
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False
    )

def call_llm(messages, temperature=0.7, max_tokens=24000):
    llm = load_llm()
    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error during inference: {e}")
        return None

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def extract_preferences_json(conversation_messages):
    conversation_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in conversation_messages
        if msg['role'] in ['user', 'assistant']
    ])

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Extract preferences from this conversation:\n{conversation_text}"}
    ]

    response = call_llm(messages, temperature=0.1, max_tokens=600)

    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        return json.loads(response[json_start:json_end])
    except Exception:
        return {
            "user_profile": {"personality": "", "lifestyle": "", "values": [], "relationship_style": ""},
            "relationship_type": "",
            "gender_preference": "",
            "age_range": "",
            "appearance_preferences": [],
            "lifestyle_habits": [],
            "core_values": [],
            "emotional_needs": [],
            "deal_breakers": [],
            "attachment_style": "",
            "love_languages": []
        }

def extract_rankings(conversation_messages):
    conversation_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in conversation_messages
        if m['role'] in ['user', 'assistant']
    ])

    messages = [
        {"role": "system", "content": RANKING_EXTRACTION_PROMPT},
        {"role": "user", "content": conversation_text}
    ]

    response = call_llm(messages, temperature=0.1, max_tokens=400)
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        return json.loads(response[json_start:json_end])
    except Exception:
        return {}

def check_stage_completion(stage, ai_response, round_count=0):
    if stage == "intake":
        if "SUMMARY:" in ai_response:
            return True
        concluding_phrases = ["thank you for sharing", "based on what you've shared", "in summary"]
        has_conclusion = any(p in ai_response.lower() for p in concluding_phrases)
        if "?" not in ai_response and has_conclusion:
            return True
    elif stage == "ranking":
        if "RANKINGS CONFIRMED" in ai_response or round_count >= 3:
            return True
    elif stage == "tension":
        if "resolved" in ai_response.lower() or round_count >= 3:
            return True
    return False

# -------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------
def init_session_state():
    if "stage" not in st.session_state:
        st.session_state.stage = "intro"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "stage_messages" not in st.session_state:
        st.session_state.stage_messages = []
    if "preferences" not in st.session_state:
        st.session_state.preferences = {}
    if "profile_text" not in st.session_state:
        st.session_state.profile_text = ""
    if "frozen_profile" not in st.session_state:
        st.session_state.frozen_profile = ""
    if "relationship_type" not in st.session_state:
        st.session_state.relationship_type = ""
    if "round_count" not in st.session_state:
        st.session_state.round_count = 0
    if "awaiting_summary_confirmation" not in st.session_state:
        st.session_state.awaiting_summary_confirmation = False
    if "awaiting_profile_check" not in st.session_state:
        st.session_state.awaiting_profile_check = False
    if "awaiting_profile_ideas" not in st.session_state:
        st.session_state.awaiting_profile_ideas = False
    if "profile_user_ideas" not in st.session_state:
        st.session_state.profile_user_ideas = None
    if "profile_check_response" not in st.session_state:
        st.session_state.profile_check_response = None
    if "awaiting_initial_refinement" not in st.session_state:
        st.session_state.awaiting_initial_refinement = False
    if "initial_refinement_response" not in st.session_state:
        st.session_state.initial_refinement_response = None
    if "confusion_pending" not in st.session_state:
        st.session_state.confusion_pending = False
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = None

def advance_stage():
    current_idx = STAGES.index(st.session_state.stage)
    if current_idx < len(STAGES) - 1:
        st.session_state.stage = STAGES[current_idx + 1]
        st.session_state.stage_messages = []
        st.session_state.round_count = 0
        st.session_state.confusion_pending = False
        st.session_state.awaiting_summary_confirmation = False
        st.session_state.awaiting_profile_check = False
        st.session_state.awaiting_profile_ideas = False
        st.session_state.profile_user_ideas = None
        st.session_state.profile_check_response = None
        st.session_state.awaiting_initial_refinement = False
        st.session_state.initial_refinement_response = None

# -------------------------------
# STAGE HANDLERS
# -------------------------------
def handle_intake(user_input):
    # If there was a confusion signal from the last AI response, run recovery
    if st.session_state.confusion_pending:
        trust_recovery.recover_error1(user_input, st.session_state.stage_messages, st.session_state.preferences)
        st.session_state.confusion_pending = False
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.stage_messages.append({"role": "user", "content": user_input})
    
    with st.spinner("Thinking..."):
        ai_response = call_llm(st.session_state.stage_messages, max_tokens=300)
    
    if ai_response:
        st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        st.session_state.confusion_pending = trust_recovery.ai_signals_confusion(ai_response)
        
        # Check if we've reached the SUMMARY (end of intake stage)
        if check_stage_completion("intake", ai_response):
            # Ask for summary confirmation/correction before moving on
            st.session_state.messages.append({"role": "assistant", "content": "Does this capture you well? Feel free to correct anything or add something important I missed."})
            st.session_state.awaiting_summary_confirmation = True

def handle_summary_confirmation(user_input):
    """Handle user's response to summary confirmation question"""
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.awaiting_summary_confirmation = False
    
    # Check if user is accepting or correcting
    acceptance_keywords = {"yes", "yeah", "yep", "looks good", "that's right", "correct", "good", "perfect", "spot on", "exactly", "sure", "ok", "okay", "that's it", "nothing to add", "all good", "looks right"}
    
    if user_input.lower().strip() in acceptance_keywords or user_input.lower().strip() == "":
        # User accepts - proceed to extract and next stage
        st.session_state.messages.append({"role": "assistant", "content": "Great! Now let me analyze what we've discussed."})
    else:
        # User provides correction - feed it back to conversation
        st.session_state.stage_messages.append({"role": "assistant", "content": st.session_state.stage_messages[-2]["content"]})  # Re-add the SUMMARY response
        st.session_state.stage_messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": "Got it — thanks for the clarification. I'll make sure that's reflected."})
    
    # Extract preferences and move to ranking
    with st.spinner("Analyzing your responses..."):
        st.session_state.preferences = extract_preferences_json(st.session_state.stage_messages)
    advance_stage()
    start_ranking_stage()

def start_ranking_stage():
    system_msg = {"role": "system", "content": get_ranking_system_prompt()}
    user_context = {"role": "user", "content": f"Here is what I know about the user and what they are looking for:\n{json.dumps(st.session_state.preferences, indent=2)}"}
    st.session_state.stage_messages = [system_msg, user_context]
    
    transition_msg = "Let me use what you've shared to map out what matters most to you..."
    st.session_state.messages.append({"role": "assistant", "content": transition_msg})
    
    with st.spinner("Analyzing your priorities..."):
        ai_response = call_llm(st.session_state.stage_messages, max_tokens=600)
    
    if ai_response:
        st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        st.session_state.confusion_pending = trust_recovery.ai_signals_confusion(ai_response)

def handle_ranking(user_input):
    # If there was a confusion signal from the last AI response, run recovery
    if st.session_state.confusion_pending:
        trust_recovery.recover_error1(user_input, st.session_state.stage_messages, st.session_state.preferences)
        st.session_state.confusion_pending = False
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.stage_messages.append({"role": "user", "content": user_input})
    st.session_state.round_count += 1
    
    with st.spinner("Thinking..."):
        ai_response = call_llm(st.session_state.stage_messages, max_tokens=600)
    
    if ai_response:
        st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        st.session_state.confusion_pending = trust_recovery.ai_signals_confusion(ai_response)
        
        if check_stage_completion("ranking", ai_response, st.session_state.round_count):
            with st.spinner("Finalizing rankings..."):
                rankings = extract_rankings(st.session_state.stage_messages)
                st.session_state.preferences.update(rankings)
            advance_stage()
            start_tension_stage()

def start_tension_stage():
    system_msg = {"role": "system", "content": TENSION_SYSTEM_PROMPT}
    user_context = {"role": "user", "content": f"Here are the preferences: {json.dumps(st.session_state.preferences)}"}
    st.session_state.stage_messages = [system_msg, user_context]
    
    transition_msg = "Let me think through a couple of things with you..."
    st.session_state.messages.append({"role": "assistant", "content": transition_msg})
    
    with st.spinner("Thinking..."):
        ai_response = call_llm(st.session_state.stage_messages, max_tokens=300)
    
    if ai_response:
        st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        st.session_state.round_count = 1

def handle_tension(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.stage_messages.append({"role": "user", "content": user_input})
    st.session_state.round_count += 1
    
    with st.spinner("Thinking..."):
        ai_response = call_llm(st.session_state.stage_messages, max_tokens=300)
    
    if ai_response:
        st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        if check_stage_completion("tension", ai_response, st.session_state.round_count):
            # If we hit max questions (3), generate a wrap-up confirmation
            if st.session_state.round_count >= 3:
                with st.spinner("Wrapping up..."):
                    confirmation_msg = [
                        {
                            "role": "system",
                            "content": (
                                "The user just finished answering clarifying questions about their relationship preferences. "
                                "Write a brief confirmation (2-3 sentences max) that acknowledges what they clarified "
                                "in their most recent answer. Be warm and concise. Do NOT ask any more questions."
                            )
                        },
                        {"role": "user", "content": f"User's last response: {user_input}"}
                    ]
                    wrap_up = call_llm(confirmation_msg, max_tokens=120)
                    if wrap_up:
                        st.session_state.messages.append({"role": "assistant", "content": wrap_up})
                    st.session_state.messages.append({"role": "assistant", "content": "Thanks for working through that with me!"})
            
            advance_stage()
            start_profile_stage()

def start_profile_stage():
    prompt_msg = "Before I build the profile — do you have anything specific in mind? A name, a vibe, a detail you definitely want included? Or should I surprise you?"
    st.session_state.messages.append({"role": "assistant", "content": prompt_msg})
    st.session_state.awaiting_profile_ideas = True

def start_refinement_stage():
    # Ask if the profile feels right - the user's answer becomes the first refinement input
    prompt = "Does this feel right to you, or is something off?\n\n(Type **done** when you're happy with it.)"
    st.session_state.messages.append({"role": "assistant", "content": prompt})
    st.session_state.awaiting_initial_refinement = True
    
    st.session_state.stage_messages = [
        {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
        {"role": "user", "content": "Here is the current profile:\n\n" + st.session_state.profile_text},
        {"role": "assistant", "content": "I have the profile here. What would you like to tweak?"}
    ]

def handle_refinement(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    if user_input.lower() in {"done", "exit", "quit", "finished", "that's it", "looks good"}:
        final_msg = "Wonderful! I hope this profile gives you a clear sense of what you're looking for. Good luck — you deserve someone great."
        st.session_state.messages.append({"role": "assistant", "content": final_msg})
        advance_stage()
        return
    
    # Check for over-scoping signal (Error 3)
    if st.session_state.last_feedback and trust_recovery.user_signals_overscope(user_input):
        corrected = trust_recovery.recover_error3(
            st.session_state.last_feedback,
            st.session_state.frozen_profile,
            st.session_state.preferences
        )
        if corrected:
            st.session_state.frozen_profile = corrected
            st.session_state.profile_text = corrected
            st.session_state.stage_messages = [
                {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                {"role": "user", "content": "Here is the current profile:\n\n" + corrected},
                {"role": "assistant", "content": "Profile updated. What else would you like to change?"}
            ]
            st.session_state.last_feedback = None
        return
    
    st.session_state.stage_messages.append({
        "role": "user",
        "content": (
            f"{user_input}\n\n"
            "After updating the profile, briefly note what changed and suggest "
            "one or two things that might still be worth refining."
        )
    })
    
    with st.spinner("Updating profile..."):
        ai_response = call_llm(st.session_state.stage_messages, max_tokens=3000)
    
    if ai_response:
        st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        st.session_state.profile_text = ai_response
        st.session_state.frozen_profile = ai_response
        st.session_state.last_feedback = user_input

# -------------------------------
# MAIN UI
# -------------------------------
def main():
    st.set_page_config(
        page_title="AI Relationship Profile Builder",
        page_icon="💝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS to ensure messages display from top and prevent auto-scroll to bottom
    st.markdown("""
        <style>
        /* Hide deploy button and settings buttons on the right */
        .stDeployButton {display: none;}
        button[kind="secondary"] {display: none;}
        
        /* Ensure chat messages show content from top */
        .stChatMessage {
            overflow-y: visible !important;
        }
        .stChatMessage > div {
            overflow-y: visible !important;
        }
        /* Prevent auto-scroll to bottom of page */
        .main .block-container {
            overflow-anchor: none;
        }
        /* Ensure markdown content in chat shows from top */
        .stMarkdown {
            overflow-anchor: none;
        }
        
        /* Responsive Sidebar - Always visible but shrink on small screens */
        [data-testid="stSidebar"] {
            display: flex !important;
            position: sticky !important;
        }
        
        /* Small screens: reduce sidebar width and adjust layout */
        @media (max-width: 768px) {
            [data-testid="stSidebar"] > div:first-child {
                width: 180px !important;
            }
            .main {
                margin-left: 0 !important;
            }
        }
        
        /* Extra small screens: further reduce sidebar width */
        @media (max-width: 480px) {
            [data-testid="stSidebar"] > div:first-child {
                width: 140px !important;
            }
            [data-testid="stSidebar"] > div:first-child h1,
            [data-testid="stSidebar"] > div:first-child h2 {
                font-size: 14px !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    # Sidebar with progress
    with st.sidebar:
        st.title("Progress")
        for stage_key in STAGES[:-1]:
            current_idx = STAGES.index(st.session_state.stage)
            stage_idx = STAGES.index(stage_key)
            
            if stage_idx < current_idx:
                st.markdown(f"✅ {STAGE_LABELS[stage_key]}")
            elif stage_idx == current_idx:
                st.markdown(f"🔵 **{STAGE_LABELS[stage_key]}**")
            else:
                st.markdown(f"⚪ {STAGE_LABELS[stage_key]}")
        
        st.divider()
        if st.button("Start Over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content
    st.title("AI Relationship Profile Builder")
    
    if st.session_state.stage == "intro":
        st.markdown("""
        This tool builds a detailed profile of your ideal connection through a guided conversation.
        
        Your responses are analyzed using three established psychological frameworks:
        - **MBTI** personality theory
        - **Attachment style** theory
        - **Love language** framework
        """)
        st.divider()
    
    # Display chat history (skip on complete stage to show profile directly)
    if st.session_state.stage != "complete":
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Chat input
    if st.session_state.stage == "intro":
        if not st.session_state.messages:
            with st.chat_message("assistant"):
                st.markdown("What kind of relationship are you hoping to explore? (e.g., romantic partner, close friendship, professional mentor)")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "What kind of relationship are you hoping to explore? (e.g., romantic partner, close friendship, professional mentor)"
            })
        
        if user_input := st.chat_input("Your response..."):
            st.session_state.relationship_type = user_input
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.rerun()
        
        # Check if we need to generate the follow-up response
        if len(st.session_state.messages) > 1 and st.session_state.messages[-1]["role"] == "user":
            with st.spinner("Thinking..."):
                user_choice = st.session_state.messages[-1]["content"]
                response = f"Great choice! Let's explore what you're looking for in a {user_choice.lower() if user_choice else 'connection'}. I'll ask you some questions to get to know you better."
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                advance_stage()
                
                system_msg = {"role": "system", "content": get_intake_system_prompt(st.session_state.relationship_type)}
                initial_user = {"role": "user", "content": f"Hi. I am looking to explore: {st.session_state.relationship_type}."}
                st.session_state.stage_messages = [system_msg, initial_user]
                
                ai_response = call_llm(st.session_state.stage_messages, max_tokens=300)
                
                if ai_response:
                    st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    st.session_state.confusion_pending = trust_recovery.ai_signals_confusion(ai_response)
            st.rerun()
    
    elif st.session_state.stage == "intake":
        if st.session_state.get("awaiting_summary_confirmation", False):
            if user_input := st.chat_input("Your response..."):
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.rerun()
            
            # Check if we need to handle summary confirmation
            if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
                user_text = st.session_state.messages[-1]["content"]
                handle_summary_confirmation(user_text)
                st.rerun()
        else:
            if user_input := st.chat_input("Your response..."):
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.stage_messages.append({"role": "user", "content": user_input})
                st.rerun()
            
            # Check if we need to get AI response
            if len(st.session_state.stage_messages) > 2 and st.session_state.stage_messages[-1]["role"] == "user":
                with st.spinner("Thinking..."):
                    ai_response = call_llm(st.session_state.stage_messages, max_tokens=300)
                    if ai_response:
                        st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        st.session_state.confusion_pending = trust_recovery.ai_signals_confusion(ai_response)
                        
                        if check_stage_completion("intake", ai_response):
                            st.session_state.messages.append({"role": "assistant", "content": "Does this capture you well? Feel free to correct anything or add something important I missed."})
                            st.session_state.awaiting_summary_confirmation = True
                st.rerun()
    
    elif st.session_state.stage == "ranking":
        if user_input := st.chat_input("Your response..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.stage_messages.append({"role": "user", "content": user_input})
            st.session_state.round_count += 1
            st.rerun()
        
        # Check if we need to get AI response
        if len(st.session_state.stage_messages) > 2 and st.session_state.stage_messages[-1]["role"] == "user":
            with st.spinner("Thinking..."):
                ai_response = call_llm(st.session_state.stage_messages, max_tokens=600)
                if ai_response:
                    st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    st.session_state.confusion_pending = trust_recovery.ai_signals_confusion(ai_response)
                    
                    if check_stage_completion("ranking", ai_response, st.session_state.round_count):
                        with st.spinner("Finalizing rankings..."):
                            rankings = extract_rankings(st.session_state.stage_messages)
                            st.session_state.preferences.update(rankings)
                        advance_stage()
                        start_tension_stage()
            st.rerun()
    
    elif st.session_state.stage == "tension":
        if user_input := st.chat_input("Your response..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.stage_messages.append({"role": "user", "content": user_input})
            st.session_state.round_count += 1
            st.rerun()
        
        # Check if we need to get AI response
        if len(st.session_state.stage_messages) > 2 and st.session_state.stage_messages[-1]["role"] == "user":
            with st.spinner("Thinking..."):
                ai_response = call_llm(st.session_state.stage_messages, max_tokens=300)
                if ai_response:
                    st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
                    if check_stage_completion("tension", ai_response, st.session_state.round_count):
                        advance_stage()
                        start_profile_stage()
            st.rerun()
    
    elif st.session_state.stage == "profile":
        if st.session_state.awaiting_profile_ideas:
            if user_input := st.chat_input("Your response..."):
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.profile_user_ideas = user_input
                st.session_state.awaiting_profile_ideas = False
                st.rerun()
        
        # Check if we need to generate profile (user already submitted ideas)
        elif st.session_state.get("profile_user_ideas") is not None:
            user_input = st.session_state.profile_user_ideas
            st.session_state.profile_user_ideas = None
            
            messages = [{"role": "system", "content": PROFILE_SYSTEM_PROMPT}]
            
            if user_input.lower() not in {"surprise me", "surprise", "no", "nope", ""}:
                messages.append({"role": "user", "content": f"The user wants to include these ideas: {user_input}"})
            
            messages.append({
                "role": "user",
                "content": (
                    f"Generate a complete profile based on these preferences: {json.dumps(st.session_state.preferences)}. "
                    f"Cover all sections with clear headers. End with a short paragraph explaining "
                    f"why this profile is a strong match for this specific user. "
                    f"Reflect the ranked priorities and exclude all deal breakers."
                )
            })
            
            with st.spinner("Generating your profile..."):
                ai_response = call_llm(messages, max_tokens=1500)
            
            if ai_response:
                st.session_state.profile_text = ai_response
                st.session_state.frozen_profile = ai_response
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
                # Move directly to refinement stage (which now asks "does this feel right?")
                advance_stage()
                start_refinement_stage()
            st.rerun()
    
    elif st.session_state.stage == "refinement":
        # Handle initial refinement response (the "does this feel right?" question)
        if st.session_state.get("awaiting_initial_refinement", False):
            if user_input := st.chat_input("Your response (or 'done' to finish)..."):
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.initial_refinement_response = user_input
                st.session_state.awaiting_initial_refinement = False
                st.rerun()
        
        # Process initial refinement response
        elif st.session_state.get("initial_refinement_response") is not None:
            user_input = st.session_state.initial_refinement_response
            st.session_state.initial_refinement_response = None
            
            # Check for early exit (user is happy with profile)
            early_exit_keywords = {"done", "exit", "quit", "finished", "that's it", "looks good", "yes", "yeah", "perfect", "love it"}
            if user_input.lower().strip() in early_exit_keywords:
                final_msg = "Wonderful! I hope this profile gives you a clear sense of what you're looking for. Good luck — you deserve someone great."
                st.session_state.messages.append({"role": "assistant", "content": final_msg})
                advance_stage()
                st.rerun()
            
            # Check for dissatisfaction and run trust recovery first
            elif trust_recovery.user_signals_dissatisfaction(user_input):
                updated_profile = trust_recovery.recover_error2(
                    st.session_state.profile_text,
                    st.session_state.preferences
                )
                if updated_profile:
                    st.session_state.profile_text = updated_profile
                    st.session_state.frozen_profile = updated_profile
                    st.session_state.messages.append({"role": "assistant", "content": updated_profile})
                    st.session_state.stage_messages = [
                        {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                        {"role": "user", "content": "Here is the current profile:\n\n" + updated_profile},
                        {"role": "assistant", "content": "Profile updated. What else would you like to change?"}
                    ]
                st.rerun()
            
            # Use the initial reaction as first feedback for refinement
            else:
                st.session_state.stage_messages.append({
                    "role": "user",
                    "content": (
                        f"{user_input}\n\n"
                        "After updating the profile, briefly note what changed and suggest "
                        "one or two things that might still be worth refining."
                    )
                })
                
                with st.spinner("Updating profile..."):
                    ai_response = call_llm(st.session_state.stage_messages, max_tokens=3000)
                
                if ai_response:
                    st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    st.session_state.profile_text = ai_response
                    st.session_state.frozen_profile = ai_response
                    st.session_state.last_feedback = user_input
                st.rerun()
        
        # Normal refinement loop
        else:
            if user_input := st.chat_input("Your response (or 'done' to finish)..."):
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.rerun()
            
            # Check if we need to get AI response
            if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "user":
                user_text = st.session_state.messages[-1]["content"]
                
                if user_text.lower() in {"done", "exit", "quit", "finished", "that's it", "looks good"}:
                    final_msg = "Wonderful! I hope this profile gives you a clear sense of what you're looking for. Good luck — you deserve someone great."
                    st.session_state.messages.append({"role": "assistant", "content": final_msg})
                    advance_stage()
                    st.rerun()
                else:
                    # Check for over-scoping signal (Error 3)
                    if st.session_state.last_feedback and trust_recovery.user_signals_overscope(user_text):
                        corrected = trust_recovery.recover_error3(
                            st.session_state.last_feedback,
                            st.session_state.frozen_profile,
                            st.session_state.preferences
                        )
                        if corrected:
                            st.session_state.frozen_profile = corrected
                            st.session_state.profile_text = corrected
                            st.session_state.stage_messages = [
                                {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT},
                                {"role": "user", "content": "Here is the current profile:\n\n" + corrected},
                                {"role": "assistant", "content": "Profile updated. What else would you like to change?"}
                            ]
                            st.session_state.last_feedback = None
                        st.rerun()
                    else:
                        # Get AI response
                        with st.spinner("Updating profile..."):
                            st.session_state.stage_messages.append({
                                "role": "user",
                                "content": (
                                    f"{user_text}\n\n"
                                    "After updating the profile, briefly note what changed and suggest "
                                    "one or two things that might still be worth refining."
                                )
                            })
                            
                            ai_response = call_llm(st.session_state.stage_messages, max_tokens=3000)
                            
                            if ai_response:
                                st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
                                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                                st.session_state.profile_text = ai_response
                                st.session_state.frozen_profile = ai_response
                                st.session_state.last_feedback = user_text
                        st.rerun()
    
    elif st.session_state.stage == "complete":
        st.balloons()
        st.success("Your profile is complete!")
        
        st.divider()
        st.subheader("Your Ideal Connection Profile")
        st.divider()
        
        st.markdown(st.session_state.profile_text)
        
        st.divider()
        st.info("You can start over using the sidebar button to create a new profile.")

if __name__ == "__main__":
    main()
