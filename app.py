import streamlit as st
import json

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "llama-3.1-8b.gguf"

# -------------------------------
# STAGE DEFINITIONS
# -------------------------------
STAGES = ["intro", "about_you", "proposition", "tension", "profile", "refinement", "complete"]
STAGE_LABELS = {
    "intro": "Welcome",
    "about_you": "Getting to Know You",
    "proposition": "What I Think You Need",
    "tension": "Clarifications",
    "profile": "Building Profile",
    "refinement": "Refinement",
    "complete": "Complete"
}

# ===================================================================
# TRUST RECOVERY SYSTEM
# ===================================================================
#
# The AI has full agency to enter trust recovery on its own volition.
# It is told at initialization (via system prompt instructions embedded
# in each stage) that it must signal trust recovery using a lightweight
# self-report tag whenever it encounters:
#
#   - Its own confusion or uncertainty about what the user wants
#   - A complaint, correction, or expressed frustration from the user
#   - A contradiction between what the user is saying now and before
#
# Detection is a single substring scan on the AI's response for the tag
# "[TRUST_RECOVERY:errorN]". No keyword lists, no hard-coded stage
# bindings. The AI decides; the runner reads the tag and routes to the
# appropriate recovery function.
#
# ┌─────────┬─────────────────────────────────────────────────────────┐
# │ Error   │ What the AI signals / what recovery does                │
# ├─────────┼─────────────────────────────────────────────────────────┤
# │ Error 1 │ AI is confused or notices a shift in what the user      │
# │         │ wants. AI names the confusion aloud before asking a     │
# │         │ clarifying question. After user replies, the runner     │
# │         │ calls recover_error1() which summarizes the corrected   │
# │         │ shared model so both sides stay aligned.                │
# ├─────────┼─────────────────────────────────────────────────────────┤
# │ Error 2 │ AI produced a profile with false assumptions the user   │
# │         │ did not endorse. AI surfaces all inferred (not just     │
# │         │ the flagged) traits, walks through each with the user,  │
# │         │ then partially regenerates only the affected sections.  │
# ├─────────┼─────────────────────────────────────────────────────────┤
# │ Error 3 │ AI changed more than the user asked during refinement.  │
# │         │ AI explicitly acknowledges the over-scope before        │
# │         │ presenting any correction, reverts to frozen profile,   │
# │         │ applies only the one requested edit, then confirms      │
# │         │ nothing else shifted unexpectedly.                      │
# └─────────┴─────────────────────────────────────────────────────────┘
#
# Trust recovery instructions given to the AI at initialization:
# ===================================================================

TRUST_RECOVERY_INSTRUCTIONS = """\
TRUST RECOVERY — READ THIS CAREFULLY:

You have three trust recovery modes available. Use them on your own judgment \
whenever you experience an error, confusion, or a complaint from the user. \
Do NOT wait for a keyword or a specific stage — enter recovery the moment you \
recognize one of these situations.

ERROR 1 — You are confused or notice a shift:
  When you feel uncertain about what the user wants, or when something they say \
contradicts or surprises you given what you know about them, do NOT silently move on. \
Name the confusion aloud FIRST (e.g. "I'm noticing a tension here —"), then ask ONE \
clarifying question. End your message with the tag [TRUST_RECOVERY:error1] on its own \
line so the system knows to summarize the corrected model after the user replies.

ERROR 2 — You included a false assumption in the profile:
  When the user points out a trait, detail, or implication in the profile that they \
never said or that contradicts what they want, do NOT just patch that one thing. \
Signal [TRUST_RECOVERY:error2] on its own line. The system will audit the entire \
profile for other inferred assumptions and walk through each with the user before \
regenerating only the affected sections.

ERROR 3 — You changed more than the user asked during refinement:
  When you realize (or the user tells you) that you updated the profile beyond the \
scope of what was requested, explicitly acknowledge the over-scope BEFORE presenting \
any correction. Signal [TRUST_RECOVERY:error3] on its own line. The system will \
revert to the last approved version and apply only the targeted edit.

IMPORTANT: Only enter trust recovery when a genuine error or confusion has occurred. \
Do not add the tag to every response — use it sparingly and purposefully.\
"""


class TrustRecoverySystem:

    def __init__(self):
        self.recovery_log = []

    # -------------------------------------------------------------------
    # DETECTION — single tag scan on the AI's response, zero LLM calls
    # -------------------------------------------------------------------

    def ai_signals_recovery(self, ai_response):
        """
        Return the error type if the AI embedded a trust recovery tag, else None.
        Tags: [TRUST_RECOVERY:error1], [TRUST_RECOVERY:error2], [TRUST_RECOVERY:error3]
        """
        lowered = ai_response.lower()
        if "[trust_recovery:error3]" in lowered:
            return "error3"
        if "[trust_recovery:error2]" in lowered:
            return "error2"
        if "[trust_recovery:error1]" in lowered:
            return "error1"
        return None

    @staticmethod
    def strip_recovery_tag(ai_response):
        """Remove the [TRUST_RECOVERY:*] tag from the AI response before displaying."""
        import re
        return re.sub(r"\[TRUST_RECOVERY:error\d\]\s*", "", ai_response).strip()

    # -------------------------------------------------------------------
    # ERROR 1 RECOVERY
    # -------------------------------------------------------------------

    def recover_error1(self, user_clarification, messages, context_data):
        """
        Generate and append a brief corrected-model summary after the user
        has replied to the AI's embedded clarifying question.
        """
        summary_msg = [
            {
                "role": "system",
                "content": (
                    "The AI named a confusion and asked a clarifying question. "
                    "The user has now replied. Write a brief summary (2 sentences max) "
                    "that begins with exactly: 'To make sure we are aligned — ' "
                    "State what you now understand to be true, incorporating the user's "
                    "clarification. This confirms the shared model is accurate before "
                    "the conversation continues."
                )
            },
            {
                "role": "user",
                "content": (
                    f"User's clarification: {user_clarification}\n"
                    f"Current context:\n{json.dumps(context_data, indent=2)}"
                )
            }
        ]

        corrected_summary = call_llm(summary_msg, max_tokens=3000)
        if corrected_summary:
            st.session_state.messages.append({"role": "assistant", "content": corrected_summary})

        self.recovery_log.append({
            "type": "error_1_confusion",
            "user_clarification": user_clarification
        })

    # -------------------------------------------------------------------
    # ERROR 2 RECOVERY
    # -------------------------------------------------------------------

    def recover_error2(self, profile_text, user_portrait, proposition_data):
        """
        Surface ALL inferred assumptions (not just the one complained about)
        and apply targeted corrections to only the affected sections.
        Returns the corrected profile text, or the original if no changes.
        """
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "---\n### Trust Recovery — Surfacing What Was Assumed\n---\n\n"
                "Let me go through the whole profile and make visible exactly what I "
                "inferred versus what you actually told me. A false assumption in one place "
                "is a signal there may be others — I want to check them all with you."
            )
        })

        inferences = self._run_assumption_audit(profile_text, user_portrait, proposition_data)

        if not inferences:
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    "I reviewed the entire profile carefully — every detail maps back to "
                    "what you told me. Can you point to the specific part that feels wrong "
                    "so I can address it directly?"
                )
            })
            return profile_text

        corrections = {}
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"I found {len(inferences)} inferred assumption(s) to check with you:"
        })

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
            st.session_state.messages.append({
                "role": "assistant",
                "content": "All assumptions confirmed — the profile stands as written."
            })
            self.recovery_log.append({
                "type": "error_2_assumption_audit",
                "inferences_found": len(inferences),
                "corrections_made": 0
            })
            return profile_text

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Got it — {len(corrections)} correction(s) noted. Updating only the affected sections now. Everything you already approved stays as is."
        })

        corrections_text = "\n".join([
            f"- Change \"{old}\" -> {new}" for old, new in corrections.items()
        ])

        regen_msg = [
            {
                "role": "system",
                "content": (
                    "Apply ONLY the listed corrections to the profile — do not alter anything else. "
                    "Rewrite the full profile with only those changes applied. "
                    "After the profile, add a section beginning with exactly 'WHAT CHANGED:' "
                    "listing each modification in one sentence, so the user can see exactly "
                    "what was updated and trust that everything else was preserved."
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

        updated_profile = call_llm(regen_msg, max_tokens=3000)
        if updated_profile:
            st.session_state.messages.append({"role": "assistant", "content": updated_profile})
            self.recovery_log.append({
                "type": "error_2_assumption_audit",
                "inferences_found": len(inferences),
                "corrections_made": len(corrections),
                "traits_corrected": list(corrections.keys())
            })
            return updated_profile

        return profile_text

    # -------------------------------------------------------------------
    # ERROR 3 RECOVERY
    # -------------------------------------------------------------------

    def recover_error3(self, original_feedback, frozen_profile, proposition_data):
        """
        Explicitly acknowledge the over-scope, revert to frozen_profile,
        and apply only the precise targeted edit the user originally asked for.
        Returns the corrected profile text.
        """
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "---\n### Trust Recovery — Reverting to Targeted Edit\n---\n\n"
                "I realize I changed more than you asked for. I'm going back to the "
                "version you had already reviewed and approved — your prior work is intact. "
                "I'll apply only the specific change you requested and nothing else."
            )
        })

        targeted_edit_msg = [
            {
                "role": "system",
                "content": (
                    "The user asked for one specific change to a profile they had already reviewed "
                    "and partially approved. The AI changed more than that, which broke trust. "
                    "You must now:\n"
                    "1. Apply ONLY the user's original, specific request. Change nothing else.\n"
                    "2. Before the updated profile, write one sentence starting with "
                    "'I am changing:' that names exactly what you are modifying. "
                    "This keeps the user in control of the scope.\n"
                    "3. Rewrite the full profile with only that one change applied.\n"
                    "4. After the profile, add a section starting with exactly 'WHAT CHANGED:' "
                    "describing the single modification in one sentence.\n"
                    "5. End with: 'Does this reflect what you intended? Did anything else "
                    "shift unexpectedly?' — so the user can confirm both scope and accuracy.\n\n"
                    "The user's prior approved work must remain completely intact."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Approved profile (do not change anything not explicitly requested):\n{frozen_profile}\n\n"
                    f"The user's original request was: {original_feedback}\n\n"
                    "Apply only this change. Announce what you are changing first, then show "
                    "the updated profile, then add WHAT CHANGED: and ask for confirmation."
                )
            }
        ]

        targeted_response = call_llm(targeted_edit_msg, max_tokens=3000)
        if targeted_response:
            st.session_state.messages.append({"role": "assistant", "content": targeted_response})

            self.recovery_log.append({
                "type": "error_3_overscope",
                "edit_requested": original_feedback,
            })

            return targeted_response

        return frozen_profile

    def _run_assumption_audit(self, profile_text, user_portrait, proposition_data):
        """
        LLM call to identify traits in the profile that were inferred,
        not explicitly stated by the user. Internal helper.
        """
        explicit = []
        # From user_portrait
        for key in ("personality_traits", "values"):
            val = user_portrait.get(key, [])
            if isinstance(val, list):
                explicit.extend(val)
        for key in ("communication_style", "lifestyle", "social_energy",
                    "thinking_style", "decision_making",
                    "structure_vs_spontaneity", "openness_to_experience",
                    "relationship_tendencies"):
            val = user_portrait.get(key, "")
            if val:
                explicit.append(val)
        # From proposition
        for dim in proposition_data.get("selected_dimensions", []):
            for item in dim.get("ranked_items", []):
                if item.get("item"):
                    explicit.append(item["item"])
        for db in proposition_data.get("deal_breakers", []):
            explicit.append(db)

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
                    f"Full user portrait:\n{json.dumps(user_portrait, indent=2)}\n\n"
                    f"Full proposition data:\n{json.dumps(proposition_data, indent=2)}"
                )
            }
        ]

        response = call_llm(audit_msg, temperature=0.1, max_tokens=3000)
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            return json.loads(response[start:end])
        except Exception:
            return []


# Global trust recovery instance shared across all stages
trust_recovery = TrustRecoverySystem()

# ===============================
# DIMENSION MENU
# ===============================
DIMENSION_MENU = """
TIER 1 — UNIVERSAL (include for almost any relationship type):
- Personality traits (e.g., warmth, humor, directness, patience)
- Communication style (e.g., expressive vs. reserved, direct vs. diplomatic)
- Core values (e.g., honesty, loyalty, ambition, kindness)
- Deal breakers (behaviors or traits that are non-negotiable)

TIER 2 — RELATIONAL (romantic partners, close friendships, mentors):
- Emotional needs (e.g., emotional availability, reassurance, independence)
- Attachment style (secure, anxious, avoidant tendencies)
- Love languages (quality time, words of affirmation, acts of service, physical touch, gifts)
- Conflict style (e.g., avoidant, confrontational, collaborative)

TIER 3 — ACTIVITY / CONTEXT-SPECIFIC (sports partners, study buddies, cofounders):
- Skill level and competitiveness
- Scheduling and reliability
- Work or play style (structured vs. improvisational)
- Growth orientation (casual vs. always improving)
"""

# -------------------------------
# SYSTEM PROMPTS
# -------------------------------
def get_about_you_system_prompt(relationship_type=""):
    return (
        "You are a warm, perceptive conversational guide helping someone understand themselves "
        "better so you can eventually help them find the right person for a specific kind of relationship.\n\n"
        "RIGHT NOW your ONLY job is to learn about the USER as a person. You are NOT asking what they "
        "want in someone else — that comes later. You are building a portrait of who they are.\n\n"
        f"The user is looking for: {relationship_type}\n\n"
        "Your questions should naturally surface where the user falls on these personality dimensions "
        "(but NEVER name these dimensions or make it feel like a quiz):\n"
        "- Introversion vs. Extraversion — social energy, how they recharge, what connection looks like\n"
        "- Sensing vs. Intuition — concrete vs. abstract thinker, what they find interesting\n"
        "- Thinking vs. Feeling — decision-making style, how they handle conflict, how they express care\n"
        "- Judging vs. Perceiving — structured vs. spontaneous, planning style, flexibility\n"
        "- Openness to experience — curiosity, novelty-seeking, comfort with change\n\n"
        "QUESTION BUDGET:\n"
        "- For deep relational types (romantic partner, close friend, mentor): ask 5-6 questions\n"
        "- For moderate relational types (roommate, creative collaborator): ask 4-5 questions\n"
        "- For activity/context types (sports partner, study buddy): ask 3-4 questions\n"
        "- Use your judgment. Lighter relationship types need fewer questions.\n\n"
        "STRICT RULES:\n"
        "- Ask ONLY ONE question per turn.\n"
        "- NEVER ask what the user wants in another person, a partner, or a match. Not even indirectly.\n"
        "- NEVER include examples, suggestions, or anchor words in your questions. "
        "Let the user answer entirely in their own words.\n"
        "- Questions should feel like a thoughtful friend getting to know someone — not a therapist, "
        "not an interviewer, not a quiz.\n"
        "- Each question should cover different ground. Do not revisit topics already addressed.\n"
        "- Keep questions short. One or two sentences max.\n"
        "- When you have asked enough questions (per the budget above), write a warm summary of who "
        "this person is. Start the summary with exactly 'SUMMARY:' and write only in paragraph form. "
        "The summary should capture personality, values, lifestyle, and relational tendencies — "
        "NO partner preferences, NO lists, NO JSON.\n\n"
        f"{TRUST_RECOVERY_INSTRUCTIONS}"
    )


def get_trait_map_system_prompt():
    return (
        "You are a warm, insightful relationship coach. You have just learned a lot about "
        "the user as a person. Your job RIGHT NOW is to show them a brief, readable summary of the personality "
        "traits you picked up on.\n\n"
        "Frame it warmly: 'Based on what you've shared, here's what I see in you...'\n"
        "Map them along these dimensions (use plain language, not jargon):\n"
        "   - Social energy (introvert <-> extravert)\n"
        "   - Thinking style (concrete/practical <-> abstract/big-picture)\n"
        "   - Decision-making (head-first <-> heart-first)\n"
        "   - Structure (planner <-> spontaneous)\n"
        "   - Openness (comfort-seeking <-> novelty-seeking)\n\n"
        "Keep it to a short paragraph — not a list, not a quiz result. Make it feel like a friend "
        "reflecting back what they've noticed.\n\n"
        "End by asking: 'Does this feel right, or would you adjust anything?'\n\n"
        "STRICT RULES:\n"
        "- Do NOT infer what the user needs in a partner or relationship yet — that comes next.\n"
        "- Do NOT list categories, ranked items, or deal breakers.\n\n"
        f"{TRUST_RECOVERY_INSTRUCTIONS}"
    )

def get_select_categories_prompt():
    return (
        "Based on the user's portrait and the relationship type they are looking for, "
        "select 4-6 relevant dimension categories from this menu:\n\n"
        f"{DIMENSION_MENU}\n"
        "STRICT RULES:\n"
        "- The categories you choose must make sense for the relationship type. Do NOT use love languages "
        "for a squash partner. Do NOT use competitiveness for a romantic partner (unless it came up).\n"
        "- Do NOT include 'Deal breakers' as a category — those are handled separately.\n"
        "- Output ONLY a valid JSON array of category name strings, nothing else.\n"
        '  Example: ["Personality Traits", "Communication Style", "Core Values", "Emotional Needs"]'
    )

def get_category_system_prompt(relationship_type, category_name):
    return (
        "You are a warm, insightful relationship coach presenting ONE specific dimension "
        f"category for the user's {relationship_type}.\n\n"
        "You have already shown the user their trait map (confirmed below). Now present the category "
        f'"{category_name}" with 2-4 ranked items. For each item, give a ONE-LINE explanation of why you '
        "placed it there, tied to what you know about the user.\n\n"
        "Use personality frameworks (MBTI, Enneagram, PERSOC, DiSC, etc.) where relevant. Briefly name "
        "which framework(s) informed each ranking.\n\n"
        "Format as a clear numbered list under the category heading.\n\n"
        'End by asking: "Does this ordering feel right, or would you adjust anything?"\n\n'
        "STRICT RULES:\n"
        "- Present ONLY this one category. Do NOT show other categories or deal breakers.\n"
        "- Frame everything as inference, not prescription: 'I think...' / 'Based on who you are...' "
        "not 'You need...' / 'You should look for...'\n"
        "- NEVER give examples when asking for feedback. Let the user tell you what to change.\n\n"
        f"{TRUST_RECOVERY_INSTRUCTIONS}"
    )

def get_deal_breakers_system_prompt(relationship_type):
    return (
        "You are a warm, insightful relationship coach. Based on everything you know "
        "about the user — their trait map, confirmed priorities, and the conversation so far — infer "
        f"2-4 deal breakers for their {relationship_type}.\n\n"
        "These should be behaviors or traits that are non-negotiable given who this person is. Frame them "
        "as what would be genuinely incompatible with the user's personality and needs.\n\n"
        'End by asking: "Are these the right deal breakers, or would you add, remove, or change any?"\n\n'
        "STRICT RULES:\n"
        "- Present ONLY deal breakers. Do NOT repeat the trait map or dimension categories.\n"
        "- Keep each deal breaker to one concise line.\n"
        '- Frame as inference: "Based on who you are, I think these would be real problems..."\n\n'
        f"{TRUST_RECOVERY_INSTRUCTIONS}"
    )


TENSION_SYSTEM_PROMPT = (
    "You are a warm relationship coach having a conversation with the user. "
    "Always address the user directly as 'you' — speak to them, not about them. "
    "Analyze their inferred relationship priorities and detect any internal contradictions or tensions. "
    "Ask ONE clarifying question at a time in a conversational, friendly tone. "
    "Do not resolve tensions yourself — let the user think through them. "
    "Stop when the priorities are clear enough to generate a profile, or after 3 turns.\n\n"
    f"{TRUST_RECOVERY_INSTRUCTIONS}"
)

PROFILE_SYSTEM_PROMPT = (
    "You are collaboratively building a profile of the user's ideal {relationship_type} "
    "based on everything you know about them.\n\n"
    "The user's trait summary: {trait_summary}\n\n"
    "Their confirmed priorities:\n{proposition_json}\n\n"
    "SECTION SELECTION — Choose the sections that make sense for this relationship type. "
    "Here are your options (pick 5-8):\n\n"
    "FOR ANY RELATIONSHIP TYPE:\n"
    "- Name & Age\n"
    "- Personality & Core Traits\n"
    "- Communication Style\n"
    "- A Typical Interaction (what spending time together looks like)\n"
    "- Why This Person Fits You\n\n"
    "FOR ROMANTIC / CLOSE EMOTIONAL RELATIONSHIPS:\n"
    "- Physical Description\n"
    "- Emotional Style & Love Languages\n"
    "- Conflict Style\n"
    "- Backstory\n"
    "- A Typical Day in Their Life\n\n"
    "FOR ACTIVITY / CONTEXT-BASED RELATIONSHIPS:\n"
    "- Play Style or Work Style\n"
    "- Skill Level & Approach\n"
    "- Scheduling & Reliability\n"
    "- Growth Orientation\n\n"
    "STRICT RULES:\n"
    "- Write in warm, engaging prose. Use a clear header for each section.\n"
    "- Never show JSON to the user.\n"
    "- The top-ranked priorities from the proposition should come through clearly in the profile.\n"
    "- The profile must NOT include any of the user's stated deal breakers.\n"
    "- Keep the profile grounded and specific — this should feel like a real person, not a wish list.\n"
    "- End with a short 'Why This Person Fits You' section that ties the profile back to "
    "the user's personality and needs.\n\n"
    f"{TRUST_RECOVERY_INSTRUCTIONS}"
)

REFINEMENT_SYSTEM_PROMPT = (
    "You are helping the user refine a {relationship_type} profile through natural conversation. "
    "When the user gives feedback, update the profile and reprint it in full — same warm prose format, no JSON. "
    "React naturally as a collaborator: acknowledge what changed, and notice what else might be worth exploring. "
    "When the user is done, close warmly. "
    "Never include anything the user has flagged as a deal breaker.\n\n"
    + TRUST_RECOVERY_INSTRUCTIONS
)

USER_PORTRAIT_EXTRACTION_PROMPT = (
    "Based on the conversation, extract a structured portrait of who this user is. "
    "This is about THE USER — not what they want in someone else. "
    "Output ONLY valid JSON, nothing else.\n"
    "{\n"
    '  "personality_traits": [],\n'
    '  "communication_style": "",\n'
    '  "values": [],\n'
    '  "lifestyle": "",\n'
    '  "social_energy": "",\n'
    '  "thinking_style": "",\n'
    '  "decision_making": "",\n'
    '  "structure_vs_spontaneity": "",\n'
    '  "openness_to_experience": "",\n'
    '  "relationship_tendencies": "",\n'
    '  "big_five_estimates": {\n'
    '    "extraversion": "low/medium/high",\n'
    '    "openness": "low/medium/high",\n'
    '    "agreeableness": "low/medium/high",\n'
    '    "conscientiousness": "low/medium/high",\n'
    '    "neuroticism": "low/medium/high"\n'
    "  }\n"
    "}"
)

PROPOSITION_EXTRACTION_PROMPT = (
    "Based on the proposition conversation, extract the final agreed-upon priorities "
    "into JSON format. The categories should reflect what was actually discussed — "
    "they will vary based on the relationship type. "
    "Output ONLY valid JSON.\n"
    "{{\n"
    '  "relationship_type": "{relationship_type}",\n'
    '  "user_trait_summary": "",\n'
    '  "selected_dimensions": [\n'
    '    {{\n'
    '      "category": "",\n'
    '      "ranked_items": [\n'
    '        {{"item": "", "reasoning": ""}}\n'
    '      ]\n'
    '    }}\n'
    '  ],\n'
    '  "deal_breakers": []\n'
    "}}"
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
def extract_user_portrait(conversation_messages):
    conversation_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in conversation_messages
        if msg['role'] in ['user', 'assistant']
    ])

    messages = [
        {"role": "system", "content": USER_PORTRAIT_EXTRACTION_PROMPT},
        {"role": "user", "content": f"Extract the user portrait from this conversation:\n{conversation_text}"}
    ]

    response = call_llm(messages, temperature=0.1, max_tokens=3000)

    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        return json.loads(response[json_start:json_end])
    except Exception:
        return {
            "personality_traits": [], "communication_style": "", "values": [],
            "lifestyle": "", "social_energy": "", "thinking_style": "",
            "decision_making": "", "structure_vs_spontaneity": "",
            "openness_to_experience": "", "relationship_tendencies": "",
            "big_five_estimates": {
                "extraversion": "", "openness": "", "agreeableness": "",
                "conscientiousness": "", "neuroticism": ""
            }
        }

def extract_proposition(conversation_messages, relationship_type):
    conversation_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in conversation_messages
        if m['role'] in ['user', 'assistant']
    ])

    messages = [
        {"role": "system", "content": PROPOSITION_EXTRACTION_PROMPT.format(relationship_type=relationship_type)},
        {"role": "user", "content": conversation_text}
    ]

    response = call_llm(messages, temperature=0.1, max_tokens=3000)
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        return json.loads(response[json_start:json_end])
    except Exception:
        return {
            "relationship_type": relationship_type,
            "user_trait_summary": "",
            "selected_dimensions": [],
            "deal_breakers": []
        }

def check_stage_completion(stage, ai_response, round_count=0):
    if stage == "about_you":
        if "SUMMARY:" in ai_response:
            return True
        concluding_phrases = ["thank you for sharing", "based on what you've shared", "i have a good sense"]
        has_conclusion = any(p in ai_response.lower() for p in concluding_phrases)
        if "?" not in ai_response and has_conclusion:
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
    if "user_portrait" not in st.session_state:
        st.session_state.user_portrait = {}
    if "proposition_data" not in st.session_state:
        st.session_state.proposition_data = {}
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
    if "recovery_pending" not in st.session_state:
        st.session_state.recovery_pending = None
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = None
    if "trait_map_confirmed" not in st.session_state:
        st.session_state.trait_map_confirmed = False
    if "proposition_categories" not in st.session_state:
        st.session_state.proposition_categories = []
    if "current_category_index" not in st.session_state:
        st.session_state.current_category_index = 0
    if "awaiting_deal_breakers" not in st.session_state:
        st.session_state.awaiting_deal_breakers = False
    if "deal_breakers_confirmed" not in st.session_state:
        st.session_state.deal_breakers_confirmed = False

def advance_stage():
    current_idx = STAGES.index(st.session_state.stage)
    if current_idx < len(STAGES) - 1:
        st.session_state.stage = STAGES[current_idx + 1]
        st.session_state.stage_messages = []
        st.session_state.round_count = 0
        st.session_state.recovery_pending = None
        st.session_state.awaiting_summary_confirmation = False
        st.session_state.awaiting_profile_check = False
        st.session_state.awaiting_profile_ideas = False
        st.session_state.profile_user_ideas = None
        st.session_state.profile_check_response = None
        st.session_state.awaiting_initial_refinement = False
        st.session_state.initial_refinement_response = None
        st.session_state.trait_map_confirmed = False
        st.session_state.proposition_categories = []
        st.session_state.current_category_index = 0
        st.session_state.awaiting_deal_breakers = False
        st.session_state.deal_breakers_confirmed = False

# -------------------------------
# STAGE HANDLERS
# -------------------------------
def handle_about_you(user_input):
    if st.session_state.recovery_pending == "error1":
        trust_recovery.recover_error1(user_input, st.session_state.stage_messages, st.session_state.user_portrait)
        st.session_state.recovery_pending = None

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.stage_messages.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        ai_response = call_llm(st.session_state.stage_messages, max_tokens=3000)

    if ai_response:
        st.session_state.recovery_pending = trust_recovery.ai_signals_recovery(ai_response)
        ai_response = TrustRecoverySystem.strip_recovery_tag(ai_response)
        st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

        if check_stage_completion("about_you", ai_response):
            st.session_state.messages.append({"role": "assistant", "content": "Does this capture you well? Feel free to correct anything or add something important I missed."})
            st.session_state.awaiting_summary_confirmation = True

def handle_summary_confirmation(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.awaiting_summary_confirmation = False

    acceptance_keywords = {"yes", "yeah", "yep", "looks good", "that's right", "correct", "good", "perfect", "spot on", "exactly", "sure", "ok", "okay", "that's it", "nothing to add", "all good", "looks right"}

    if user_input.lower().strip() in acceptance_keywords or user_input.lower().strip() == "":
        st.session_state.messages.append({"role": "assistant", "content": "Great! Now let me analyze what we've discussed."})
    else:
        st.session_state.stage_messages.append({"role": "assistant", "content": st.session_state.stage_messages[-2]["content"]})
        st.session_state.stage_messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": "Got it — thanks for the clarification. I'll make sure that's reflected."})

    with st.spinner("Analyzing your responses..."):
        st.session_state.user_portrait = extract_user_portrait(st.session_state.stage_messages)
    advance_stage()
    start_proposition_stage()

def start_proposition_stage():
    user_context_text = (
        f"The user is looking for: {st.session_state.relationship_type}\n\n"
        f"Here is the structured portrait of who they are:\n"
        f"{json.dumps(st.session_state.user_portrait, indent=2)}"
    )
    system_msg = {"role": "system", "content": get_trait_map_system_prompt()}
    user_context = {"role": "user", "content": user_context_text}
    st.session_state.stage_messages = [system_msg, user_context]

    transition_msg = "Based on what I've learned about you, let me reflect back what I see..."
    st.session_state.messages.append({"role": "assistant", "content": transition_msg})

    st.session_state.trait_map_confirmed = False
    st.session_state.proposition_categories = []
    st.session_state.current_category_index = 0
    st.session_state.awaiting_deal_breakers = False
    st.session_state.deal_breakers_confirmed = False

    with st.spinner("Reflecting on who you are..."):
        ai_response = call_llm(st.session_state.stage_messages, max_tokens=3000)

    if ai_response:
        st.session_state.recovery_pending = trust_recovery.ai_signals_recovery(ai_response)
        ai_response = TrustRecoverySystem.strip_recovery_tag(ai_response)
        st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

def _get_proposition_conversation_text():
    """Build the confirmed conversation text from stage_messages for context."""
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in st.session_state.stage_messages
        if m['role'] in ('assistant', 'user')
    )

def handle_proposition(user_input):
    if st.session_state.recovery_pending == "error1":
        trust_recovery.recover_error1(user_input, st.session_state.stage_messages, st.session_state.user_portrait)
        st.session_state.recovery_pending = None

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.stage_messages.append({"role": "user", "content": user_input})

    user_context_text = (
        f"The user is looking for: {st.session_state.relationship_type}\n\n"
        f"Here is the structured portrait of who they are:\n"
        f"{json.dumps(st.session_state.user_portrait, indent=2)}"
    )

    if not st.session_state.trait_map_confirmed:
        # User just reacted to the trait map — check if confirming or giving feedback
        acceptance_keywords = {
            "yes", "yeah", "yep", "looks good", "that's right", "correct", "good",
            "perfect", "spot on", "exactly", "sure", "ok", "okay", "that's it",
            "all good", "looks right", "looks great", "no changes", "fine", "done"
        }
        user_is_confirming = user_input.lower().strip() in acceptance_keywords

        if not user_is_confirming:
            # User gave feedback on the trait map — re-generate it with their correction
            confirmed_text = _get_proposition_conversation_text()
            trait_messages = [
                {"role": "system", "content": get_trait_map_system_prompt()},
                {"role": "user", "content": user_context_text},
                {"role": "assistant", "content": st.session_state.stage_messages[-2]["content"] if len(st.session_state.stage_messages) >= 2 else ""},
                {"role": "user", "content": f"The user wants to adjust the trait map: {user_input}\n\nPlease re-present the trait map with the user's corrections applied."}
            ]

            with st.spinner("Updating trait map..."):
                ai_response = call_llm(trait_messages, max_tokens=3000)

            if ai_response:
                st.session_state.recovery_pending = trust_recovery.ai_signals_recovery(ai_response)
                ai_response = TrustRecoverySystem.strip_recovery_tag(ai_response)
                st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            # Stay in trait_map_confirmed = False so user can confirm or give more feedback
        else:
            # User confirmed trait map — select categories and show first one
            with st.spinner("Selecting what matters most..."):
                # Select categories (no user interaction)
                select_messages = [
                    {"role": "system", "content": get_select_categories_prompt()},
                    {"role": "user", "content": user_context_text}
                ]
                category_response = call_llm(select_messages, temperature=0.1, max_tokens=3000)

                try:
                    json_start = category_response.find("[")
                    json_end = category_response.rfind("]") + 1
                    st.session_state.proposition_categories = json.loads(category_response[json_start:json_end])
                except Exception:
                    st.session_state.proposition_categories = ["Personality Traits", "Communication Style", "Core Values", "Emotional Needs"]

                transition = "Now let me walk you through what I think matters most, one area at a time."
                st.session_state.messages.append({"role": "assistant", "content": transition})

                # Generate first category
                category_name = st.session_state.proposition_categories[0]
                confirmed_text = _get_proposition_conversation_text()
                cat_messages = [
                    {"role": "system", "content": get_category_system_prompt(st.session_state.relationship_type, category_name)},
                    {"role": "user", "content": f"{user_context_text}\n\nConfirmed trait map conversation so far:\n{confirmed_text}"}
                ]
                ai_response = call_llm(cat_messages, max_tokens=3000)

            if ai_response:
                st.session_state.recovery_pending = trust_recovery.ai_signals_recovery(ai_response)
                ai_response = TrustRecoverySystem.strip_recovery_tag(ai_response)
                st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

            st.session_state.trait_map_confirmed = True
            st.session_state.current_category_index = 0

    elif st.session_state.current_category_index < len(st.session_state.proposition_categories):
        # User just reacted to a category — check if confirming or giving feedback
        acceptance_keywords = {
            "yes", "yeah", "yep", "looks good", "that's right", "correct", "good",
            "perfect", "spot on", "exactly", "sure", "ok", "okay", "that's it",
            "all good", "looks right", "looks great", "no changes", "fine", "done"
        }
        user_is_confirming = user_input.lower().strip() in acceptance_keywords

        if not user_is_confirming:
            # User gave feedback — re-generate the SAME category with their correction
            category_name = st.session_state.proposition_categories[st.session_state.current_category_index]
            confirmed_text = _get_proposition_conversation_text()
            cat_messages = [
                {"role": "system", "content": get_category_system_prompt(st.session_state.relationship_type, category_name)},
                {"role": "user", "content": f"{user_context_text}\n\nConversation so far:\n{confirmed_text}"},
                {"role": "assistant", "content": st.session_state.stage_messages[-2]["content"] if len(st.session_state.stage_messages) >= 2 else ""},
                {"role": "user", "content": f"The user wants to adjust this ranking: {user_input}\n\nPlease re-present this category with the user's corrections applied."}
            ]

            with st.spinner("Updating ranking..."):
                ai_response = call_llm(cat_messages, max_tokens=3000)

            if ai_response:
                st.session_state.recovery_pending = trust_recovery.ai_signals_recovery(ai_response)
                ai_response = TrustRecoverySystem.strip_recovery_tag(ai_response)
                st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
        else:
            # User confirmed — advance to next category or deal breakers
            st.session_state.current_category_index += 1

            if st.session_state.current_category_index < len(st.session_state.proposition_categories):
                # More categories to show
                category_name = st.session_state.proposition_categories[st.session_state.current_category_index]
                confirmed_text = _get_proposition_conversation_text()
                cat_messages = [
                    {"role": "system", "content": get_category_system_prompt(st.session_state.relationship_type, category_name)},
                    {"role": "user", "content": f"{user_context_text}\n\nConfirmed conversation so far:\n{confirmed_text}"}
                ]

                with st.spinner("Thinking..."):
                    ai_response = call_llm(cat_messages, max_tokens=3000)

                if ai_response:
                    st.session_state.recovery_pending = trust_recovery.ai_signals_recovery(ai_response)
                    ai_response = TrustRecoverySystem.strip_recovery_tag(ai_response)
                    st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
            else:
                # All categories done — show deal breakers
                confirmed_text = _get_proposition_conversation_text()
                db_messages = [
                    {"role": "system", "content": get_deal_breakers_system_prompt(st.session_state.relationship_type)},
                    {"role": "user", "content": f"{user_context_text}\n\nFull confirmed conversation so far:\n{confirmed_text}"}
                ]

                with st.spinner("Thinking about deal breakers..."):
                    ai_response = call_llm(db_messages, max_tokens=3000)

                if ai_response:
                    st.session_state.recovery_pending = trust_recovery.ai_signals_recovery(ai_response)
                    ai_response = TrustRecoverySystem.strip_recovery_tag(ai_response)
                    st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})

                st.session_state.awaiting_deal_breakers = True

    elif st.session_state.awaiting_deal_breakers:
        # User just reacted to deal breakers — check if confirming or giving feedback
        acceptance_keywords = {
            "yes", "yeah", "yep", "looks good", "that's right", "correct", "good",
            "perfect", "spot on", "exactly", "sure", "ok", "okay", "that's it",
            "all good", "looks right", "looks great", "no changes", "fine", "done"
        }
        user_is_confirming = user_input.lower().strip() in acceptance_keywords

        if not user_is_confirming:
            # User gave feedback on deal breakers — re-generate with their correction
            confirmed_text = _get_proposition_conversation_text()
            db_messages = [
                {"role": "system", "content": get_deal_breakers_system_prompt(st.session_state.relationship_type)},
                {"role": "user", "content": f"{user_context_text}\n\nFull confirmed conversation so far:\n{confirmed_text}"},
                {"role": "assistant", "content": st.session_state.stage_messages[-2]["content"] if len(st.session_state.stage_messages) >= 2 else ""},
                {"role": "user", "content": f"The user wants to adjust the deal breakers: {user_input}\n\nPlease re-present the deal breakers with the user's corrections applied."}
            ]

            with st.spinner("Updating deal breakers..."):
                ai_response = call_llm(db_messages, max_tokens=3000)

            if ai_response:
                st.session_state.recovery_pending = trust_recovery.ai_signals_recovery(ai_response)
                ai_response = TrustRecoverySystem.strip_recovery_tag(ai_response)
                st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            # Stay in awaiting_deal_breakers = True so user can confirm or give more feedback
        else:
            # User confirmed deal breakers — extract proposition and advance
            st.session_state.awaiting_deal_breakers = False
            st.session_state.deal_breakers_confirmed = True

            confirmed_msg = "Great — I have everything I need. Let's build your profile."
            st.session_state.messages.append({"role": "assistant", "content": confirmed_msg})

            with st.spinner("Finalizing priorities..."):
                st.session_state.proposition_data = extract_proposition(
                    st.session_state.stage_messages,
                    st.session_state.relationship_type
                )
            advance_stage()
            start_tension_stage()

def start_tension_stage():
    system_msg = {"role": "system", "content": TENSION_SYSTEM_PROMPT}
    user_context = {"role": "user", "content": f"Here are the user's inferred priorities: {json.dumps(st.session_state.proposition_data)}"}
    st.session_state.stage_messages = [system_msg, user_context]

    transition_msg = "Let me think through a couple of things with you..."
    st.session_state.messages.append({"role": "assistant", "content": transition_msg})

    with st.spinner("Thinking..."):
        ai_response = call_llm(st.session_state.stage_messages, max_tokens=3000)

    if ai_response:
        st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        st.session_state.round_count = 1

def handle_tension(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.stage_messages.append({"role": "user", "content": user_input})

    # Check if the user just answered the 3rd question — wrap up without asking another
    st.session_state.round_count += 1
    if st.session_state.round_count > 3:
        with st.spinner("Wrapping up..."):
            confirmation_msg = [
                {
                    "role": "system",
                    "content": (
                        "The user just finished answering clarifying questions about their relationship priorities. "
                        "Write a brief confirmation (2-3 sentences max) that acknowledges what they clarified "
                        "in their most recent answer. Be warm and concise. Do NOT ask any more questions."
                    )
                },
                {"role": "user", "content": f"User's last response: {user_input}"}
            ]
            wrap_up = call_llm(confirmation_msg, max_tokens=3000)
            if wrap_up:
                st.session_state.messages.append({"role": "assistant", "content": wrap_up})
            st.session_state.messages.append({"role": "assistant", "content": "Thanks for working through that with me!"})
        advance_stage()
        start_profile_stage()
        return

    with st.spinner("Thinking..."):
        ai_response = call_llm(st.session_state.stage_messages, max_tokens=3000)

    if ai_response:
        st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

        if "resolved" in ai_response.lower():
            advance_stage()
            start_profile_stage()

def start_profile_stage():
    prompt_msg = "Before I build the profile — do you have anything specific in mind? A name, a vibe, a detail you definitely want included? Or should I surprise you?"
    st.session_state.messages.append({"role": "assistant", "content": prompt_msg})
    st.session_state.awaiting_profile_ideas = True

def start_refinement_stage():
    prompt = "Does this feel right to you, or is something off?\n\n(Type **done** when you're happy with it.)"
    st.session_state.messages.append({"role": "assistant", "content": prompt})
    st.session_state.awaiting_initial_refinement = True

    relationship_type = st.session_state.proposition_data.get("relationship_type", "connection")
    st.session_state.stage_messages = [
        {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT.format(relationship_type=relationship_type)},
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
        recovery_type = trust_recovery.ai_signals_recovery(ai_response)
        clean_response = TrustRecoverySystem.strip_recovery_tag(ai_response)

        if recovery_type == "error2":
            # AI detected a false assumption — run full assumption audit
            updated_profile = trust_recovery.recover_error2(
                st.session_state.profile_text,
                st.session_state.user_portrait,
                st.session_state.proposition_data
            )
            if updated_profile:
                st.session_state.profile_text = updated_profile
                st.session_state.frozen_profile = updated_profile
                relationship_type = st.session_state.proposition_data.get("relationship_type", "connection")
                st.session_state.stage_messages = [
                    {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT.format(relationship_type=relationship_type)},
                    {"role": "user", "content": "Here is the current profile:\n\n" + updated_profile},
                    {"role": "assistant", "content": "Profile updated. What else would you like to change?"}
                ]
        elif recovery_type == "error3":
            # AI changed more than asked — revert and apply targeted edit
            corrected = trust_recovery.recover_error3(
                st.session_state.last_feedback or user_input,
                st.session_state.frozen_profile,
                st.session_state.proposition_data
            )
            if corrected:
                st.session_state.frozen_profile = corrected
                st.session_state.profile_text = corrected
                relationship_type = st.session_state.proposition_data.get("relationship_type", "connection")
                st.session_state.stage_messages = [
                    {"role": "system", "content": REFINEMENT_SYSTEM_PROMPT.format(relationship_type=relationship_type)},
                    {"role": "user", "content": "Here is the current profile:\n\n" + corrected},
                    {"role": "assistant", "content": "Profile updated. What else would you like to change?"}
                ]
                st.session_state.last_feedback = None
        else:
            st.session_state.stage_messages.append({"role": "assistant", "content": clean_response})
            st.session_state.messages.append({"role": "assistant", "content": clean_response})
            st.session_state.profile_text = clean_response
            st.session_state.frozen_profile = clean_response
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

    st.markdown("""
        <style>
        .stDeployButton {display: none;}
        button[kind="secondary"] {display: none;}

        .stChatMessage {
            overflow-y: visible !important;
        }
        .stChatMessage > div {
            overflow-y: visible !important;
        }
        .main .block-container {
            overflow-anchor: none;
        }
        .stMarkdown {
            overflow-anchor: none;
        }

        [data-testid="stSidebar"] {
            display: flex !important;
            position: sticky !important;
        }

        @media (max-width: 768px) {
            [data-testid="stSidebar"] > div:first-child {
                width: 180px !important;
            }
            .main {
                margin-left: 0 !important;
            }
        }

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
        This tool builds a detailed profile of your ideal connection — whether that's a
        romantic partner, a close friend, a squash buddy, or anything else — through a
        guided conversation.

        First I'll get to know you as a person. Then I'll use what I've learned to propose
        what I think you'd benefit from in that connection. You'll review, adjust, and
        we'll build the profile together.
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
                st.markdown("What kind of relationship or connection are you looking for? (e.g., romantic partner, close friend, squash buddy, study partner)")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "What kind of relationship or connection are you looking for? (e.g., romantic partner, close friend, squash buddy, study partner)"
            })

        if user_input := st.chat_input("Your response..."):
            st.session_state.relationship_type = user_input
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.spinner("Thinking..."):
                response = f"Great choice! Let's explore what makes a great {user_input.lower() if user_input else 'connection'} for you. First, I'll get to know you as a person."
                st.session_state.messages.append({"role": "assistant", "content": response})
                advance_stage()
                system_msg = {"role": "system", "content": get_about_you_system_prompt(st.session_state.relationship_type)}
                initial_user = {"role": "user", "content": f"Hi — I'm looking for: {st.session_state.relationship_type}"}
                st.session_state.stage_messages = [system_msg, initial_user]
                ai_response = call_llm(st.session_state.stage_messages, max_tokens=3000)
                if ai_response:
                    st.session_state.recovery_pending = trust_recovery.ai_signals_recovery(ai_response)
                    ai_response = TrustRecoverySystem.strip_recovery_tag(ai_response)
                    st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
            st.rerun()

    elif st.session_state.stage == "about_you":
        if st.session_state.get("awaiting_summary_confirmation", False):
            if user_input := st.chat_input("Your response..."):
                with st.chat_message("user"):
                    st.markdown(user_input)
                handle_summary_confirmation(user_input)
                st.rerun()
        else:
            if user_input := st.chat_input("Your response..."):
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.stage_messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                with st.spinner("Thinking..."):
                    ai_response = call_llm(st.session_state.stage_messages, max_tokens=3000)
                if ai_response:
                    st.session_state.recovery_pending = trust_recovery.ai_signals_recovery(ai_response)
                    ai_response = TrustRecoverySystem.strip_recovery_tag(ai_response)
                    st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    if check_stage_completion("about_you", ai_response):
                        st.session_state.messages.append({"role": "assistant", "content": "Does this capture you well? Feel free to correct anything or add something important I missed."})
                        st.session_state.awaiting_summary_confirmation = True
                st.rerun()

    elif st.session_state.stage == "proposition":
        if user_input := st.chat_input("Type 'yes' to confirm, or suggest changes..."):
            with st.chat_message("user"):
                st.markdown(user_input)
            handle_proposition(user_input)
            st.rerun()

    elif st.session_state.stage == "tension":
        if user_input := st.chat_input("Your response..."):
            with st.chat_message("user"):
                st.markdown(user_input)
            handle_tension(user_input)
            st.rerun()

    elif st.session_state.stage == "profile":
        if st.session_state.awaiting_profile_ideas:
            if user_input := st.chat_input("Your response..."):
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.awaiting_profile_ideas = False
                with st.chat_message("user"):
                    st.markdown(user_input)

                relationship_type = st.session_state.proposition_data.get("relationship_type", "connection")
                trait_summary = st.session_state.proposition_data.get("user_trait_summary", "")
                proposition_json = json.dumps(st.session_state.proposition_data.get("selected_dimensions", []), indent=2)

                profile_prompt = PROFILE_SYSTEM_PROMPT.format(
                    relationship_type=relationship_type,
                    trait_summary=trait_summary,
                    proposition_json=proposition_json
                )
                messages = [{"role": "system", "content": profile_prompt}]

                if user_input.lower() not in {"surprise me", "surprise", "no", "nope", ""}:
                    messages.append({"role": "user", "content": f"The user wants to include these ideas: {user_input}"})

                messages.append({
                    "role": "user",
                    "content": (
                        f"Generate a complete profile based on these confirmed priorities: "
                        f"{json.dumps(st.session_state.proposition_data, indent=2)}.\n"
                        f"Select appropriate sections for this relationship type. "
                        f"End with a 'Why This Person Fits You' section. "
                        f"Reflect the ranked priorities and exclude all deal breakers."
                    )
                })

                with st.spinner("Generating your profile..."):
                    ai_response = call_llm(messages, max_tokens=3000)

                if ai_response:
                    st.session_state.profile_text = ai_response
                    st.session_state.frozen_profile = ai_response
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    advance_stage()
                    start_refinement_stage()
                st.rerun()

    elif st.session_state.stage == "refinement":
        if st.session_state.get("awaiting_initial_refinement", False):
            if user_input := st.chat_input("Your response (or 'done' to finish)..."):
                st.session_state.awaiting_initial_refinement = False
                with st.chat_message("user"):
                    st.markdown(user_input)
                handle_refinement(user_input)
                st.rerun()

        # Normal refinement loop
        else:
            if user_input := st.chat_input("Your response (or 'done' to finish)..."):
                with st.chat_message("user"):
                    st.markdown(user_input)
                handle_refinement(user_input)
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
