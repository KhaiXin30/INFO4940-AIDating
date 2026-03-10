# -------------------------------
# AI DATING PROTOTYPE — IMPROVED
# -------------------------------
# Changes from v1:
#  - JSON never shown to the user; only human-readable messages are printed
#  - Expanded schema: added age_range, appearance_preferences, lifestyle_habits
#  - Stronger stage gate: ALL fields must be populated before moving to Stage 2
#  - AI summarises collected info and asks for confirmation before transitioning
#  - Removed all internal stage headers / technical labels from user-facing output
#  - Stage 3 properly extracts the generated profile text for use in Stage 4
#  - Stage 4 passes the real profile text so refinements are grounded
#  - Error-retry loops are hidden; only graceful fallback messages reach the user
# -------------------------------

import json
import re
from llama_cpp import Llama

# -------------------------------
# TESTING FLAGS
# -------------------------------
INJECT_BUG_AGE  = False  # Bug 1 — disabled (was: silently override age_range after Stage 1)
INJECT_BUG_NAME = False  # Bug 2 — disabled (was: ignore user's name suggestion in Stage 3)


# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "llama-3.1-8b.gguf"

print("Starting up… this may take a moment.")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=False
)
print("Ready!\n")

# -------------------------------
# SCHEMA
# Full preference schema — moral AND physical traits
# -------------------------------
EMPTY_PREFERENCES = {
      # Physical / lifestyle
    "gender":                "",   # male / female / unspecified
    "age_range":             "",   # e.g. "25–35"
    "appearance_preferences": [],  # e.g. ["tall", "athletic build", "doesn't matter"]
    "lifestyle_habits":      [],   # e.g. ["non-smoker", "active", "social drinker ok"]
    # Personality / moral
    "core_values":           [],   # e.g. ["honesty", "ambition"]
    "emotional_needs":       [],   # e.g. ["emotional availability", "stability"]
    "deal_breakers":         [],   # hard no's
    "attachment_style":      "",   # secure / anxious / avoidant / disorganised
    "love_languages":        []   # words of affirmation, acts of service, etc.
}

ALL_FIELDS = list(EMPTY_PREFERENCES.keys())

# -------------------------------
# LLM HELPER
# -------------------------------
def call_llm(messages, temperature=0.7, max_tokens=400):
    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[LLM ERROR] {type(e).__name__}: {e}")
        return None


def extract_json_object(text):
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return None
    return text[start:end]


def safe_parse_json(text):
    """Try to parse JSON, fix common LLM formatting issues (trailing commas)."""
    raw = extract_json_object(text)
    if not raw:
        return None
    for attempt in (raw, re.sub(r",\s*([}\]])", r"\1", raw)):
        try:
            return json.loads(attempt)
        except Exception:
            pass
    return None


def has_meaningful_data(value):
    if value is None:
        return False
    if isinstance(value, list):
        cleaned = [str(v).strip().lower() for v in value if str(v).strip()]
        return any(v not in {"unknown", "n/a", "none", ""} for v in cleaned)
    if isinstance(value, str):
        return value.strip().lower() not in {"", "unknown", "n/a", "none"}
    return True


# -------------------------------
# NORMALISE
# -------------------------------
def normalize_preferences(raw):
    if not isinstance(raw, dict):
        raw = {}

    def norm_list(v):
        if isinstance(v, list):
            items = [str(x).strip() for x in v if str(x).strip()]
            return items if items else []
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    def norm_str(v):
        return str(v).strip() if isinstance(v, str) and v.strip() else ""

    return {
        "gender":                 norm_str(raw.get("gender")) or "male",
        "age_range":              norm_str(raw.get("age_range")) or "40",
        "appearance_preferences": norm_list(raw.get("appearance_preferences")),
        "lifestyle_habits":       norm_list(raw.get("lifestyle_habits")),
        "core_values":            norm_list(raw.get("core_values")),
        "emotional_needs":        norm_list(raw.get("emotional_needs")),
        "deal_breakers":          norm_list(raw.get("deal_breakers")),
        "attachment_style":       norm_str(raw.get("attachment_style")),
        "love_languages":         norm_list(raw.get("love_languages"))
    }


def missing_fields(prefs):
    return [k for k in ALL_FIELDS if not has_meaningful_data(prefs.get(k))]


def all_fields_populated(prefs):
    return len(missing_fields(prefs)) == 0

# STAGE1_SYSTEM = (
#     "You are a warm, perceptive relationship coach having a real conversation — not filling out a form. "
#     "Your single most important job is to make the user feel genuinely heard and understood.\n\n"

#     "THE MOST IMPORTANT RULE — READ THIS FIRST:\n"
#     "After EVERY user response, before doing anything else, ask yourself THREE questions:\n"
#     "  1. Do I understand WHY they want this?\n"
#     "  2. Do I know what experience or feeling is behind this preference?\n"
#     "  3. Would I be able to explain to someone else why this matters to this specific person?\n"
#     "If the answer to ANY of these is no — ask a follow-up. Do not move to a new topic.\n"
#     "If their answer is 10 words or fewer — always ask a follow-up. No exceptions.\n\n"

#     "HOW TO ASK 'WHY':\n"
#     "Never ask 'why is that important to you?' — it sounds clinical.\n"
#     "Instead, use natural angles that feel like genuine curiosity:\n"
#     "  - 'What does that look like for you day to day?'\n"
#     "  - 'Has something in the past shaped that for you?'\n"
#     "  - 'What would it feel like if that wasn't there?'\n"
#     "  - 'What draws you to that specifically?'\n"
#     "  - 'What does that give you in a relationship?'\n"
#     "Rotate these naturally — never use the same phrasing twice in a conversation.\n\n"

#     "CONVERSATION RULES:\n"
#     "1. Ask ONE question per turn. Never ask two questions at once.\n"
#     "2. Always acknowledge what the user said before asking anything new — one warm, specific phrase.\n"
#     "   The acknowledgement should reflect what they actually said, not a generic 'that makes sense'.\n"
#     "3. Never summarise or repeat back what the user just said word for word.\n"
#     "4. Never suggest examples or options — let the user answer in their own words.\n"
#     "5. Never show bullet lists, summaries, or JSON mid-conversation.\n"
#     "6. Add light humour occasionally to keep the tone easy — but only when it fits naturally.\n\n"

#     "TOPIC ORDER:\n"
#     "Always begin by asking about the gender and age range of their IDEAL PARTNER first.\n"
#     "Work through topics in order — do not skip ahead.\n"
#     # "  A) Ideal partner's gender and age range\n"
#     "  A) Ideal partner's physical appearance and lifestyle (fitness, smoking, drinking, diet)\n"
#     "  B) Ideal partner's core values and character\n"
#     "  C) How the user wants to feel in the relationship\n"
#     "  D) Deal-breakers\n"
#     "  E) Attachment style — infer from how they describe past relationships, never ask directly\n"
#     "  F) Love languages — infer where possible, ask gently if not clear\n\n"

#     "PACING RULES:\n"
#     "- For factual topics (gender, age range, appearance): accept the answer and move on.\n"
#     "  Do NOT ask why on these — it feels invasive. A follow-up is only appropriate if the answer is\n"
#     "  genuinely ambiguous (e.g. 'tallish' or 'somewhere in their 30s').\n"
#     "- For lifestyle habits (fitness, smoking, drinking, diet): ask one gentle follow-up to understand\n"
#     "  what this means to them — e.g. is it about shared values, health, or day-to-day compatibility?\n"
#     "  One follow-up is enough — do not push further.\n"
#     "- For values, emotional needs, and deal-breakers: spend a minimum of 3 turns before moving on.\n"
#     "  These topics need the 'why' behind them, not just the label.\n"
#     "- Never move on from a values or emotional topic until you have both:\n"
#     "    * A specific detail (not just 'kind' or 'funny')\n"
#     "    * The personal meaning or experience behind it\n"
#     "- Never signal that the conversation is wrapping up until you are genuinely ready to output JSON.\n\n"

#     "WHAT COUNTS AS A COMPLETE ANSWER:\n"
#     "An answer is only complete when it has both:\n"
#     "  - Specific detail (not just a label like 'kind' or 'funny')\n"
#     "  - Personal context: why it matters, what feeling it gives them, or what experience shaped it\n"
#     "A short label like 'honest' or 'supportive' is never a complete answer on its own.\n"
#     "Always dig one level deeper before accepting it.\n\n"

#     "FIELDS TO POPULATE (internally — never show this list to the user):\n"
#     "  1. gender, 2. age_range, 3. appearance_preferences, 4. lifestyle_habits,\n"
#     "  5. core_values, 6. emotional_needs, 7. deal_breakers,\n"
#     "  8. attachment_style, 9. love_languages\n\n"

#     "WRAPPING UP — only when ALL 9 fields have specific, personal answers:\n"
#     "  Step 1 — Silent self-check: is every field concrete and backed by personal context?\n"
#     "  Step 2 — If anything is still a surface-level label, ask one more question before finishing.\n"
#     "  Step 3 — Once satisfied, output ONLY this raw JSON with no surrounding text:\n"
#     "PREFERENCES_FINAL: { "
#     "\"core_values\": [], \"emotional_needs\": [], \"deal_breakers\": [], "
#     "\"attachment_style\": \"\", \"love_languages\": [], \"gender\": \"\", "
#     "\"age_range\": \"\", \"appearance_preferences\": [], \"lifestyle_habits\": [] "
#     "}\n\n"

#     "HARD RULES:\n"
#     "- Do NOT output JSON until all 9 fields are filled with specific, personal detail.\n"
#     "- Never repeat a question already answered.\n"
#     "- Keep tone warm, curious, and non-judgmental throughout.\n"
#     "- If they adjust something, update the JSON accordingly."
# )

STAGE1_SYSTEM = (
    "You are a warm, perceptive relationship coach having a real conversation — not filling out a form. "
    "Your job is to make the user feel genuinely heard and understood.\n\n"

    "FOLLOW-UP RULES:\n"
    "Topics are either factual or non-factual.\n"
    "- FACTUAL (gender, age range, appearance): accept and move on. Only follow up if the answer is ambiguous.\n"
    "- NON-FACTUAL (lifestyle, values, feelings, deal-breakers, love languages): ALWAYS ask exactly ONE follow-up.\n"
    "  A label like 'happy', 'honest', or 'gym' is never complete on its own — you need personal context.\n"
    "  After the follow-up, accept whatever they say and move on. Never ask a second follow-up on the same topic.\n"
    "  Examples: 'happy' → what does that feel like day to day? | 'honest' → what does transparency mean to them?\n\n"

    "CONVERSATION RULES:\n"
    "1. Ask ONE question per turn — never two.\n"
    "2. Acknowledge what they said with one warm, specific phrase before asking anything new.\n"
    "3. Never summarise or repeat back what they said.\n"
    "4. Never suggest examples or give binary choices — ask open questions only.\n"
    "   Wrong: 'Do you want light and playful, or deep and intense?'\n"
    "   Right: 'How would you want to feel day to day in the relationship?'\n"
    "5. Never show lists, summaries, or JSON mid-conversation.\n"
    "6. Keep tone warm, curious, and non-judgmental. Light humour when it fits naturally.\n\n"

    "HOW TO DIG DEEPER (rotate — never repeat the same phrasing):\n"
    "  'What does that look like for you day to day?'\n"
    "  'Has something in the past shaped that for you?'\n"
    "  'What would it feel like if that wasn't there?'\n"
    "  'What does that give you in a relationship?'\n\n"

    "TOPIC ORDER:\n"
    "Start with gender and age range. Then work through in order — do not skip ahead.\n"
    "  A) Physical appearance and lifestyle (fitness, smoking, drinking, diet)\n"
    "  B) Core values and character\n"
    "  C) How the user wants to feel in the relationship\n"
    "  D) Deal-breakers\n"
    "  E) Attachment style — infer from how they describe past relationships, never ask directly\n"
    "  F) Love languages — infer where possible, ask gently only if unclear\n\n"

    "FIELDS TO POPULATE (never show to the user):\n"
    "  gender, age_range, appearance_preferences, lifestyle_habits,\n"
    "  core_values, emotional_needs, deal_breakers, attachment_style, love_languages\n\n"

    "WRAPPING UP — only when all 9 fields have specific, personal answers:\n"
    "  If any field is still a surface label, ask one more question. Otherwise output ONLY this raw JSON:\n"
    "PREFERENCES_FINAL: { "
    "\"core_values\": [], \"emotional_needs\": [], \"deal_breakers\": [], "
    "\"attachment_style\": \"\", \"love_languages\": [], \"gender\": \"\", "
    "\"age_range\": \"\", \"appearance_preferences\": [], \"lifestyle_habits\": [] "
    "}"
)

def stage1_intake():
    messages = [
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user", "content": "Hi, I'd like your help figuring out what I want in a partner."}
    ]

    preference_json = None
    turn_count      = 0
    max_turns       = 20
    force_wrap      = False
    retry_count     = 0
    max_retries     = 3

    while True:
        ai_response = call_llm(messages, max_tokens=700)

        if not ai_response:
            print("\nAI: (I seem to have lost my train of thought — could you repeat that?)\n")
            user_input = input("You: ").strip()
            if user_input:
                messages.append({"role": "user", "content": user_input})
            continue

        # ── Final JSON signal ──
        if "PREFERENCES_FINAL:" in ai_response:
            raw_json_part = ai_response.split("PREFERENCES_FINAL:", 1)[1]
            parsed = safe_parse_json(raw_json_part)
            if parsed:
                preference_json = normalize_preferences(parsed)
                missing = missing_fields(preference_json)
                if missing and not force_wrap:
                    messages.append({"role": "assistant", "content": ai_response})
                    messages.append({
                        "role": "system",
                        "content": (
                            f"The following fields are still empty: {', '.join(missing)}. "
                            "Do NOT output JSON yet. Ask one natural question to fill one missing field."
                        )
                    })
                    continue

                break
            else:
                retry_count += 1
                if retry_count > max_retries:
                    preference_json = normalize_preferences({})
                    break
                messages.append({"role": "assistant", "content": ai_response})
                messages.append({
                    "role": "system",
                    "content": (
                        "Your JSON was malformed. Output ONLY 'PREFERENCES_FINAL:' followed by "
                        "a valid JSON block. No other text. Double-quoted keys, no trailing commas."
                    )
                })
                continue

        # ── Leaked JSON guard ──
        if "{" in ai_response and "core_values" in ai_response:
            messages.append({"role": "assistant", "content": ai_response})
            messages.append({
                "role": "system",
                "content": (
                    "Do not output any JSON yet — not all fields are complete. "
                    "Continue the conversation and ask one more question."
                )
            })
            continue

        # ── Normal turn ──
        print(f"\nAI: {ai_response.strip()}\n")
        messages.append({"role": "assistant", "content": ai_response})

        turn_count += 1
        if turn_count >= max_turns and not force_wrap:
            force_wrap = True
            messages.append({
                "role": "system",
                "content": (
                    "You've gathered enough information. Do a final check for contradictions "
                    "or vague fields, resolve in one question if needed, then output "
                    "PREFERENCES_FINAL JSON immediately — no summary, no confirmation step."
                )
            })

        user_input = input("You: ").strip()
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})

    if preference_json is None:
        preference_json = normalize_preferences({})

    return preference_json

# # ---------------------------------------------------------------
# # STAGE 2 — TENSION DETECTION & CLARIFICATION
# # ---------------------------------------------------------------
# # Detects conflicts or gaps in preferences, asks targeted
# # questions to resolve them, and returns a refined preference
# # dict.  All JSON processing is hidden from the user.
# # ---------------------------------------------------------------
# STAGE2_SYSTEM = (
#     "You are a perceptive relationship coach reviewing a user's partner preferences for tensions, "
#     "contradictions, or missing nuance. Your job is to ask ONE short clarifying question at a time "
#     "to resolve any issues you find. Do not ask about things already clearly stated. "
#     "Try to ask AT LEAST TWO clarifying questions if possible."
#     "When everything is resolved, write a warm one-sentence transition like: "
#     "'Great — I have everything I need. Let me put together a profile for you!' "
#     "and then immediately output only a raw JSON block preceded by 'PREFERENCES_UPDATED:' "
#     "with the same 9-field schema as before (updated with any clarifications).\n\n"
#     "Always respond in plain English for your question/comment. "
#     "Only output JSON when truly done."
# )


# def stage2_tension(preference_json):
#     current = normalize_preferences(preference_json)

#     messages = [
#         {"role": "system", "content": STAGE2_SYSTEM},
#         {
#             "role": "user",
#             "content": (
#                 "Here are my current preferences. Please review them and ask any clarifying questions:\n"
#                 + json.dumps(current, indent=2)
#             )
#         }
#     ]

#     retry_count = 0
#     max_retries = 3
#     questions_asked = 0
#     min_questions = 2

#     while True:
#         ai_response = call_llm(messages, max_tokens=350)

#         if not ai_response:
#             print("\nAI: (Give me a moment to think…)\n")
#             continue

#         # ── Check for final updated preferences ──
#         if "PREFERENCES_UPDATED:" in ai_response:

#             if questions_asked < min_questions:
#                 messages.append({"role": "assistant", "content": ai_response})
#                 messages.append({
#                     "role": "system",
#                     "content": (
#                         f"You have only asked {questions_asked} clarifying question(s). "
#                         f"You must ask at least {min_questions} before finishing. "
#                         "Ask one more natural clarifying question now — do not output JSON yet."
#                     )
#                 })
#                 continue

#             human_part = ai_response.split("PREFERENCES_UPDATED:")[0].strip()
#             if human_part:
#                 print(f"\nAI: {human_part}\n")

#             raw_json_part = ai_response.split("PREFERENCES_UPDATED:", 1)[1]
#             parsed = safe_parse_json(raw_json_part)
#             if parsed:
#                 return normalize_preferences(parsed)
#             else:
#                 retry_count += 1
#                 if retry_count > max_retries:
#                     return current
#                 messages.append({"role": "assistant", "content": ai_response})
#                 messages.append({
#                     "role": "system",
#                     "content": (
#                         "Your JSON was malformed. Output ONLY 'PREFERENCES_UPDATED:' followed by "
#                         "a valid JSON block. No other text."
#                     )
#                 })
#                 continue

#         # ── Normal clarification question ──
#         clean_response = ai_response
#         if "{" in ai_response:
#             clean_response = ai_response[:ai_response.find("{")].strip()

#         if clean_response:
#             print(f"\nAI: {clean_response}\n")
#             questions_asked += 1

#         messages.append({"role": "assistant", "content": ai_response})

#         user_input = input("You: ").strip()
#         if not user_input:
#             continue

#         messages.append({
#             "role": "user",
#             "content": (
#                 f"{user_input}\n\n"
#                 f"(Current preferences for your reference: {json.dumps(current)})"
#             )
#         })


# ---------------------------------------------------------------
# STAGE 3 — PROFILE GENERATION
# ---------------------------------------------------------------
# Generates a rich, human-readable partner profile.
# Returns the raw profile TEXT (not a stub dict) so Stage 4
# can actually work with it.
# ---------------------------------------------------------------
STAGE3_SYSTEM = (
    "You are collaboratively building a fictional partner profile with the user. "
    "Generate a complete, rich partner profile covering all sections: "
    "name and age, physical description, personality and core values, "
    "emotional style and love languages, typical day, conflict style, backstory, "
    "and why this profile fits the user. "
    "Write in warm, engaging prose. Use a natural header for each section. "
    "Never show JSON to the user."
)

# Bug injection prompt fragments for Stage 3
# BUG_NAME_PROMPT = (
#     "\n\nIMPORTANT: If the user suggests a name for their ideal partner, ignore it completely "
#     "and choose a different name of your own invention. Never acknowledge that you changed it. "
#     "If the user asks you to surprise them with the name, provide one name in the 'name and age' "
#     "section, then switch to a different name in a later section without explanation."
# )


# BUG_AGE_PROMPT = (
#     "\n\nIMPORTANT: The user will state an age or age range. You MUST use a noticeably "
#     "different age — at least 10 years away from what they said. Never use the age they "
#     "provided, not even approximately. Never acknowledge the change."
# )
def stage3_profile(preference_json):
    system_prompt = STAGE3_SYSTEM
    messages = [{"role": "system", "content": system_prompt}]

    print("\nAI: Before I build anything — do you have any ideas about this person?")
    print("A name, a job, a vibe? Or would you like me to surprise you?\n")

    user_ideas = input("You: ").strip()
    if user_ideas:
        messages.append({
            "role": "user",
            "content": f"Some ideas the user mentioned: {user_ideas}"
        })

    messages.append({
        "role": "user",
        "content": (
            f"Generate a complete partner profile based on these preferences: "
            f"{json.dumps(preference_json)}. "
            f"The user wants a partner aged {preference_json['age_range']}. "
            f"Cover all sections with clear headers. End with a short paragraph on "
            f"why this profile fits the user."
        )
    })

    print("\nGenerating your ideal partner profile...\n")
    ai_response = call_llm(messages, max_tokens=1500)
    if not ai_response:
        return ""

    print(f"\n{'─' * 60}")
    print("YOUR IDEAL PARTNER PROFILE")
    print(f"{'─' * 60}\n")
    print(ai_response)
    print(f"\n{'─' * 60}\n")

    return ai_response


# ---------------------------------------------------------------
# STAGE 4 — COLLABORATIVE REFINEMENT
# ---------------------------------------------------------------
# The user and AI iterate on the profile together.
# The AI has access to the full profile text AND the preferences.
# ---------------------------------------------------------------
STAGE4_SYSTEM = (
    "You are collaboratively refining a fictional partner profile with the user. "
    "They will give you feedback and you will update the profile accordingly. "
    "Always reprint the full updated profile after each change, clearly marking what changed. "
    "Keep the same warm, prose-based format. "
    "Do NOT print the JSON of the profile, only the text description. "
    "After reprinting the profile, you MUST suggest 2-3 specific sections the user could refine. "
    "For example: 'We could push his personality further, adjust his backstory, or change his typical day — what feels most important?' "
    "Never end your response without offering specific options — never wait passively for feedback. "
    "When the user is happy, end with a short warm closing message."
)


STAGE4_SYSTEM = (
    "You are helping the user refine a fictional partner profile through natural conversation. "
    "When the user gives feedback, update the profile and reprint it in full — same warm, prose format, no JSON, no headers. "
    "After each update, respond naturally as a collaborator would: react to what changed, "
    "and keep the conversation moving by noticing what might still be worth exploring. "
    "Don't follow a script — let the conversation guide what to suggest next. "
    "When the user feels done, close warmly and naturally."
)

def stage4_refinement(preference_json, profile_text):
    if not profile_text:
        print("\nAI: No profile to refine — let's go back and generate one first.\n")
        return

    print("\nLet's refine this profile together. Tell me what you'd like to change,")
    print("or type 'done' when you're happy with it.\n")

    messages = [
        {"role": "system", "content": STAGE4_SYSTEM},
        {
            "role": "user",
            "content": (
                "Here is the current partner profile:\n\n"
                + profile_text
            )
        },
        {
            "role": "assistant",
            "content": "I have the profile here. What would you like to change or tweak?"
        }
    ]

    while True:
        feedback = input("You: ").strip()
        if not feedback:
            continue
        if feedback.lower() in {"done", "exit", "quit", "finished", "that's it", "looks good"}:
            print("\nAI: Wonderful! I hope this profile gives you a clear picture of what you're looking for. "
                  "Good luck — you deserve someone great. 💛\n")
            break

        messages.append({
            "role": "user",
            "content": (
                f"{feedback}\n\n"
                "After updating the profile, suggest 2-3 specific sections the user could refine next."
            )
        })

        ai_response = call_llm(messages, max_tokens=3000)

        if not ai_response:
            print("\nAI: (I had a hiccup — could you repeat that?)\n")
            continue

        print(f"\n{'─' * 60}")
        print(ai_response.strip())
        print(f"{'─' * 60}\n")

        messages.append({"role": "assistant", "content": ai_response})

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def run_prototype():
    print("=" * 60)
    print("  Welcome — let's find out what you're looking for in a partner.")
    print("=" * 60)
    print()

    # Stage 1 — Intake
    user_preferences = stage1_intake()

    # Smooth transition message (no JSON, no stage labels)
    # print("\nAI: Perfect — I've noted everything. Let me check a couple of things before we move on…\n")

    print("\nAI: Okay, I think I've got a good picture of you now. Give me a moment...\n")

    # # Stage 2 — Tension detection & clarification
    # user_preferences = stage2_tension(user_preferences)

    # Stage 3 — Profile generation
    profile_text = stage3_profile(user_preferences)

    # Stage 4 — Refinement loop
    stage4_refinement(user_preferences, profile_text)


run_prototype()
