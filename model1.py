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
# CONFIG
# -------------------------------
MODEL_PATH = "qwen2-7b-instruct-q4_0.gguf"

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
    # Personality / moral
    "core_values":           [],   # e.g. ["honesty", "ambition"]
    "emotional_needs":       [],   # e.g. ["emotional availability", "stability"]
    "deal_breakers":         [],   # hard no's
    "attachment_style":      "",   # secure / anxious / avoidant / disorganised
    "love_languages":        [],   # words of affirmation, acts of service, etc.
    # Physical / lifestyle
    "age_range":             "",   # e.g. "25–35"
    "appearance_preferences": [],  # e.g. ["tall", "athletic build", "doesn't matter"]
    "lifestyle_habits":      [],   # e.g. ["non-smoker", "active", "social drinker ok"]
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
        "core_values":            norm_list(raw.get("core_values")),
        "emotional_needs":        norm_list(raw.get("emotional_needs")),
        "deal_breakers":          norm_list(raw.get("deal_breakers")),
        "attachment_style":       norm_str(raw.get("attachment_style")),
        "love_languages":         norm_list(raw.get("love_languages")),
        "age_range":              norm_str(raw.get("age_range")),
        "appearance_preferences": norm_list(raw.get("appearance_preferences")),
        "lifestyle_habits":       norm_list(raw.get("lifestyle_habits")),
    }


def missing_fields(prefs):
    return [k for k in ALL_FIELDS if not has_meaningful_data(prefs.get(k))]


def all_fields_populated(prefs):
    return len(missing_fields(prefs)) == 0


# ---------------------------------------------------------------
# STAGE 1 — COLLABORATIVE INTAKE
# ---------------------------------------------------------------
# Goal: gather ALL 8 fields through natural conversation.
# The AI asks one question at a time, covers personality AND
# physical/lifestyle topics, then summarises and confirms with
# the user before handing off to Stage 2.
# JSON is NEVER shown to the user.
# ---------------------------------------------------------------
STAGE1_SYSTEM = (
    "You are a warm, perceptive relationship coach helping the user discover what they truly want in a long-term partner. "
    "Your job is to gather information through friendly, natural conversation — one question at a time. "
    "Never overwhelm the user. Never repeat a question they've already answered. "
    "Cover BOTH personality traits AND physical/lifestyle preferences — don't skip either. "
    "You must eventually populate ALL of these fields:\n"
    "  1. core_values         — key character traits or life values they want (e.g. ambition, kindness, honesty)\n"
    "  2. emotional_needs     — how they need to feel in the relationship (e.g. secure, heard, calm)\n"
    "  3. deal_breakers       — absolute hard no's\n"
    "  4. attachment_style    — one of: secure, anxious, avoidant, disorganised\n"
    "  5. love_languages      — words of affirmation, quality time, acts of service, physical touch, gifts\n"
    "  6. age_range           — preferred age range of a partner (e.g. '25–35')\n"
    "  7. appearance_preferences — physical traits that matter to them (can include 'doesn't matter')\n"
    "  8. lifestyle_habits    — preferences on smoking, drinking, fitness, diet, etc.\n\n"
    "When ALL fields have concrete answers, output a brief, warm summary of what you've learned "
    "(in plain conversational English — NO JSON), then ask the user: "
    "'Does this feel right, or is there anything you'd like to adjust before we move on?' "
    "Wait for their confirmation. Once they confirm (or make small adjustments), "
    "output ONLY a raw JSON block (no surrounding text) with this exact structure:\n"
    "PREFERENCES_FINAL: { "
    "\"core_values\": [], \"emotional_needs\": [], \"deal_breakers\": [], "
    "\"attachment_style\": \"\", \"love_languages\": [], "
    "\"age_range\": \"\", \"appearance_preferences\": [], \"lifestyle_habits\": [] "
    "}\n\n"
    "Rules:\n"
    "- Do NOT output JSON until all 8 fields are filled AND the user has confirmed.\n"
    "- If they adjust something in their confirmation reply, update the JSON accordingly.\n"
    "- After at most 12 questions without completing all fields, do your best with available data, "
    "summarise, and ask for confirmation.\n"
    "- Keep the tone warm, curious, and non-judgmental."
)


def stage1_intake():
    messages = [
        {"role": "system", "content": STAGE1_SYSTEM},
        {"role": "user", "content": "Hi, I'd like your help figuring out what I want in a partner."}
    ]

    preference_json = None
    turn_count = 0
    max_turns = 14          # generous limit before forced wrap-up
    force_wrap = False
    retry_count = 0
    max_retries = 3

    while True:
        ai_response = call_llm(messages, max_tokens=350)

        # Graceful failure if model returns nothing
        if not ai_response:
            print("\nAI: (I seem to have lost my train of thought — could you repeat that?)\n")
            user_input = input("You: ").strip()
            if user_input:
                messages.append({"role": "user", "content": user_input})
            continue

        # ── Check if AI has emitted the final JSON signal ──
        if "PREFERENCES_FINAL:" in ai_response:
            raw_json_part = ai_response.split("PREFERENCES_FINAL:", 1)[1]
            parsed = safe_parse_json(raw_json_part)
            if parsed:
                preference_json = normalize_preferences(parsed)
                missing = missing_fields(preference_json)
                if missing and not force_wrap:
                    # Fields still empty — ask AI to fill them before finishing
                    messages.append({"role": "assistant", "content": ai_response})
                    messages.append({
                        "role": "system",
                        "content": (
                            f"The following fields are still empty: {', '.join(missing)}. "
                            "Do NOT output the final JSON yet. "
                            "Ask one natural question to fill one of those missing fields."
                        )
                    })
                    continue
                # All good — exit Stage 1 silently
                break
            else:
                # JSON was malformed — ask AI to retry
                retry_count += 1
                if retry_count > max_retries:
                    # Give up and use whatever we have (or defaults)
                    preference_json = normalize_preferences({})
                    break
                messages.append({"role": "assistant", "content": ai_response})
                messages.append({
                    "role": "system",
                    "content": (
                        "Your JSON was malformed. Output ONLY the raw JSON block with no surrounding text, "
                        "preceded by 'PREFERENCES_FINAL:'. Use strict JSON (double-quoted keys, no trailing commas)."
                    )
                })
                continue

        # ── Normal conversational turn ──
        # Show only the human-readable part (strip any accidental JSON fragments)
        user_facing = ai_response
        if "{" in ai_response and "core_values" in ai_response:
            # AI leaked partial JSON — strip it, ask it to rephrase
            messages.append({"role": "assistant", "content": ai_response})
            messages.append({
                "role": "system",
                "content": (
                    "Do not output any JSON yet — not all fields are filled and the user hasn't confirmed. "
                    "Continue the conversation naturally and ask one more question."
                )
            })
            continue

        print(f"\nAI: {user_facing.strip()}\n")
        messages.append({"role": "assistant", "content": ai_response})

        # ── Force wrap-up after max turns ──
        turn_count += 1
        if turn_count >= max_turns and not force_wrap:
            force_wrap = True
            messages.append({
                "role": "system",
                "content": (
                    "You've asked enough questions. Summarise what you've learned in warm plain English, "
                    "fill any missing fields with your best inference, "
                    "then ask the user to confirm before outputting PREFERENCES_FINAL JSON."
                )
            })

        user_input = input("You: ").strip()
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})

    # Fallback if we somehow exit without a valid object
    if preference_json is None:
        preference_json = normalize_preferences({})

    return preference_json


# ---------------------------------------------------------------
# STAGE 2 — TENSION DETECTION & CLARIFICATION
# ---------------------------------------------------------------
# Detects conflicts or gaps in preferences, asks targeted
# questions to resolve them, and returns a refined preference
# dict.  All JSON processing is hidden from the user.
# ---------------------------------------------------------------
STAGE2_SYSTEM = (
    "You are a perceptive relationship coach reviewing a user's partner preferences for tensions, "
    "contradictions, or missing nuance. Your job is to ask ONE short clarifying question at a time "
    "to resolve any issues you find. Do not ask about things already clearly stated. "
    "When everything is resolved, write a warm one-sentence transition like: "
    "'Great — I have everything I need. Let me put together a profile for you!' "
    "and then immediately output only a raw JSON block preceded by 'PREFERENCES_UPDATED:' "
    "with the same 8-field schema as before (updated with any clarifications).\n\n"
    "Always respond in plain English for your question/comment. "
    "Only output JSON when truly done."
)


def stage2_tension(preference_json):
    current = normalize_preferences(preference_json)

    messages = [
        {"role": "system", "content": STAGE2_SYSTEM},
        {
            "role": "user",
            "content": (
                "Here are my current preferences. Please review them and ask any clarifying questions:\n"
                + json.dumps(current, indent=2)
            )
        }
    ]

    retry_count = 0
    max_retries = 3

    while True:
        ai_response = call_llm(messages, max_tokens=350)

        if not ai_response:
            print("\nAI: (Give me a moment to think…)\n")
            continue

        # ── Check for final updated preferences ──
        if "PREFERENCES_UPDATED:" in ai_response:
            # Print only the conversational part (before the JSON signal)
            human_part = ai_response.split("PREFERENCES_UPDATED:")[0].strip()
            if human_part:
                print(f"\nAI: {human_part}\n")

            raw_json_part = ai_response.split("PREFERENCES_UPDATED:", 1)[1]
            parsed = safe_parse_json(raw_json_part)
            if parsed:
                return normalize_preferences(parsed)
            else:
                retry_count += 1
                if retry_count > max_retries:
                    return current  # fall back to what we had
                messages.append({"role": "assistant", "content": ai_response})
                messages.append({
                    "role": "system",
                    "content": (
                        "Your JSON was malformed. Output ONLY 'PREFERENCES_UPDATED:' followed by "
                        "a valid JSON block. No other text."
                    )
                })
                continue

        # ── Normal clarification question ──
        # Strip any accidental JSON leakage
        clean_response = ai_response
        if "{" in ai_response:
            clean_response = ai_response[:ai_response.find("{")].strip()

        if clean_response:
            print(f"\nAI: {clean_response}\n")

        messages.append({"role": "assistant", "content": ai_response})

        user_input = input("You: ").strip()
        if not user_input:
            continue

        messages.append({
            "role": "user",
            "content": (
                f"{user_input}\n\n"
                f"(Current preferences for your reference: {json.dumps(current)})"
            )
        })


# ---------------------------------------------------------------
# STAGE 3 — PROFILE GENERATION
# ---------------------------------------------------------------
# Generates a rich, human-readable partner profile.
# Returns the raw profile TEXT (not a stub dict) so Stage 4
# can actually work with it.
# ---------------------------------------------------------------
STAGE3_SYSTEM = (
    "You are crafting a vivid, believable fictional partner profile tailored to the user's preferences. "
    "Write it in warm, engaging prose — like a detailed character sketch, not a list. "
    "Include:\n"
    "  • Name and age\n"
    "  • Physical description (grounded in their appearance preferences)\n"
    "  • Personality and core values\n"
    "  • Emotional style and how they show love\n"
    "  • A typical day in their life\n"
    "  • How they handle conflict\n"
    "  • A short backstory\n"
    "  • Why this profile fits the user's preferences (1–2 sentences)\n\n"
    "End with one gentle question asking the user what they think and whether anything feels off."
)


def stage3_profile(preference_json):
    messages = [
        {"role": "system", "content": STAGE3_SYSTEM},
        {
            "role": "user",
            "content": (
                "Here are my preferences. Please create a partner profile for me:\n"
                + json.dumps(preference_json, indent=2)
            )
        }
    ]

    ai_response = call_llm(messages, max_tokens=800)

    if not ai_response:
        print("\nAI: I'm having trouble generating the profile right now. Let's try again.\n")
        return None

    print(f"\n{'─' * 60}")
    print("YOUR IDEAL PARTNER PROFILE")
    print(f"{'─' * 60}\n")
    print(ai_response.strip())
    print(f"\n{'─' * 60}\n")

    return ai_response  # Return the actual text, not a placeholder


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
    "After each update, ask if there's anything else they'd like to change. "
    "When the user is happy, end with a short warm closing message."
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
                + "\n\nMy original preferences for reference:\n"
                + json.dumps(preference_json, indent=2)
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

        messages.append({"role": "user", "content": feedback})
        ai_response = call_llm(messages, max_tokens=800)

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
    print("\nAI: Perfect — I've noted everything. Let me check a couple of things before we move on…\n")

    # Stage 2 — Tension detection & clarification
    user_preferences = stage2_tension(user_preferences)

    # Stage 3 — Profile generation
    profile_text = stage3_profile(user_preferences)

    # Stage 4 — Refinement loop
    stage4_refinement(user_preferences, profile_text)


run_prototype()
