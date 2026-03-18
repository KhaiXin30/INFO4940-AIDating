# -------------------------------
# AI RELATIONSHIP PROFILE BUILDER (v2 — Combined)
# -------------------------------
# 5-stage architecture:
#   Stage 1: About You — personality-informed questions, no partner preferences
#   Stage 2: The Proposition — trait map + inferred priorities with framework ranking
#   Stage 3: Tension Detection — surface contradictions, max 3 clarifying questions
#   Stage 4: Profile Generation — dynamic sections based on relationship type
#   Stage 5: Refinement — polish loop with trust recovery (Error 2 + Error 3)
# -------------------------------

import json

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "llama-3.1-8b.gguf"
TEST_MODE = False  # Set to True to run flow test without the real model

# -------------------------------
# TEST MODE — mock LLM responses and user inputs
# -------------------------------
_test_llm_idx = 0
_test_input_idx = 0

# Mock LLM responses (15) — ordered to match the exact call sequence:
#   Stage 1: 5 personality questions (0-4) + 1 SUMMARY (5)
#   extract_user_portrait: JSON (6)
#   Stage 2 round 1: trait map + ranked priorities (7)
#   Stage 2 round 2: PROPOSITION CONFIRMED (8)
#   extract_proposition: JSON (9)
#   Stage 3: 2 tension questions + wrap-up confirmation (10-12)
#   Stage 4: profile text (13)
#   Stage 5: refinement suggestions (14)
_TEST_LLM_RESPONSES = [
    # Stage 1 — learning about the user (personality only, no partner prefs)
    "What's something you find yourself drawn to — whether it's an activity, a topic, or a way of spending your time — that you think says a lot about who you are?",
    "That's really interesting. When you're around people, what's your natural mode — do you tend to take the lead in conversations, or do you prefer to listen and observe first?",
    "I can picture that. How do you tend to handle it when plans change unexpectedly — does that energize you or throw you off?",
    "That makes sense. When you're facing a tough decision — especially one that affects someone else — do you tend to lead with logic or with how it'll make people feel?",
    "Got it. One more thing — when you think about trying something completely new and unfamiliar, what's your gut reaction?",
    # SUMMARY triggers end of Stage 1 loop
    "SUMMARY: You are a thoughtful, introspective person who values depth and meaning in your daily life. You're drawn to reading, hiking, and long conversations about abstract ideas — pursuits that reflect your curiosity and preference for substance over surface. Socially, you tend to listen first and contribute when you have something meaningful to say, suggesting a quieter but deeply engaged social energy. You handle unexpected changes with mild discomfort but adapt well, pointing to a preference for structure with enough flexibility to go with the flow. Your decision-making leans toward empathy — you consider how things affect others before defaulting to logic. And when it comes to novelty, you're genuinely curious but prefer to ease in rather than dive headfirst, suggesting openness balanced with caution.",
    # extract_user_portrait — valid JSON
    '{"personality_traits": ["introspective", "curious", "empathetic", "thoughtful"], "communication_style": "Listens first, contributes when meaningful — reserved but deeply engaged", "values": ["authenticity", "depth", "growth", "kindness"], "lifestyle": "Enjoys reading, hiking, long conversations; prefers substance over surface", "social_energy": "Introverted — recharges alone, values small meaningful connections", "thinking_style": "Abstract and reflective, drawn to big-picture ideas", "decision_making": "Empathy-first — considers impact on others before logic", "structure_vs_spontaneity": "Prefers structure but adapts when needed", "openness_to_experience": "Curious but cautious — eases into novelty rather than diving in", "relationship_tendencies": "Supportive, values emotional depth, shows care through attentiveness", "big_five_estimates": {"extraversion": "low", "openness": "medium-high", "agreeableness": "high", "conscientiousness": "medium-high", "neuroticism": "low-medium"}}',
    # Stage 2 round 1 — trait map + inferred priorities with framework ranking
    "Based on what you've shared, here's what I see in you:\n\nYou're someone who moves through the world thoughtfully and deliberately. Your social energy leans introverted — you don't need a crowd to feel connected, but when you do engage, you bring real depth to the conversation. You're a big-picture thinker who values ideas and meaning, and you make decisions with empathy at the center. You like having a plan, but you're not rigid about it. And you're genuinely curious about new experiences, as long as you can approach them at your own pace.\n\n**Personality Traits (informed by MBTI: INFJ pattern + Enneagram Type 4/5 blend):**\n1. Emotional intelligence — someone who reads between the lines and meets you where you are\n2. Warmth — genuine kindness, not performative niceness\n3. Intellectual curiosity — a mind that lights up at ideas, not just small talk\n4. Patience — comfortable with your reflective pace\n\n**Communication Style (informed by DiSC: Steadiness pattern):**\n1. Thoughtful listener — someone who hears what you're actually saying\n2. Direct but gentle — honest without being blunt\n3. Comfortable with silence — doesn't need to fill every gap\n\n**Core Values:**\n1. Authenticity — what you see is what you get\n2. Growth orientation — always becoming, never stagnant\n3. Loyalty — shows up consistently\n\n**Deal Breakers:**\n1. Emotional unavailability\n2. Dismissiveness or contempt\n3. Superficiality\n\nHow does this land? Anything you'd reorder, add, or remove?",
    # Stage 2 round 2 — user confirms
    "PROPOSITION CONFIRMED",
    # extract_proposition — valid JSON
    '{"relationship_type": "romantic partner", "user_trait_summary": "Introspective, empathetic, curious. Introverted social energy, abstract thinker, empathy-first decisions, prefers structure with flexibility, cautiously open to novelty.", "selected_dimensions": [{"category": "Personality Traits", "ranked_items": [{"item": "Emotional intelligence", "reasoning": "Matches user empathy-first style (INFJ pattern)"}, {"item": "Warmth and genuine kindness", "reasoning": "Complements user high agreeableness"}, {"item": "Intellectual curiosity", "reasoning": "Matches user abstract thinking style"}, {"item": "Patience", "reasoning": "Supports user reflective pace"}]}, {"category": "Communication Style", "ranked_items": [{"item": "Thoughtful listener", "reasoning": "Matches user reserved engagement style (DiSC Steadiness)"}, {"item": "Direct but gentle", "reasoning": "User values authenticity without harshness"}, {"item": "Comfortable with silence", "reasoning": "Respects user introverted recharge needs"}]}, {"category": "Core Values", "ranked_items": [{"item": "Authenticity", "reasoning": "User top value"}, {"item": "Growth orientation", "reasoning": "User drawn to depth and becoming"}, {"item": "Loyalty", "reasoning": "User values consistent presence"}]}, {"category": "Emotional Needs", "ranked_items": [{"item": "Emotional availability", "reasoning": "User empathy-first — needs reciprocation"}, {"item": "Deep conversation", "reasoning": "User primary connection mode"}, {"item": "Stability with warmth", "reasoning": "Matches structure preference + high agreeableness"}]}], "deal_breakers": ["Emotional unavailability", "Dismissiveness or contempt", "Superficiality"]}',
    # Stage 3 — tension questions
    "I noticed something interesting — you value both intellectual curiosity and patience. Sometimes the most intellectually driven people can be impatient or restless. How important is it that this person matches your intellectual energy versus simply being a calm, grounding presence?",
    "That makes a lot of sense. You also mentioned you prefer structure but you're adaptable — and you want someone who's growing. How do you feel about someone who's more spontaneous and shakes up your routine in pursuit of growth?",
    # Stage 3 — wrap-up ("resolved" keyword triggers loop exit, no user input consumed)
    "I think those tensions are resolved. You want intellectual depth and emotional warmth to coexist — and growth that happens within a stable foundation, not by constantly uprooting things. I've got what I need to build your profile.",
    # Stage 4 — profile text
    "## Meet Elara\n\n### Name & Age\nElara, 29\n\n### Personality & Core Traits\nElara is the kind of person who notices the thing no one else caught — the shift in someone's expression, the undercurrent in a conversation, the quiet detail in a painting everyone else walked past. She's deeply empathetic without being fragile, intellectually sharp without being cold. She thinks in patterns and connections, drawn to ideas that sit at the intersection of philosophy, psychology, and everyday life. She's not loud about her intelligence — it shows up in the questions she asks and the way she listens.\n\n### Communication Style\nElara listens first. When she speaks, it's considered — she chooses her words carefully, not out of anxiety but out of respect for what words can do. She's direct when it matters, but wraps honesty in warmth. Silence with her isn't awkward; it's comfortable. She doesn't fill space for the sake of it.\n\n### Emotional Style & Love Languages\nHer primary love language is quality time — she'd rather spend two undistracted hours with you than an entire busy weekend. Words of affirmation matter to her too, but only when they're specific and real. She's emotionally available in the way that counts: she'll sit with difficult feelings instead of rushing to fix them.\n\n### A Typical Interaction\nA Saturday afternoon with Elara might start with a farmers' market walk where she tells you about something she read that week — not to show off, but because she genuinely wants to know what you think about it. You'd end up at a quiet cafe, talking for longer than intended, and she'd notice when your energy shifts before you do.\n\n### Conflict Style\nElara doesn't avoid conflict, but she doesn't escalate it either. She takes a breath, names what she's feeling, and asks what's going on for you. She believes most conflicts come from misunderstanding, not malice — and she treats them that way.\n\n### Backstory\nElara grew up in a university town, the daughter of a librarian and a high school science teacher. Books were everywhere. She studied comparative literature, then pivoted to UX research — a field that lets her combine her love of understanding people with her analytical mind. She's been in one serious relationship that ended amicably when they realized they were growing in different directions.\n\n### Why This Person Fits You\nElara mirrors your depth without duplicating it. Where you lean toward abstract reflection, she brings grounded empathy. She meets your intellectual curiosity head-on but never makes connection feel like a debate to win. Her patience matches your reflective pace, and her emotional availability addresses the core need you identified. She's someone who grows within stability — exactly the balance you described.",
    # Stage 5 — refinement suggestions
    "A few things you might want to personalize:\n- Elara's specific hobbies beyond reading — does she play an instrument, cook, garden?\n- The tone of her backstory — would you like her to have overcome something, or is the calm upbringing right?\n- Any physical details you'd like to add or adjust.",
]

# Mock user inputs (12) — ordered to match the exact input() call sequence
_TEST_USER_INPUTS = [
    # Upfront relationship type question (run_prototype)
    "A romantic partner — someone for a serious long-term relationship.",
    # Stage 1 — 5 personality answers (no partner prefs asked)
    "I'm drawn to reading and hiking — I like things that let me think deeply or be in nature. Long conversations about abstract ideas are my favorite thing.",
    "I'm definitely more of a listener. I observe first, then contribute when I have something meaningful to say.",
    "Honestly, it throws me off a bit at first, but I adapt. I like having a plan, but I'm not rigid about it.",
    "I lean toward how it'll make people feel. I always consider the emotional impact first, then the logic.",
    "I'm curious about it, but I like to ease in rather than jump headfirst. I warm up to new things on my own terms.",
    # Stage 2 — 1 answer after seeing trait map + priorities
    "This is really accurate. I'd keep everything as is — maybe just emphasize that emotional availability is the single most important thing to me.",
    # Stage 3 — 2 tension answers (3rd turn is wrap-up, no input needed)
    "Both matter, but if I had to choose, emotional presence wins. I'd rather be with someone warm and grounding than someone who matches me intellectually but feels distant.",
    "I'd be open to some spontaneity, but I'd want the foundation to feel stable. Growth through adventure is fine as long as we're not constantly in chaos.",
    # Stage 4 — user ideas prompt
    "Surprise me.",
    # Stage 5 — profile check
    "Yes, this feels right.",
    # Stage 5 — done
    "done",
]


def _mock_input(prompt=""):
    global _test_input_idx
    print(prompt, end="", flush=True)
    if _test_input_idx < len(_TEST_USER_INPUTS):
        val = _TEST_USER_INPUTS[_test_input_idx]
        print(val)
        _test_input_idx += 1
        return val
    print("done")
    return "done"


if TEST_MODE:
    input = _mock_input

# -------------------------------
# MODEL LOADING (skipped in test mode)
# -------------------------------
if not TEST_MODE:
    from llama_cpp import Llama
    print("Loading model... (this may take a moment)")
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_gpu_layers=-1,
        verbose=False
    )
    print("Model loaded!")

# -------------------------------
# HELPER: CALL LLM
# -------------------------------
def call_llm(messages, temperature=0.7, max_tokens=24000):
    if TEST_MODE:
        global _test_llm_idx
        if _test_llm_idx < len(_TEST_LLM_RESPONSES):
            response = _TEST_LLM_RESPONSES[_test_llm_idx]
            print(f"  [TEST: mock LLM response #{_test_llm_idx}]")
            _test_llm_idx += 1
            return response
        return "I think we have everything we need."
    try:
        response = llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print("Error during inference:", e)
        return None


# ===============================
# DIMENSION MENU
# ===============================
# The LLM selects 4-6 of these based on relationship type.
# Tier 1 (universal) should almost always be included.
# Tier 2/3 are pulled in based on context.

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


# ===================================================================
# TRUST RECOVERY SYSTEM
# ===================================================================
#
# All three recovery paths are fully REACTIVE. No extra LLM calls are
# ever made for detection — detection is entirely keyword matching on
# text that already exists in the natural conversation flow.
#
# ┌─────────┬──────────────────────────────┬───────────────────────────────────┐
# │ Error   │ What triggers detection      │ Cost of detection                 │
# ├─────────┼──────────────────────────────┼───────────────────────────────────┤
# │ Error 1 │ Confusion marker phrases in  │ Zero — keyword scan on the AI's   │
# │         │ the AI's own response.       │ existing response text.           │
# │         │ AI system prompt instructs   │                                   │
# │         │ it to use specific markers   │                                   │
# │         │ when confused.               │                                   │
# ├─────────┼──────────────────────────────┼───────────────────────────────────┤
# │ Error 2 │ Dissatisfaction in the       │ Zero — keyword scan on the user's │
# │         │ user's reply to a lightweight│ reply to the post-profile check.  │
# │         │ "does this feel right?" ask  │                                   │
# │         │ after profile generation.    │                                   │
# ├─────────┼──────────────────────────────┼───────────────────────────────────┤
# │ Error 3 │ Over-scope complaint in the  │ Zero — keyword scan on the user's │
# │         │ user's NEXT message after    │ next refinement message.          │
# │         │ seeing a refinement.         │                                   │
# └─────────┴──────────────────────────────┴───────────────────────────────────┘
#
# Recovery LLM calls only happen when a genuine signal has been detected.
# ===================================================================

class TrustRecoverySystem:

    # Phrases the AI is explicitly instructed to use when confused.
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

    # -------------------------------------------------------------------
    # DETECTION — keyword matching only, zero LLM calls
    # -------------------------------------------------------------------

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

    # -------------------------------------------------------------------
    # ERROR 1 RECOVERY
    # Triggered when ai_signals_confusion() returns True.
    #
    # The AI already named its confusion and embedded a clarifying question
    # in its natural response (Steps 1-2 from the framework). This function
    # handles only Step 3: a corrected-model summary after the user replies,
    # so the user knows their correction was heard before the flow continues.
    # One LLM call, only on trigger.
    # -------------------------------------------------------------------

    def recover_error1(self, user_clarification, messages, context_data):
        """
        Generate and print a brief corrected-model summary.
        Called after the user has replied to the AI's embedded clarifying question.
        """
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
                    f"Current context:\n{json.dumps(context_data, indent=2)}"
                )
            }
        ]

        corrected_summary = call_llm(summary_msg, max_tokens=120)
        if corrected_summary:
            print(f"\nAI: {corrected_summary}\n")

        self.recovery_log.append({
            "type": "error_1_confusion",
            "user_clarification": user_clarification
        })

    # -------------------------------------------------------------------
    # ERROR 2 RECOVERY
    # Triggered when user_signals_dissatisfaction() returns True.
    #
    # Step 1 — Run the assumption audit and make inferences visible.
    # Step 2 — Walk through each inference; user confirms or corrects.
    # Step 3 — Partially regenerate only affected sections; show what changed.
    # -------------------------------------------------------------------

    def recover_error2(self, profile_text, user_portrait, proposition_data):
        """
        Surface inferred assumptions and apply targeted corrections.
        Only called when the user has expressed dissatisfaction with the profile.
        Returns the corrected profile text, or the original if no changes.
        """
        print("\n" + "-" * 60)
        print("  TRUST RECOVERY — Surfacing What Was Assumed")
        print("-" * 60)
        print(
            "\nAI: Let me find what I inferred versus what you actually told me, "
            "so we can fix exactly what feels off.\n"
        )

        inferences = self._run_assumption_audit(profile_text, user_portrait, proposition_data)

        if not inferences:
            print(
                "AI: I reviewed the profile carefully — everything maps back to what you told me. "
                "Can you point to the specific part that feels wrong so I can address it directly?\n"
            )
            print("-" * 60 + "\n")
            return profile_text

        corrections = {}
        print(f"  I found {len(inferences)} assumption(s) to check with you:\n")

        for i, item in enumerate(inferences, 1):
            trait = item.get("trait", "").strip()
            reason = item.get("reason", "").strip()
            if not trait:
                continue

            print(f"  [{i}] I assumed: \"{trait}\"")
            if reason:
                print(f"       My reasoning: {reason}")
            user_reaction = input(
                "       Press Enter to keep it, or type a correction: "
            ).strip()
            print()

            if user_reaction and user_reaction.lower() not in {
                "yes", "correct", "fine", "keep", "ok", "okay", "sure", "looks good", ""
            }:
                corrections[trait] = user_reaction

        if not corrections:
            print("AI: All assumptions confirmed — the profile stands as written.\n")
            print("-" * 60 + "\n")
            self.recovery_log.append({
                "type": "error_2_assumption_audit",
                "inferences_found": len(inferences),
                "corrections_made": 0
            })
            return profile_text

        print(
            f"\nAI: Got it — {len(corrections)} correction(s) noted. "
            "Updating only the affected sections now.\n"
        )

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
            print(f"\n{'-' * 60}")
            print("  UPDATED PROFILE (targeted corrections only)")
            print(f"{'-' * 60}\n")
            print(updated_profile)
            print(f"\n{'-' * 60}\n")
            self.recovery_log.append({
                "type": "error_2_assumption_audit",
                "inferences_found": len(inferences),
                "corrections_made": len(corrections),
                "traits_corrected": list(corrections.keys())
            })
            return updated_profile

        print("-" * 60 + "\n")
        return profile_text

    def _run_assumption_audit(self, profile_text, user_portrait, proposition_data):
        """
        LLM call to identify traits in the profile that were inferred,
        not explicitly stated by the user. Internal helper.
        Returns a list of dicts: [{"trait": "...", "reason": "..."}, ...]
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

        response = call_llm(audit_msg, temperature=0.1, max_tokens=500)
        try:
            start = response.find("[")
            end = response.rfind("]") + 1
            return json.loads(response[start:end])
        except Exception:
            return []

    # -------------------------------------------------------------------
    # ERROR 3 RECOVERY
    # Triggered when user_signals_overscope() returns True.
    #
    # Step 1 — Acknowledge the over-scope; restore frozen_profile as base.
    # Step 2 — Apply only the original targeted edit the user requested.
    # Step 3 — Show the change in context; confirm scope with user.
    # -------------------------------------------------------------------

    def recover_error3(self, original_feedback, frozen_profile, proposition_data):
        """
        Revert to frozen_profile and apply only the precise targeted edit.
        Only called when the user has signaled the AI changed too much.
        Returns the corrected profile text.
        """
        print("\n" + "-" * 60)
        print("  TRUST RECOVERY — Reverting to Targeted Edit")
        print("-" * 60)
        print(
            "\nAI: You are right — I changed more than you asked. "
            "Going back to the version you already reviewed and applying "
            "only the specific change you requested.\n"
        )

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
            print(f"\n{'-' * 60}")
            print("  REVISED PROFILE (targeted edit only)")
            print(f"{'-' * 60}\n")
            print(targeted_response)
            print(f"\n{'-' * 60}\n")

            user_confirmation = input("You: ").strip()
            print()

            self.recovery_log.append({
                "type": "error_3_overscope",
                "edit_requested": original_feedback,
                "user_confirmation": user_confirmation
            })

            return targeted_response

        print("-" * 60 + "\n")
        return frozen_profile


# Global trust recovery instance shared across all stages
trust_recovery = TrustRecoverySystem()


# ===============================
# STAGE 1: ABOUT YOU
# ===============================
# Learn about the user as a person. Questions quietly map Big Five / MBTI
# trait dimensions without feeling like a personality quiz.
# NEVER ask about partner preferences — that's Stage 2's job.
#
# Trust Recovery (Error 1):
#   The AI's system prompt instructs it to flag confusion using specific
#   marker phrases. After each AI response, ai_signals_confusion() scans
#   for those markers using keyword matching — no extra LLM call. If a
#   signal is found, the AI already embedded the clarifying question in
#   its response. After the user replies, recover_error1() adds the
#   corrected-model summary (one LLM call, only on trigger).
# ===============================

STAGE1_SYSTEM = """You are a warm, perceptive conversational guide helping someone understand themselves \
better so you can eventually help them find the right person for a specific kind of relationship.

RIGHT NOW your ONLY job is to learn about the USER as a person. You are NOT asking what they \
want in someone else — that comes later. You are building a portrait of who they are.

The user is looking for: {relationship_type}

Your questions should naturally surface where the user falls on these personality dimensions \
(but NEVER name these dimensions or make it feel like a quiz):
- Introversion vs. Extraversion — social energy, how they recharge, what connection looks like
- Sensing vs. Intuition — concrete vs. abstract thinker, what they find interesting
- Thinking vs. Feeling — decision-making style, how they handle conflict, how they express care
- Judging vs. Perceiving — structured vs. spontaneous, planning style, flexibility
- Openness to experience — curiosity, novelty-seeking, comfort with change

QUESTION BUDGET:
- For deep relational types (romantic partner, close friend, mentor): ask 5-6 questions
- For moderate relational types (roommate, creative collaborator): ask 4-5 questions
- For activity/context types (sports partner, study buddy): ask 3-4 questions
- Use your judgment. Lighter relationship types need fewer questions.

STRICT RULES:
- Ask ONLY ONE question per turn.
- NEVER ask what the user wants in another person, a partner, or a match. Not even indirectly.
- NEVER include examples, suggestions, or anchor words in your questions. \
Let the user answer entirely in their own words.
- Questions should feel like a thoughtful friend getting to know someone — not a therapist, \
not an interviewer, not a quiz.
- Each question should cover different ground. Do not revisit topics already addressed.
- Keep questions short. One or two sentences max.
- If something the user says seems to contradict what they said earlier, or genuinely \
surprises you given patterns you have noticed, surface that confusion honestly. \
Start your message with exactly 'I want to check something —' or \
'I notice a shift here —' so the user knows you caught it. \
Then ask one clarifying question. Do NOT resolve the tension yourself.
- When you have asked enough questions (per the budget above), write a warm summary of who \
this person is. Start the summary with exactly 'SUMMARY:' and write only in paragraph form. \
The summary should capture personality, values, lifestyle, and relational tendencies — \
NO partner preferences, NO lists, NO JSON."""


def stage1_about_you(relationship_type):
    system_msg = {
        "role": "system",
        "content": STAGE1_SYSTEM.format(relationship_type=relationship_type)
    }

    messages = [
        system_msg,
        {"role": "user", "content": f"Hi — I'm looking for: {relationship_type}"}
    ]

    print("\nLet's start by getting to know you.\n")

    confusion_pending = False

    while True:
        ai_response = call_llm(messages, max_tokens=300)
        if not ai_response:
            break
        print("AI:", ai_response, "\n")

        if "SUMMARY:" in ai_response:
            break

        # Safety valve: if LLM stops asking questions without a summary
        concluding_phrases = ["thank you for sharing", "based on what you've shared", "i have a good sense"]
        has_conclusion = any(p in ai_response.lower() for p in concluding_phrases)
        if "?" not in ai_response and has_conclusion:
            break

        # Trust Recovery: Error 1 detection (zero cost)
        confusion_pending = trust_recovery.ai_signals_confusion(ai_response)

        messages.append({"role": "assistant", "content": ai_response})
        user_input = input("You: ")
        messages.append({"role": "user", "content": user_input})
        print()

        # Trust Recovery: Error 1 recovery (one LLM call, on trigger)
        if confusion_pending:
            trust_recovery.recover_error1(user_input, messages, {})
            confusion_pending = False

    # Extract structured user portrait (internal — never shown to user)
    user_portrait = extract_user_portrait(messages)
    return user_portrait, messages


def extract_user_portrait(conversation_messages):
    """Extract a structured portrait of the USER (not their preferences). Internal only."""
    extraction_msg = {
        "role": "system",
        "content": (
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
    }

    conversation_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in conversation_messages
        if msg['role'] in ['user', 'assistant']
    ])

    messages = [
        extraction_msg,
        {"role": "user", "content": f"Extract the user portrait from this conversation:\n{conversation_text}"}
    ]

    response = call_llm(messages, temperature=0.1, max_tokens=600)

    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        return json.loads(response[json_start:json_end])
    except Exception as e:
        print(f"Warning: Could not parse user portrait: {e}")
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


# ===============================
# STAGE 2: THE PROPOSITION
# ===============================
# Show the user their trait map, then infer what they'd benefit from
# in their stated relationship type. Present as organized categories.
# User reacts, adjusts, confirms. Max 3 rounds.
#
# Trust Recovery (Error 1):
#   Same pattern as Stage 1 — confusion markers in AI response trigger
#   keyword scan, corrected-model summary after user replies.
# ===============================

STAGE2_SYSTEM = """You are a warm, insightful relationship coach. You have just learned a lot about \
the user as a person. Now your job is to do TWO things in your FIRST message:

1. TRAIT MAP — Show the user a brief, readable summary of the personality traits you picked up on. \
Frame it warmly: "Based on what you've shared, here's what I see in you..." \
Map them along these dimensions (use plain language, not jargon):
   - Social energy (introvert <-> extravert)
   - Thinking style (concrete/practical <-> abstract/big-picture)
   - Decision-making (head-first <-> heart-first)
   - Structure (planner <-> spontaneous)
   - Openness (comfort-seeking <-> novelty-seeking)
Keep it to a short paragraph — not a list, not a quiz result. Make it feel like a friend \
reflecting back what they've noticed.

2. INFERRED PRIORITIES — Based on who this person IS (their trait map), infer what they would \
benefit from in their {relationship_type}. Do NOT just repeat what they said — make genuine \
inferences based on personality-compatibility principles.

Use multiple personality frameworks such as MBTI personality theory, Enneagram, PERSOC dynamics, \
and DiSC to analyze the user's traits and infer what they would benefit from. Choose specific \
frameworks based on the relationship type — for example, love languages for romantic relationships, \
DiSC for workplace relationships. Briefly name which framework(s) informed each ranking.

Select 4-6 relevant dimension categories from this menu, based on the relationship type:

{dimension_menu}

For each selected category, list 2-4 ranked items with a ONE-LINE explanation of why you \
placed it there, tied to what you know about the user. Format as clear, organized categories \
with numbered items.

End by asking the user to react: reorder, add, remove, or confirm.

STRICT RULES:
- Present the trait map FIRST, then the inferred priorities.
- The categories you choose must make sense for the relationship type. Do NOT use love languages \
for a squash partner. Do NOT use competitiveness for a romantic partner (unless it came up).
- Frame everything as inference, not prescription: "I think..." / "Based on who you are..." \
not "You need..." / "You should look for..."
- NEVER give examples when asking for feedback. Let the user tell you what to change.
- If a user's correction seems to contradict something they said earlier, start your response \
with exactly 'I want to check something —' before asking for clarification.
- Maximum 3 rounds of back-and-forth. After the user confirms (or after 3 rounds), \
output exactly 'PROPOSITION CONFIRMED' and stop."""


def stage2_proposition(user_portrait, relationship_type):
    system_msg = {
        "role": "system",
        "content": STAGE2_SYSTEM.format(
            relationship_type=relationship_type,
            dimension_menu=DIMENSION_MENU
        )
    }

    user_context = (
        f"The user is looking for: {relationship_type}\n\n"
        f"Here is the structured portrait of who they are:\n"
        f"{json.dumps(user_portrait, indent=2)}"
    )

    messages = [system_msg, {"role": "user", "content": user_context}]

    print("\nBased on what I've learned about you, let me map out what I think matters most...\n")

    round_count = 0
    confusion_pending = False

    while round_count < 3:
        ai_response = call_llm(messages, max_tokens=800)
        if not ai_response:
            break
        print("AI:", ai_response, "\n")

        if "PROPOSITION CONFIRMED" in ai_response:
            break

        # Trust Recovery: Error 1 detection (zero cost)
        confusion_pending = trust_recovery.ai_signals_confusion(ai_response)

        messages.append({"role": "assistant", "content": ai_response})
        user_input = input("You: ").strip()
        messages.append({"role": "user", "content": user_input})
        print()

        # Trust Recovery: Error 1 recovery (one LLM call, on trigger)
        if confusion_pending:
            trust_recovery.recover_error1(user_input, messages, user_portrait)
            confusion_pending = False

        round_count += 1

    if round_count == 3:
        print("AI: Great — I've noted your adjustments. Let's build your profile.\n")

    # Extract the final confirmed proposition as structured data (internal)
    proposition_data = extract_proposition(messages, relationship_type)
    return proposition_data


def extract_proposition(conversation_messages, relationship_type):
    """Extract the final confirmed proposition as structured JSON. Internal only."""
    conversation_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in conversation_messages
        if m['role'] in ['user', 'assistant']
    ])

    extraction_msg = [
        {
            "role": "system",
            "content": (
                "Based on the proposition conversation, extract the final agreed-upon priorities "
                "into JSON format. The categories should reflect what was actually discussed — "
                "they will vary based on the relationship type. "
                "Output ONLY valid JSON.\n"
                "{\n"
                f'  "relationship_type": "{relationship_type}",\n'
                '  "user_trait_summary": "",\n'
                '  "selected_dimensions": [\n'
                '    {\n'
                '      "category": "",\n'
                '      "ranked_items": [\n'
                '        {"item": "", "reasoning": ""}\n'
                '      ]\n'
                '    }\n'
                '  ],\n'
                '  "deal_breakers": []\n'
                "}"
            )
        },
        {"role": "user", "content": conversation_text}
    ]

    response = call_llm(extraction_msg, temperature=0.1, max_tokens=600)

    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        return json.loads(response[json_start:json_end])
    except Exception as e:
        print(f"Warning: Could not parse proposition: {e}")
        return {
            "relationship_type": relationship_type,
            "user_trait_summary": "",
            "selected_dimensions": [],
            "deal_breakers": []
        }


# ===============================
# STAGE 3: TENSION DETECTION
# ===============================
# Analyze the user's inferred priorities for internal contradictions.
# Ask up to 3 clarifying questions. Do not resolve tensions — let the
# user think through them.
# ===============================

def stage3_tension(proposition_data):
    MAX_TENSION_QUESTIONS = 3

    system_msg = {
        "role": "system",
        "content": (
            "You are a warm relationship coach having a conversation with the user. "
            "Always address the user directly as 'you' — speak to them, not about them. "
            "Analyze their inferred relationship priorities and detect any internal "
            "contradictions or tensions. "
            "Ask ONE clarifying question at a time in a conversational, friendly tone. "
            "Do not resolve tensions yourself — let the user think through them. "
            "Stop when the priorities are clear enough to generate a profile, or after 3 turns."
        )
    }

    messages = [
        system_msg,
        {"role": "user", "content": f"Here are the user's inferred priorities: {json.dumps(proposition_data)}"}
    ]

    print("\nLet me think through a couple of things with you...\n")
    question_count = 0

    while True:
        ai_response = call_llm(messages, max_tokens=300)
        if not ai_response:
            break
        print("AI:", ai_response, "\n")

        if "resolved" in ai_response.lower():
            break

        messages.append({"role": "assistant", "content": ai_response})
        user_input = input("You: ")
        messages.append({"role": "user", "content": user_input})
        print()

        question_count += 1
        if question_count >= MAX_TENSION_QUESTIONS:
            # Brief confirmation so the user knows their last answer was heard
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
            wrap_up = call_llm(confirmation_msg, max_tokens=120)
            if wrap_up:
                print(f"AI: {wrap_up}\n")
            print("Thanks for working through that with me!\n")
            break

    return proposition_data


# ===============================
# STAGE 4: PROFILE GENERATION
# ===============================
# Build a full profile. Sections are dynamic based on relationship type.
# The LLM selects which sections make sense.
# ===============================

STAGE4_SYSTEM = """You are collaboratively building a profile of the user's ideal {relationship_type} \
based on everything you know about them.

The user's trait summary: {trait_summary}

Their confirmed priorities:
{proposition_json}

SECTION SELECTION — Choose the sections that make sense for this relationship type. \
Here are your options (pick 5-8):

FOR ANY RELATIONSHIP TYPE:
- Name & Age
- Personality & Core Traits
- Communication Style
- A Typical Interaction (what spending time together looks like)
- Why This Person Fits You

FOR ROMANTIC / CLOSE EMOTIONAL RELATIONSHIPS:
- Physical Description
- Emotional Style & Love Languages
- Conflict Style
- Backstory
- A Typical Day in Their Life

FOR ACTIVITY / CONTEXT-BASED RELATIONSHIPS:
- Play Style or Work Style
- Skill Level & Approach
- Scheduling & Reliability
- Growth Orientation

STRICT RULES:
- Write in warm, engaging prose. Use a clear header for each section.
- Never show JSON to the user.
- The top-ranked priorities from the proposition should come through clearly in the profile.
- The profile must NOT include any of the user's stated deal breakers.
- Keep the profile grounded and specific — this should feel like a real person, not a wish list.
- End with a short 'Why This Person Fits You' section that ties the profile back to \
the user's personality and needs."""


def stage4_profile(proposition_data):
    relationship_type = proposition_data.get("relationship_type", "connection")
    trait_summary = proposition_data.get("user_trait_summary", "")

    system_content = STAGE4_SYSTEM.format(
        relationship_type=relationship_type,
        trait_summary=trait_summary,
        proposition_json=json.dumps(proposition_data.get("selected_dimensions", []), indent=2)
    )

    messages = [{"role": "system", "content": system_content}]

    print("AI: Before I build the profile — do you have anything specific in mind?")
    print("A name, a vibe, a detail you want included? Or should I surprise you?\n")

    user_ideas = input("You: ").strip()
    if user_ideas and user_ideas.lower() not in {"surprise me", "surprise", "no", "nope", ""}:
        messages.append({
            "role": "user",
            "content": f"The user wants to include these ideas: {user_ideas}"
        })

    messages.append({
        "role": "user",
        "content": (
            f"Generate a complete profile based on these confirmed priorities: "
            f"{json.dumps(proposition_data, indent=2)}.\n"
            f"Select appropriate sections for this relationship type. "
            f"End with a 'Why This Person Fits You' section. "
            f"Reflect the ranked priorities and exclude all deal breakers."
        )
    })

    print("\nGenerating your profile...\n")
    ai_response = call_llm(messages, max_tokens=1500)
    if not ai_response:
        return ""

    print(f"\n{'-' * 60}")
    print("YOUR IDEAL PROFILE")
    print(f"{'-' * 60}\n")
    print(ai_response)
    print(f"\n{'-' * 60}\n")

    return ai_response


# ===============================
# STAGE 5: REFINEMENT
# ===============================
# Entry check (Error 2): Ask "Does this feel right?" — if user signals
# dissatisfaction, run assumption audit before entering the edit loop.
# Edit loop (Error 3): Track frozen_profile. If user signals over-scope
# after an edit, revert to frozen and apply only the targeted change.
# ===============================

STAGE5_SYSTEM = """You are helping the user refine a {relationship_type} profile through natural conversation.

When the user gives feedback, update the profile and reprint it in full — same warm prose format, \
no JSON. React naturally as a collaborator: acknowledge what changed, and notice what else \
might be worth exploring.

When the user is done, close warmly.

Never include anything the user has flagged as a deal breaker."""


def stage5_refinement(user_portrait, proposition_data, profile_text):
    if not profile_text:
        print("\nAI: No profile to refine yet.\n")
        return

    relationship_type = proposition_data.get("relationship_type", "connection")

    # --- Trust Recovery: Error 2 detection (zero cost) ---
    # Ask if the profile feels right — the user's answer becomes the
    # first refinement input so feedback is never lost.
    print("AI: Does this feel right to you, or is something off?")
    print("(Type 'done' when you're happy with it.)\n")
    initial_reaction = input("You: ").strip()
    print()

    # If user is already happy, exit early
    if initial_reaction.lower() in {"done", "exit", "quit", "finished", "that's it",
                                     "looks good", "yes", "yeah", "perfect", "love it"}:
        print(
            f"\nAI: Wonderful! I hope this gives you a clear picture of the {relationship_type} "
            f"you're looking for. Good luck out there.\n"
        )
        return

    # If dissatisfaction detected, run trust recovery first
    if trust_recovery.user_signals_dissatisfaction(initial_reaction):
        profile_text = trust_recovery.recover_error2(profile_text, user_portrait, proposition_data)

    # frozen_profile: the last version the user has implicitly approved
    frozen_profile = profile_text

    # last_feedback: the request that produced the currently displayed profile.
    last_feedback = None

    # Generate initial suggestions
    suggestion_msg = [
        {
            "role": "system",
            "content": (
                "Based on the profile below, suggest 2-3 specific things the user might want "
                "to personalize or adjust. Be brief and conversational. "
                "Do NOT give generic suggestions — tie them to the specific profile content."
            )
        },
        {"role": "user", "content": f"Profile:\n{profile_text}"}
    ]
    suggestions = call_llm(suggestion_msg, max_tokens=150)

    print("What would you like to change? A few things you might consider:")
    print(suggestions if suggestions else "Any details that would make this person feel more real to you.")
    print("\nType 'done' when you're happy with it.\n")

    messages = [
        {
            "role": "system",
            "content": STAGE5_SYSTEM.format(relationship_type=relationship_type)
        },
        {"role": "user", "content": "Here is the current profile:\n\n" + profile_text},
        {"role": "assistant", "content": "I have the profile ready. What would you like to tweak?"}
    ]

    # Use the initial reaction as the first feedback if it wasn't a
    # dissatisfaction signal (those are already handled above)
    first_feedback = initial_reaction if not trust_recovery.user_signals_dissatisfaction(initial_reaction) else None

    while True:
        if first_feedback:
            feedback = first_feedback
            first_feedback = None
        else:
            feedback = input("You: ").strip()
        if not feedback:
            continue
        if feedback.lower() in {"done", "exit", "quit", "finished", "that's it", "looks good"}:
            print(
                f"\nAI: Great — I hope this gives you a clear picture of the {relationship_type} "
                f"you're looking for. Good luck out there.\n"
            )
            break

        # --- Trust Recovery: Error 3 detection (zero cost) ---
        if last_feedback and trust_recovery.user_signals_overscope(feedback):
            corrected = trust_recovery.recover_error3(
                last_feedback, frozen_profile, proposition_data
            )
            if corrected:
                frozen_profile = corrected
                messages = [
                    {"role": "system", "content": STAGE5_SYSTEM.format(relationship_type=relationship_type)},
                    {"role": "user", "content": "Here is the current profile:\n\n" + frozen_profile},
                    {"role": "assistant", "content": "Profile updated. What else would you like to change?"}
                ]
                last_feedback = None
            continue

        messages.append({
            "role": "user",
            "content": (
                f"{feedback}\n\n"
                "After updating the profile, reprint it in full. "
                "Briefly note what changed and suggest one or two things "
                "that might still be worth refining."
            )
        })

        ai_response = call_llm(messages, max_tokens=3000)
        if not ai_response:
            print("AI: Something went wrong. Let's try again.")
            continue

        messages.append({"role": "assistant", "content": ai_response})
        print(f"\n{'-' * 60}\n")
        print(ai_response)
        print(f"\n{'-' * 60}")
        print("(Type 'done' when you're happy with it.)\n")

        frozen_profile = ai_response
        last_feedback = feedback


# ===============================
# RUN FULL PROTOTYPE
# ===============================
def run_prototype():
    print("\n" + "=" * 60)
    print("  AI RELATIONSHIP PROFILE BUILDER")
    print("=" * 60)
    print(
        "\nThis tool builds a detailed profile of your ideal connection — "
        "whether that's a romantic partner, a close friend, a squash buddy, "
        "or anything else — through a guided conversation.\n"
        "\nFirst I'll get to know you as a person. Then I'll use what I've learned "
        "to propose what I think you'd benefit from in that connection. "
        "You'll review, adjust, and we'll build the profile together.\n"
    )
    print("What kind of relationship or connection are you looking for?\n")
    relationship_type = input("You: ").strip()
    print()

    # Stage 1: About You
    print("\n" + "=" * 40)
    print("  GETTING TO KNOW YOU")
    print("=" * 40)
    user_portrait, stage1_messages = stage1_about_you(relationship_type)

    # Stage 2: The Proposition
    print("\n" + "=" * 40)
    print("  WHAT I THINK YOU NEED")
    print("=" * 40)
    proposition_data = stage2_proposition(user_portrait, relationship_type)

    # Stage 3: Tension Detection
    print("\n" + "=" * 40)
    print("  A FEW CLARIFICATIONS")
    print("=" * 40)
    proposition_data = stage3_tension(proposition_data)

    # Stage 4: Profile Generation
    print("\n" + "=" * 40)
    print("  BUILDING YOUR PROFILE")
    print("=" * 40)
    profile_text = stage4_profile(proposition_data)

    # Stage 5: Refinement
    print("\n" + "=" * 40)
    print("  REFINE YOUR PROFILE")
    print("=" * 40)
    stage5_refinement(user_portrait, proposition_data, profile_text)

    # Trust recovery session summary
    if trust_recovery.recovery_log:
        print("\n" + "-" * 60)
        print("  SESSION TRUST RECOVERY SUMMARY")
        print("-" * 60)
        for i, event in enumerate(trust_recovery.recovery_log, 1):
            etype = event.get("type", "unknown")
            if etype == "error_1_confusion":
                print(f"  [{i}] AI confusion resolved after user clarified a shift")
            elif etype == "error_2_assumption_audit":
                n = event.get("corrections_made", 0)
                found = event.get("inferences_found", 0)
                print(f"  [{i}] Assumption audit (user-triggered): {found} found, {n} corrected")
            elif etype == "error_3_overscope":
                print(f"  [{i}] Over-scope corrected for: \"{event.get('edit_requested', '')}\"")
        print("-" * 60 + "\n")


# ===============================
# START
# ===============================
if __name__ == "__main__":
    run_prototype()
