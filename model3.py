# -------------------------------
# AI RELATIONSHIP PROFILE BUILDER
# -------------------------------

import json

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "llama-3.1-8b.gguf"
TEST_MODE = False  # Set to True to run flow test without the real model

# -------------------------------
# TEST MODE — mock LLM responses and user inputs
# These simulate one full run through the flow to verify every stage.
# -------------------------------
_test_llm_idx = 0
_test_input_idx = 0

# 16 mock LLM responses — ordered to match the exact call sequence:
#   Stage 1: 6 questions (indices 0-5) + 1 SUMMARY (6)
#     (relationship type is asked upfront in run_prototype, not by the AI)
#   extract_preferences_json: 1 JSON response (7)
#   stage_ranking round 1: rankings + review question (8)
#   stage_ranking round 2: RANKINGS CONFIRMED (9)
#   stage_ranking extract: JSON (10)
#   stage2_tension: 3 questions (11-13)
#   stage3_profile: profile text (14)
#   stage4_refinement suggestions: suggestion text (15)
_TEST_LLM_RESPONSES = [
    # Stage 1 — learning about the user (relationship type already known, so not asked again)
    "Tell me a bit about yourself — who are you as a person?",
    "What kinds of things bring you the most joy or meaning in your day-to-day life?",
    "How do you tend to show up in close relationships — what role do you usually play?",
    "Do you have any preferences when it comes to age or gender?",
    "What qualities do you most hope this person brings to the relationship?",
    "Have there been any red flags or deal breakers in past relationships that you'd want to avoid this time?",
    # SUMMARY triggers end of Stage 1 loop
    "SUMMARY: You are a thoughtful, introverted software engineer in your early 30s who values genuine connection and intellectual depth. You are looking for a romantic partner — a woman in her late 20s to mid-30s — who is emotionally intelligent, kind, and curious about the world. Past experiences have shown you that emotional unavailability is a firm deal breaker for you.",
    # extract_preferences_json — must be valid JSON
    '{"user_profile": {"personality": "introverted, analytical, thoughtful", "lifestyle": "enjoys hiking, reading, long conversations", "values": ["authenticity", "growth", "connection"], "relationship_style": "supportive, values depth"}, "relationship_type": "romantic", "gender_preference": "female", "age_range": "27-35", "appearance_preferences": [], "lifestyle_habits": [], "core_values": ["kindness", "emotional intelligence", "intellectual curiosity"], "emotional_needs": ["emotional availability", "deep conversation", "stability"], "deal_breakers": ["emotional unavailability", "dismissiveness"], "attachment_style": "", "love_languages": ["quality time", "words of affirmation"]}',
    # stage_ranking — round 1: present ranked lists and ask for review
    "Based on who you are and what you have shared, here is what I believe you value most in a partner:\n\n**Core Values (ranked):**\n1. Emotional intelligence\n2. Kindness\n3. Intellectual curiosity\n\n**Emotional Needs (ranked):**\n1. Emotional availability\n2. Deep, meaningful conversation\n3. Stability and consistency\n\n**Personality Traits:**\n1. Empathetic\n2. Curious\n3. Grounded\n\n**Love Languages:**\n1. Quality time\n2. Words of affirmation\n\nDoes this feel right to you, or would you like to shift, add, or remove anything?",
    # stage_ranking — round 2: user confirms, RANKINGS CONFIRMED triggers exit
    "RANKINGS CONFIRMED",
    # stage_ranking extract — valid JSON
    '{"ranked_core_values": ["emotional intelligence", "kindness", "intellectual curiosity"], "ranked_emotional_needs": ["emotional availability", "deep conversation", "stability"], "ranked_personality_traits": ["empathetic", "curious", "grounded"], "ranked_love_languages": ["quality time", "words of affirmation"], "ranked_lifestyle_habits": ["values quiet evenings", "intellectually active", "enjoys nature"], "deal_breakers": ["emotional unavailability", "dismissiveness"]}',
    # stage2_tension — 3 clarifying questions (3rd triggers MAX break, no user input needed)
    "I noticed you value both deep intellectual connection and emotional warmth — sometimes those pull in different directions. How important is it that this person matches your intellectual pace versus simply being emotionally present?",
    "That makes sense. You also mentioned stability alongside curiosity and growth. How do you balance wanting someone grounded with wanting someone who keeps evolving?",
    "That gives me a much clearer picture — I think you have a strong sense of what you need.",
    # stage3_profile — full profile text
    "## Meet Nora\n\n**Name & Age:** Nora, 31\n\n**Physical Description:** Nora carries herself with quiet warmth — unhurried and comfortable in her own skin.\n\n**Personality & Core Values:** At her core, Nora is deeply empathetic and emotionally attuned. She values authenticity above all else.\n\n**Emotional Style & Love Languages:** Nora's primary love language is quality time. She is fully present when she is with someone she cares about.\n\n**A Typical Day:** Nora starts her mornings slowly with coffee and whatever book she is halfway through.\n\n**Conflict Style:** Nora avoids drama. When tensions arise she prefers to take a breath, then talk things through calmly and honestly.\n\n**Backstory:** Nora grew up in a mid-sized city, the middle child in a close-knit family.\n\n**Why This Fits You:** Nora embodies the emotional availability and intellectual depth you value most. She will not shut down when things get hard — she shows up.",
    # stage4_refinement — initial suggestions
    "A few things you might want to personalize: Nora's specific career, her relationship with her family, or a small quirk that makes her feel real.",
]

# 13 mock user inputs — ordered to match the exact input() call sequence
_TEST_USER_INPUTS = [
    # Upfront relationship type question (run_prototype)
    "A romantic relationship — I am looking for a serious long-term partner.",
    # Stage 1 — 6 answers (one per question before SUMMARY; relationship type already known)
    "I am a 30-year-old software engineer. Pretty introverted and analytical.",
    "Reading, hiking, and long conversations about ideas that go nowhere useful.",
    "I am usually the supportive one. I care deeply about the people close to me.",
    "Women — probably late 20s to mid 30s.",
    "Emotional intelligence, kindness, someone genuinely curious about the world.",
    "People who shut down emotionally or become dismissive when things get hard.",
    # stage_ranking — 1 answer after seeing ranked lists
    "That looks right. I would put emotional availability as number one under emotional needs.",
    # stage2_tension — 2 answers (3rd question hits MAX and breaks without input)
    "Emotional presence matters more to me than matching my intellect exactly.",
    "I want someone grounded but still open to growth. They do not need to be at my pace.",
    # stage3_profile — user ideas prompt
    "Surprise me.",
    # stage3_profile — "does this feel right?" check (happy path — no recovery triggered)
    "Yes, this feels right.",
    # stage4_refinement — user says done
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
# HELPER FUNCTION TO CALL LLM
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
    # in its natural response (Steps 1–2 from the framework). This function
    # handles only Step 3: a corrected-model summary after the user replies,
    # so the user knows their correction was heard before the flow continues.
    # One LLM call, only on trigger.
    # -------------------------------------------------------------------

    def recover_error1(self, user_clarification, messages, preference_json):
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
                    f"Current preference summary:\n{json.dumps(preference_json, indent=2)}"
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
    #           One false assumption is treated as a signal to audit all —
    #           not just spot-fix the one the user noticed.
    # Step 3 — Partially regenerate only affected sections; show what changed.
    # -------------------------------------------------------------------

    def recover_error2(self, profile_text, preference_json):
        """
        Surface inferred assumptions and apply targeted corrections.
        Only called when the user has expressed dissatisfaction with the profile.
        Returns the corrected profile text, or the original if no changes.
        """
        print("\n" + "─" * 60)
        print("  TRUST RECOVERY — Surfacing What Was Assumed")
        print("─" * 60)
        print(
            "\nAI: Let me find what I inferred versus what you actually told me, "
            "so we can fix exactly what feels off.\n"
        )

        inferences = self._run_assumption_audit(profile_text, preference_json)

        if not inferences:
            print(
                "AI: I reviewed the profile carefully — everything maps back to what you told me. "
                "Can you point to the specific part that feels wrong so I can address it directly?\n"
            )
            print("─" * 60 + "\n")
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
            print("─" * 60 + "\n")
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
            print(f"\n{'─' * 60}")
            print("  UPDATED PROFILE (targeted corrections only)")
            print(f"{'─' * 60}\n")
            print(updated_profile)
            print(f"\n{'─' * 60}\n")
            self.recovery_log.append({
                "type": "error_2_assumption_audit",
                "inferences_found": len(inferences),
                "corrections_made": len(corrections),
                "traits_corrected": list(corrections.keys())
            })
            return updated_profile

        print("─" * 60 + "\n")
        return profile_text

    def _run_assumption_audit(self, profile_text, preference_json):
        """
        LLM call to identify traits in the profile that were inferred,
        not explicitly stated by the user. Internal helper.
        Returns a list of dicts: [{"trait": "...", "reason": "..."}, ...]
        """
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

    # -------------------------------------------------------------------
    # ERROR 3 RECOVERY
    # Triggered when user_signals_overscope() returns True.
    #
    # Step 1 — Acknowledge the over-scope; restore frozen_profile as base.
    # Step 2 — Apply only the original targeted edit the user requested.
    # Step 3 — Show the change in context; confirm scope with user.
    # -------------------------------------------------------------------

    def recover_error3(self, original_feedback, frozen_profile, preference_json):
        """
        Revert to frozen_profile and apply only the precise targeted edit.
        Only called when the user has signaled the AI changed too much.
        Returns the corrected profile text.
        """
        print("\n" + "─" * 60)
        print("  TRUST RECOVERY — Reverting to Targeted Edit")
        print("─" * 60)
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
            print(f"\n{'─' * 60}")
            print("  REVISED PROFILE (targeted edit only)")
            print(f"{'─' * 60}\n")
            print(targeted_response)
            print(f"\n{'─' * 60}\n")

            user_confirmation = input("You: ").strip()
            print()

            self.recovery_log.append({
                "type": "error_3_overscope",
                "edit_requested": original_feedback,
                "user_confirmation": user_confirmation
            })

            return targeted_response

        print("─" * 60 + "\n")
        return frozen_profile


# Global trust recovery instance shared across all stages
trust_recovery = TrustRecoverySystem()


# -------------------------------
# STAGE 1: INTAKE
# First learn who the user is as a person, then what they are looking for.
#
# Trust Recovery (Error 1):
#   The AI's system prompt instructs it to flag confusion using specific
#   marker phrases ("I want to check something —" / "I notice a shift here —").
#   After each AI response, ai_signals_confusion() scans for those markers
#   using keyword matching — no extra LLM call. If a signal is found, the AI
#   already embedded the clarifying question in its response. After the user
#   replies, recover_error1() adds the corrected-model summary (one LLM call,
#   only on trigger).
# -------------------------------
def stage1_intake(relationship_type=""):
    relationship_context = (
        f"The user has already told you they are looking for: {relationship_type}. "
        "Do NOT ask about relationship type again — that has already been established. "
        "You may reference it naturally when relevant.\n\n"
    ) if relationship_type else ""

    system_msg = {
        "role": "system",
        "content": (
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
    }

    initial_user_content = (
        f"Hi. I am looking to explore: {relationship_type}." if relationship_type else "Hi."
    )
    messages = [system_msg, {"role": "user", "content": initial_user_content}]

    # Tracks whether the last AI turn contained a confusion signal,
    # so we know to run the corrected-model summary after the user replies.
    confusion_pending = False

    # Lightweight running preferences used only to give recover_error1
    # enough context for the corrected summary. Full extraction happens
    # after the loop — no extra LLM calls here.
    running_preferences = {"relationship_type": relationship_type}

    print("\nGreat — now let's get to know you a bit better before we build your profile.\n")
    while True:
        ai_response = call_llm(messages, max_tokens=300)
        if not ai_response:
            break
        print("AI:", ai_response, "\n")

        if "SUMMARY:" in ai_response:
            break

        concluding_phrases = ["thank you for sharing", "based on what you've shared", "in summary"]
        has_conclusion = any(p in ai_response.lower() for p in concluding_phrases)
        if "?" not in ai_response and has_conclusion:
            break

        # ── Trust Recovery: Error 1 detection (zero cost) ────────────
        # Scan the AI's existing response for confusion markers.
        # Pure keyword match — no LLM call, no context inflation.
        confusion_pending = trust_recovery.ai_signals_confusion(ai_response)
        # ─────────────────────────────────────────────────────────────

        messages.append({"role": "assistant", "content": ai_response})
        user_input = input("You: ")
        print()
        messages.append({"role": "user", "content": user_input})

        # ── Trust Recovery: Error 1 recovery (one LLM call, on trigger) ─
        # The clarifying question was already in the AI's response.
        # We only add the corrected-model summary now that the user replied.
        if confusion_pending:
            trust_recovery.recover_error1(user_input, messages, running_preferences)
            confusion_pending = False
        # ─────────────────────────────────────────────────────────────

    preference_json = extract_preferences_json(messages)
    return preference_json


def extract_preferences_json(conversation_messages):
    """Extract structured preferences from conversation — never shown to user."""
    extraction_msg = {
        "role": "system",
        "content": (
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
    }

    conversation_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in conversation_messages
        if msg['role'] in ['user', 'assistant']
    ])

    messages = [
        extraction_msg,
        {"role": "user", "content": f"Extract preferences from this conversation:\n{conversation_text}"}
    ]

    response = call_llm(messages, temperature=0.1, max_tokens=600)

    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        return json.loads(response[json_start:json_end])
    except Exception as e:
        print(f"Warning: Could not parse preferences: {e}")
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

# -------------------------------
# STAGE 2: FRAMEWORK-BASED RANKING
# Use MBTI, attachment style, and love language theory to infer ranked
# priorities from who the user is. User proofreads and corrects.
#
# Trust Recovery (Error 1):
#   Same pattern as Stage 1 — the ranking system prompt instructs the AI
#   to use confusion markers if a correction contradicts earlier statements.
#   Detection and recovery are identical: keyword scan after each AI response,
#   corrected-model summary after the user replies, only on trigger.
# -------------------------------
def stage_ranking(preference_json):
    system_msg = {
        "role": "system",
        "content": (
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
    }

    user_context = (
        f"Here is what I know about the user and what they are looking for:\n"
        f"{json.dumps(preference_json, indent=2)}"
    )

    messages = [system_msg, {"role": "user", "content": user_context}]

    print("\nLet me use what you've shared to map out what matters most to you...\n")
    round_count = 0
    confusion_pending = False

    while round_count < 3:
        ai_response = call_llm(messages, max_tokens=600)
        if not ai_response:
            break
        print("AI:", ai_response, "\n")

        if "RANKINGS CONFIRMED" in ai_response:
            break

        # ── Trust Recovery: Error 1 detection (zero cost) ────────────
        confusion_pending = trust_recovery.ai_signals_confusion(ai_response)
        # ─────────────────────────────────────────────────────────────

        messages.append({"role": "assistant", "content": ai_response})
        user_input = input("You: ").strip()
        print()
        messages.append({"role": "user", "content": user_input})

        # ── Trust Recovery: Error 1 recovery (one LLM call, on trigger) ─
        if confusion_pending:
            trust_recovery.recover_error1(user_input, messages, preference_json)
            confusion_pending = False
        # ─────────────────────────────────────────────────────────────

        round_count += 1

    if round_count == 3:
        print("AI: Great — I have noted your adjustments. Let's move on.\n")

    # Extract the final agreed rankings and merge into preference_json
    conversation_text = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in messages
        if m['role'] in ['user', 'assistant']
    ])

    ranking_extract_msg = [
        {
            "role": "system",
            "content": (
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
        },
        {"role": "user", "content": conversation_text}
    ]

    ranking_response = call_llm(ranking_extract_msg, temperature=0.1, max_tokens=400)
    try:
        json_start = ranking_response.find("{")
        json_end = ranking_response.rfind("}") + 1
        updated_rankings = json.loads(ranking_response[json_start:json_end])
        preference_json.update(updated_rankings)
    except Exception as e:
        print(f"Warning: Could not parse rankings: {e}")

    return preference_json

# -------------------------------
# STAGE 3: TENSION DETECTION + CLARIFICATION
# -------------------------------
def stage2_tension(preference_json):
    MAX_TENSION_QUESTIONS = 3

    system_msg = {
        "role": "system",
        "content": (
            "You are a warm relationship coach having a conversation with the user. "
            "Always address the user directly as 'you' — speak to them, not about them. "
            "Analyze their relationship preferences and detect any internal contradictions or tensions. "
            "Ask ONE clarifying question at a time in a conversational, friendly tone. "
            "Do not resolve tensions yourself — let the user think through them. "
            "Stop when the preferences are clear enough to generate a profile, or after 3 turns."
        )
    }

    messages = [
        system_msg,
        {"role": "user", "content": f"Here are the preferences: {json.dumps(preference_json)}"}
    ]

    print("\nLet me think through a couple of things with you...\n")
    question_count = 0
    while True:
        ai_response = call_llm(messages, max_tokens=300)
        if not ai_response:
            break
        print("AI:", ai_response, "\n")

        question_count += 1
        if "resolved" in ai_response.lower() or question_count >= MAX_TENSION_QUESTIONS:
            if question_count >= MAX_TENSION_QUESTIONS:
                print("Thanks for working through that with me!\n")
            break

        messages.append({"role": "assistant", "content": ai_response})
        user_input = input("You: ")
        messages.append({"role": "user", "content": user_input})
        print()

    return preference_json

# -------------------------------
# STAGE 4: PROFILE GENERATION
#
# Trust Recovery (Error 2):
#   After displaying the profile, a natural "does this feel right?" question
#   is asked. user_signals_dissatisfaction() scans the user's reply for
#   displeasure markers — zero LLM cost. Only if dissatisfaction is detected
#   does the assumption audit run (two LLM calls: audit + optional regeneration).
# -------------------------------
STAGE3_SYSTEM = (
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

def stage3_profile(preference_json):
    messages = [{"role": "system", "content": STAGE3_SYSTEM}]

    print("AI: Before I build the profile — do you have anything specific in mind?")
    print("A name, a vibe, a detail you definitely want included? Or should I surprise you?\n")

    user_ideas = input("You: ").strip()
    if user_ideas and user_ideas.lower() not in {"surprise me", "surprise", "no", "nope", ""}:
        messages.append({
            "role": "user",
            "content": f"The user wants to include these ideas: {user_ideas}"
        })

    messages.append({
        "role": "user",
        "content": (
            f"Generate a complete profile based on these preferences: {json.dumps(preference_json)}. "
            f"Cover all sections with clear headers. End with a short paragraph explaining "
            f"why this profile is a strong match for this specific user. "
            f"Reflect the ranked priorities and exclude all deal breakers."
        )
    })

    print("\nGenerating your profile...\n")
    ai_response = call_llm(messages, max_tokens=1500)
    if not ai_response:
        return ""

    print(f"\n{'─' * 60}")
    print("YOUR IDEAL PROFILE")
    print(f"{'─' * 60}\n")
    print(ai_response)
    print(f"\n{'─' * 60}\n")

    # ── Trust Recovery: Error 2 detection (zero cost) ────────────────
    # A single natural question — already a normal part of the conversation.
    # No extra prompting, no framing around trust recovery.
    # Keyword scan on the user's reply — no LLM call.
    print("AI: Does this feel right to you, or is something off?\n")
    profile_reaction = input("You: ").strip()
    print()

    if trust_recovery.user_signals_dissatisfaction(profile_reaction):
        ai_response = trust_recovery.recover_error2(ai_response, preference_json)
    # ─────────────────────────────────────────────────────────────────

    return ai_response

# -------------------------------
# STAGE 5: REFINEMENT
#
# Trust Recovery (Error 3):
#   The AI's refinement response is always shown normally first.
#   frozen_profile tracks the last user-approved state.
#   If the user's NEXT message signals over-scoping (keyword scan,
#   zero LLM cost), recover_error3() is triggered: frozen_profile becomes
#   the base, the precise targeted edit is applied, and the user confirms.
# -------------------------------
STAGE4_SYSTEM = (
    "You are helping the user refine a relationship profile through natural conversation. "
    "When the user gives feedback, update the profile and reprint it in full — same warm prose format, no JSON. "
    "React naturally as a collaborator: acknowledge what changed, and notice what else might be worth exploring. "
    "When the user is done, close warmly. "
    "Never include anything the user has flagged as a deal breaker or red flag."
)

def stage4_refinement(preference_json, profile_text):
    if not profile_text:
        print("\nAI: No profile to refine yet.\n")
        return

    suggestion_msg = [
        {
            "role": "system",
            "content": (
                "Based on the profile below, suggest 2-3 specific things the user might want "
                "to personalize or adjust. Be brief and conversational."
            )
        },
        {"role": "user", "content": f"Profile:\n{profile_text}"}
    ]
    suggestions = call_llm(suggestion_msg, max_tokens=150)

    print("What would you like to change? A few things you might consider:")
    print(suggestions if suggestions else "The backstory, the career, or a small quirk that makes the person feel real.")
    print("\nType 'done' when you're happy with it.\n")

    # frozen_profile: the last version the user has implicitly approved
    # by not complaining about it. Advances on every accepted edit.
    frozen_profile = profile_text

    # last_feedback: the request that produced the currently displayed profile.
    # Needed so recover_error3 knows what to re-apply if over-scoping is flagged.
    last_feedback = None

    messages = [
        {"role": "system", "content": STAGE4_SYSTEM},
        {"role": "user", "content": "Here is the current profile:\n\n" + profile_text},
        {"role": "assistant", "content": "I have the profile here. What would you like to tweak?"}
    ]

    while True:
        feedback = input("You: ").strip()
        if not feedback:
            continue
        if feedback.lower() in {"done", "exit", "quit", "finished", "that's it", "looks good"}:
            print(
                "\nAI: Wonderful! I hope this profile gives you a clear sense of what you're looking for. "
                "Good luck — you deserve someone great.\n"
            )
            break

        # ── Trust Recovery: Error 3 detection (zero cost) ────────────
        # If the user's message signals the last edit over-scoped, bypass
        # a new LLM generation entirely and run targeted recovery instead.
        # Keyword scan only — no LLM call for detection.
        if last_feedback and trust_recovery.user_signals_overscope(feedback):
            corrected = trust_recovery.recover_error3(
                last_feedback, frozen_profile, preference_json
            )
            if corrected:
                frozen_profile = corrected
                # Rebuild message history from the corrected profile so
                # future edits are grounded in the accurate state.
                messages = [
                    {"role": "system", "content": STAGE4_SYSTEM},
                    {"role": "user", "content": "Here is the current profile:\n\n" + frozen_profile},
                    {"role": "assistant", "content": "Profile updated. What else would you like to change?"}
                ]
                last_feedback = None
            continue
        # ─────────────────────────────────────────────────────────────

        messages.append({
            "role": "user",
            "content": (
                f"{feedback}\n\n"
                "After updating the profile, briefly note what changed and suggest "
                "one or two things that might still be worth refining."
            )
        })

        ai_response = call_llm(messages, max_tokens=3000)
        if not ai_response:
            print("AI: Something went wrong. Let's try again.")
            continue

        messages.append({"role": "assistant", "content": ai_response})
        print(f"\n{'─' * 60}\n")
        print(ai_response)
        print(f"\n{'─' * 60}\n")

        # Advance the frozen reference to this new version.
        # If the user complains on the next turn, we revert to this.
        frozen_profile = ai_response
        last_feedback = feedback

# -------------------------------
# RUN FULL PROTOTYPE
# -------------------------------
def run_prototype():
    print("\n" + "=" * 60)
    print("  WELCOME TO THE AI RELATIONSHIP PROFILE BUILDER")
    print("=" * 60)
    print(
        "\nThis tool builds a detailed profile of your ideal connection "
        "through a guided conversation about you and your preferences")
    print("Before we begin — what kind of relationship are you hoping to explore?")
    print("(e.g., romantic partner, close friendship, professional mentor, etc.)\n")
    relationship_type = input("You: ").strip()
    print()

    print("\n===== Getting to Know You =====")
    user_preferences = stage1_intake(relationship_type)

    print("\n===== Mapping Your Priorities =====")
    user_preferences = stage_ranking(user_preferences)

    print("\n===== A Few Clarifications =====")
    user_preferences = stage2_tension(user_preferences)

    print("\n===== Building Your Profile =====")
    profile_text = stage3_profile(user_preferences)

    print("\n===== Refine Your Profile =====")
    stage4_refinement(user_preferences, profile_text)

    # Session summary of any trust recovery events
    if trust_recovery.recovery_log:
        print("\n" + "─" * 60)
        print("  SESSION TRUST RECOVERY SUMMARY")
        print("─" * 60)
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
        print("─" * 60 + "\n")

# -------------------------------
# START
# -------------------------------
if __name__ == "__main__":
    run_prototype()