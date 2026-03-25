import re
import json
import streamlit as st
import streamlit.components.v1 as components

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "llama-3.1-8b.gguf"
TEST_MODE = False  # Set to False to use the real model

# 24 mock LLM responses covering the full happy path + trust recovery.
# Ordered to match the call sequence:
#   R0-R4:  about_you questions (4 Qs + SUMMARY)
#   R5:     extract_user_portrait JSON (hidden)
#   R6-R7:  proposition trait map (first pass + revised after user tweak)
#   R8:     deal breakers
#   R9:     extract_proposition JSON (hidden)
#   R10-R12: tension clarifying questions
#   R13:    tension wrap-up
#   R14:    profile generation
#   R15-R16: refinement updates
#   R17-R18: _llm_classify_confirmation fallbacks ("CONFIRM")
#   R19:    error1 recovery — corrected model summary
#   R20:    error2 recovery — assumption audit opener
#   R21:    error2 recovery — corrected profile
#   R22:    error3 recovery — reverted + targeted edit
#   R23:    final refinement update
_TEST_LLM_RESPONSES = [
    # R0 — about_you Q1: personality
    "What's something you genuinely enjoy that you think says a lot about who you are as a person?",
    # R1 — about_you Q2: lifestyle
    "When you have a free afternoon with no plans or obligations, how do you usually end up spending it?",
    # R2 — about_you Q3: how you show up in relationships
    "How would someone who knows you really well describe the way you show up in close relationships?",
    # R3 — about_you Q4: what you're looking for in a partner
    "What kind of person do you imagine being most at ease with — in terms of their energy, pace, and how they move through the world?",
    # R4 — about_you SUMMARY: triggers end of stage and summary confirmation
    "SUMMARY: You are a thoughtful, introspective person who values depth and genuine connection. You gravitate toward meaningful conversations and quiet, intentional ways of spending time. In relationships, you show up as the steady, caring presence — the one who listens deeply and remembers the small things. You tend to do best with someone whose energy is warm and grounded, someone who takes their time with people and doesn't need to fill every silence.",
    # R5 — extract_user_portrait JSON (never shown to user)
    '{"personality_traits": ["thoughtful", "introspective", "empathetic", "steady"], "communication_style": "Listens carefully, speaks with intention — quiet but deeply engaged", "values": ["authenticity", "depth", "loyalty", "growth"], "lifestyle": "Enjoys meaningful conversations, reading, time in nature, intentional downtime", "social_energy": "Introverted — recharges alone, values small close connections over large social groups", "thinking_style": "Reflective and intuitive, drawn to meaning and pattern", "decision_making": "Empathy-first — considers emotional impact before logic", "structure_vs_spontaneity": "Prefers structure and predictability, open to flexibility within that", "openness_to_experience": "Curious but measured — eases into novelty at their own pace", "relationship_tendencies": "Loyal, caring, shows love through attentiveness and consistency", "big_five_estimates": {"extraversion": "low", "openness": "medium-high", "agreeableness": "high", "conscientiousness": "medium-high", "neuroticism": "low-medium"}}',
    # R6 — proposition trait map first pass
    "Based on everything you've shared, here's what I see in you and what I think you'd need most in a connection:\n\n1. **Social energy** — You recharge alone and do best in small, close relationships. Someone who matches this quieter energy would feel like a relief, not a compromise.\n2. **Thinking style** — You're drawn to depth and meaning. Surface-level small talk drains you; a partner who can go deep on ideas and feelings is essential.\n3. **Decision-making** — You lead with how things feel before you reach for logic. You'd need someone emotionally present enough to meet you there.\n4. **Structure** — You like having a plan. A partner who respects that rhythm — rather than constantly pulling you into chaos — will let you thrive.\n5. **Openness** — You're genuinely curious but move at your own pace. You'd do well with someone who introduces new things gently, not forcefully.\n6. **Connection through loyalty** — You love through consistency and small, remembered details. Someone who shows up the same way every time is more meaningful to you than grand gestures.\n\nDoes this feel right, or would you adjust anything?",
    # R7 — proposition revised trait map (after user feedback)
    "Got it — here's the updated reflection:\n\n1. **Social energy** — You recharge alone and do best in small, close relationships. Someone who matches this quieter energy would feel like a relief, not a compromise.\n2. **Thinking style** — You're drawn to depth and meaning. Surface-level small talk drains you; a partner who can go deep on ideas and feelings is essential.\n3. **Decision-making** — You lead with how things feel before you reach for logic. You'd need someone emotionally present enough to meet you there.\n4. **Structure** — You like having a plan. A partner who respects that rhythm — rather than constantly pulling you into chaos — will let you thrive.\n5. **Openness** — You're genuinely curious but move at your own pace. You'd do well with someone who introduces new things gently, not forcefully.\n6. **Connection through presence** — You love through undivided attention and small, consistent moments. Someone who is truly there — not distracted or half-present — matters more to you than elaborate gestures.\n\nDoes this feel right now?",
    # R8 — deal breakers
    "Based on who you are, I think these would be genuine deal breakers for you:\n\n1. **Emotional unavailability** — You lead with empathy and need that reciprocated. Someone who shuts down or goes cold would leave you feeling invisible.\n2. **Dismissiveness** — You think carefully before you speak. A partner who brushes off what you share would erode trust quickly.\n3. **Constant unpredictability** — You're not rigid, but you need a stable foundation. Someone who creates ongoing chaos would exhaust and unsettle you.\n\nLet me know if these feel right, or if you'd like to add, remove, or change anything.",
    # R9 — extract_proposition JSON (never shown to user)
    '{"relationship_type": "romantic partner", "user_trait_summary": "Thoughtful, empathetic, introverted. Values depth and loyalty. Leads with emotion, prefers structure, curious but measured. Shows love through presence and consistency.", "selected_dimensions": [{"category": "Main reflection", "ranked_items": [{"item": "Social energy — quiet, close connections over large circles", "reasoning": "User is introverted and recharges alone"}, {"item": "Thinking style — depth and meaning over surface talk", "reasoning": "User drawn to reflective, meaningful conversation"}, {"item": "Decision-making — emotion-first", "reasoning": "User leads with empathy before logic"}, {"item": "Structure — respects planning and rhythm", "reasoning": "User prefers predictability with flexibility"}, {"item": "Openness — curious but measured pace", "reasoning": "User eases into novelty"}, {"item": "Connection through presence and consistency", "reasoning": "User values undivided attention and small consistent moments"}]}], "deal_breakers": ["Emotional unavailability", "Dismissiveness", "Constant unpredictability"]}',
    # R10 — tension Q1
    "I noticed something worth exploring — you value both emotional depth and structure, but deep emotional connection can sometimes be unpredictable and messy. How do you hold those two things together — does emotional depth feel worth the instability it sometimes brings, or do you need a relationship that stays fairly even-keeled?",
    # R11 — tension Q2
    "That makes a lot of sense. You also described yourself as someone who eases into new things at your own pace — but growth in a relationship often means being pushed a little outside your comfort zone. How much do you want a partner who challenges you versus one who simply supports where you already are?",
    # R12 — tension Q3
    "One last thing — you mentioned showing love through consistency and small remembered details, but also valuing someone who can go emotionally deep. Do you need your partner to express things verbally, or is it enough if they show up consistently even if they're not naturally expressive with words?",
    # R13 — tension wrap-up (after user answers Q3)
    "That's a really clear picture — you want emotional depth and consistency to coexist, and you'd rather be challenged gently from within a stable foundation than pulled into constant change. That all fits together well.",
    # R14 — profile generation
    "**Meet Soren, a 31 year old man.**\n\n### Personality & Core Traits\nSoren is the kind of person who actually listens — not politely, but with the kind of attention that makes you feel like the most important person in the room. He's unhurried and deliberate, someone who thinks before he speaks and means what he says. He has a dry, quiet sense of humor that surfaces when you least expect it. He's not trying to impress anyone; he's just genuinely himself.\n\n### Communication Style\nSoren doesn't fill silence for the sake of it. He'll sit with you in a quiet moment without reaching for his phone. When he does speak, it tends to be worth hearing — considered, specific, and warm. He's honest without being blunt, and he knows how to name what he's feeling without making it dramatic.\n\n### Emotional Style & Love Languages\nHis primary love language is quality time — undistracted, unhurried presence. He also notices the small things: he'll remember what you said you were anxious about last week and ask how it went. He's not performative with affection, but the consistency of it is unmistakable.\n\n### A Typical Day in His Life\nSoren starts his mornings slowly — coffee, reading, thirty minutes without a screen. He works in landscape architecture and spends his afternoons between a desk and project sites. His evenings are quiet: cooking something from scratch, a long walk, or a film he actually wants to talk about afterward.\n\n### Conflict Style\nHe doesn't avoid hard conversations, but he doesn't rush into them either. He takes a breath, waits until he can speak from understanding rather than reaction, and leads with curiosity — what happened, what did you need, what can we do differently.\n\n### Backstory\nSoren grew up in a mid-sized city, the eldest of three. He was close to his mother, who was a high school art teacher, and learned early that paying attention to people was its own kind of love. He had one long relationship in his late twenties that ended amicably when they realized they wanted different versions of the future.\n\n### Why This Person Fits You\nSoren offers exactly the combination you described: emotional depth within a stable, consistent presence. He won't overwhelm you with intensity, but he won't give you surface level either. His pace matches yours — unhurried, intentional, showing up the same way every time.",
    # R15 — refinement update 1
    "**Meet Soren, a 31 year old man.**\n\n### Personality & Core Traits\nSoren is the kind of person who actually listens — not politely, but with the kind of attention that makes you feel like the most important person in the room. He's unhurried and deliberate, someone who thinks before he speaks and means what he says. He has a dry, quiet sense of humor that surfaces when you least expect it.\n\n### Communication Style\nSoren doesn't fill silence for the sake of it. When he does speak, it tends to be worth hearing — considered, specific, and warm.\n\n### Emotional Style & Love Languages\nHis primary love language is quality time — undistracted, unhurried presence. He notices the small things and shows up the same way every time.\n\n### A Typical Day in His Life\nSoren starts his mornings with a long run before the city wakes up, then coffee and reading. He works in landscape architecture. His evenings are quiet: cooking something from scratch, a long walk, or a film he actually wants to talk about afterward. On weekends you'd find him at the climbing gym or on a trail he hasn't tried yet.\n\n### Conflict Style\nHe doesn't avoid hard conversations. He takes a breath, waits until he can speak from understanding rather than reaction, and leads with curiosity.\n\n### Backstory\nSoren grew up in a mid-sized city, the eldest of three, close to his mother who was a high school art teacher.\n\n### Why This Person Fits You\nSoren offers emotional depth within a stable, consistent presence. His active lifestyle adds energy without pressure to change.\n\n*Updated: Added outdoor and climbing interests to A Typical Day. You might also consider personalizing his career or adding a small quirk.*",
    # R16 — refinement update 2
    "**Meet Soren, a 31 year old man.**\n\n### Personality & Core Traits\nSoren is the kind of person who actually listens — not politely, but with genuine attention. He collects vintage maps — not seriously, just the ones that catch his eye at markets — and has an inexplicable loyalty to a single brand of terrible instant coffee that he drinks completely unironically.\n\n### Communication Style\nSoren doesn't fill silence for the sake of it. When he speaks, it's worth hearing — considered, specific, and warm.\n\n### Emotional Style & Love Languages\nHis primary love language is quality time — undistracted, unhurried presence. He notices the small things.\n\n### A Typical Day in His Life\nMornings start with a run, then coffee and reading. Evenings are quiet: cooking, a long walk, or a film he wants to talk about afterward.\n\n### Conflict Style\nHe takes a breath before responding, leads with curiosity, and believes conflict handled well can bring people closer.\n\n### Backstory\nGrew up in a mid-sized city, eldest of three, close to his art-teacher mother.\n\n### Why This Person Fits You\nSoren offers exactly the depth and consistency you need. The quirks make him feel real rather than ideal.\n\n*Updated: Added the map collection and instant coffee detail. No further changes suggested — this feels like a solid final version.*",
    # R17 — _llm_classify_confirmation fallback
    "CONFIRM",
    # R18 — _llm_classify_confirmation fallback
    "CONFIRM",
    # R19 — error1 recovery: corrected model summary
    "To make sure we are aligned — you're looking for someone who is emotionally present and warm, but you've clarified that you don't need constant verbal affirmation; consistent actions and attentiveness matter more to you than expressive words.",
    # R20 — error2 recovery: assumption audit opener
    "I want to walk back through a few things I included in the profile to make sure they all come from what you actually told me, rather than assumptions I made. Let me go through them one by one.",
    # R21 — error2 recovery: corrected profile
    "I've updated the profile based on your corrections — removing the inferred details and keeping only what you confirmed.\n\n**Meet Soren, a 31 year old man.**\n\nSoren is thoughtful, unhurried, and genuinely present. He listens in a way that makes people feel heard. He's consistent — someone who shows up the same way each time, without drama or performance. The rest of his profile has been revised to reflect only what you confirmed, with the assumptions removed.",
    # R22 — error3 recovery: reverted and targeted edit applied
    "I overstepped — I changed more than you asked. I've reverted to the previous version and applied only the one change you requested.\n\n**Meet Soren, a 31 year old man.**\n\nEverything is as it was, with only the specific detail you asked to change updated. Let me know if anything else feels off.",
    # R23 — final refinement update
    "**Meet Soren, a 31 year old man.**\n\nThe profile is updated with your final note. I think this version feels complete — specific enough to feel real, and grounded in everything you shared.\n\n*No further changes suggested — this feels like a solid final version.*",
]

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

INTRO_ACKNOWLEDGMENT = (
    "Great choice! Let's explore what makes a great connection for you. "
    "First, I'll get to know you as a person."
)


def intro_acknowledgment_message(_relationship_description: str = "") -> str:
    """Fixed intro line after the user describes what they're looking for."""
    return INTRO_ACKNOWLEDGMENT

def render_sidebar_timeline():
    ss = st.session_state
    current_idx = STAGES.index(ss.stage)
    stages_to_show = STAGES[:-1]

    for i, stage_key in enumerate(stages_to_show):
        stage_idx = STAGES.index(stage_key)
        is_done = stage_idx < current_idx
        is_active = stage_idx == current_idx

        # Connector line above (skip first)
        if i > 0:
            color = "#fecdd3" if stage_idx <= current_idx else "#e5e7eb"
            st.markdown(f'<div style="width:2px;height:16px;background:{color};margin-left:9px;"></div>', unsafe_allow_html=True)

        # Dot styling
        if is_done:
            bg, border, text, lbl = "#f43f5e", "#f43f5e", "✓", "color:#1a1a1a;"
        elif is_active:
            bg, border, text, lbl = "transparent", "#f43f5e", "", "color:#1a1a1a;font-weight:700;"
        else:
            bg, border, text, lbl = "transparent", "#ffffff", "", "color:#1a1a1a;"

        st.markdown(f'<div style="display:flex;align-items:center;gap:12px;"><div style="width:20px;height:20px;border-radius:50%;background:{bg};border:2px solid {border};display:flex;align-items:center;justify-content:center;font-size:10px;color:white;font-weight:700;flex-shrink:0;">{text}</div><span style="font-size:14px;{lbl}">{STAGE_LABELS[stage_key]}</span></div>', unsafe_allow_html=True)

        if is_active:
            _render_substeps_inline(stage_key)


def _render_substeps_inline(stage_key: str):
    ss = st.session_state

    if stage_key == "about_you":
        steps = ["Chat about you", "Confirm summary"]
        active_i = 1 if ss.get("awaiting_summary_confirmation") else 0
    elif stage_key == "proposition":
        steps = ["What you're looking for", "Deal breakers"]
        active_i = 0 if not ss.get("trait_map_confirmed") else 1
    elif stage_key == "refinement":
        steps = ["Does it feel right?", "Fine-tune (optional)"]
        active_i = 0 if ss.get("awaiting_initial_refinement") else 1
    else:
        return

    for i, label in enumerate(steps):
        if i < active_i:
            txt = "color:#1a1a1a;"
            line_color = "#f43f5e"
        elif i == active_i:
            txt = "color:#1a1a1a;font-weight:700;"
            line_color = "#e5e7eb"  # gray from active onward
        else:
            txt = "color:#1a1a1a;"
            line_color = "#e5e7eb"

        st.markdown(f'<div style="display:flex;align-items:stretch;"><div style="width:2px;background:{line_color};margin-left:9px;margin-right:22px;flex-shrink:0;"></div><div style="padding:5px 0;"><span style="font-size:12px;{txt}">{label}</span></div></div>', unsafe_allow_html=True)
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
Do not add the tag to every response — use it sparingly and purposefully.

ADDITIONS vs CONTRADICTIONS (especially during the proposition reflection / deal breakers):
  If the user simply adds a detail, hobby, or wish (e.g. wants someone to play a sport with them) \
that fits what you already said, treat it as an ADD-ON: weave it in briefly. Do NOT use error1 \
and do NOT re-litigate the whole portrait as if it were a conflict unless they explicitly \
disagree with something you wrote. Reserve error1 for real ambiguity or a clear contradiction.\
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


def get_unified_proposition_system_prompt(relationship_type):
    return (
        "You are a warm, insightful relationship coach. You have a structured portrait of the user.\n\n"
        f"They are looking for: {relationship_type}.\n\n"
        "Your job is **one** assistant message that reflects how you read them **and** what seems to matter "
        "for this kind of connection — as a **single** numbered list with **uniform depth**. "
        "Do not give a short skim for some items and long paragraphs for others; every item should be similarly detailed.\n\n"
        "OUTPUT FORMAT:\n"
        "1) Opening: one sentence with this meaning (your wording): "
        "\"Here are the main things I think you are looking for in this relationship.\" "
        "Use \"this connection\" instead of \"relationship\" when it fits better (e.g. bandmate, study partner).\n"
        "2) Blank line, then **one** numbered list only — no separate markdown bullet block, no second list, no extra headings.\n"
        "   - **5-8 items** total.\n"
        "   - Each item: a short **bold line or title**, then **2–4 sentences** grounded in what they said. "
        "Use the same richness for every item.\n"
        "   - The **first five items** must cover these dimensions (clear labels, your phrasing): "
        "Social energy (introvert ↔ extravert); Thinking style (concrete ↔ abstract); "
        "Decision-making (head ↔ heart); Structure (planner ↔ spontaneous); Openness (comfort ↔ novelty).\n"
        "   - The **remaining items** are themes for *this* connection type (collaboration, logistics, creative fit, "
        "pace, boundaries, etc.) — only what their answers support.\n"
        "3) One line break, then ask exactly: \"Does this feel right, or would you adjust anything?\"\n\n"
        "STRICT RULES:\n"
        "- Describe how they show up and what would fit them; save blunt non‑negotiables for the separate deal-breakers step.\n"
        "- Frame as inference (\"I hear…\", \"Based on what you shared…\") — not \"You need…\".\n"
        "- Do NOT list deal breakers here.\n"
        "- If revising after user feedback, keep the same shape (opening + one numbered list + same closing question); "
        "only change what their message affects. Do NOT use [TRUST_RECOVERY:error1] for simple add-ons.\n\n"
        f"{TRUST_RECOVERY_INSTRUCTIONS}"
    )

def get_deal_breakers_system_prompt(relationship_type):
    return (
        "You are a warm, insightful relationship coach. Based on everything you know "
        "about the user — their confirmed reflection on what they're looking for and the conversation so far — infer "
        f"2-4 deal breakers for their {relationship_type}.\n\n"
        "These should be behaviors or traits that are non-negotiable given who this person is. Frame them "
        "as what would be genuinely incompatible with the user's personality and needs.\n\n"
        "OUTPUT — After listing the deal breakers, you MUST end with a single short closing paragraph or line that:\n"
        "- Invites them to **confirm** if these deal breakers feel right, AND\n"
        "- Makes clear they can **change** anything (add, remove, or revise).\n"
        "Example (paraphrase in your own voice): \"Let me know if you're happy to go with these, or if you'd like "
        "to change anything.\" Do not skip this closing — the user needs an explicit confirm-or-change prompt.\n\n"
        "STRICT RULES:\n"
        "- Present ONLY deal breakers (then the closing question). Do NOT repeat the long numbered reflection list.\n"
        "- Keep each deal breaker to one concise line.\n"
        '- Frame as inference: "Based on who you are, I think these would be real problems..."\n\n'
        f"{TRUST_RECOVERY_INSTRUCTIONS}"
    )


TENSION_SYSTEM_PROMPT = (
    "You are a warm relationship coach having a conversation with the user. "
    "Always address the user directly as 'you' — speak to them, not about them. "
    "Analyze their inferred relationship priorities and detect any internal contradictions or tensions.\n\n"
    "Each time you reply, the user message will say which clarification this is (**1 of 3**, **2 of 3**, or **3 of 3**). "
    "Follow that line strictly:\n"
    "- Ask **exactly ONE** clarifying question per message (one `?` only). "
    "Optional: at most 1–2 short setup sentences before the question.\n"
    "- Do **not** stack, number, or bullet multiple questions.\n"
    "- On **3 of 3**: ask one final clarifying question. Do NOT add any disclaimers, notes, "
    "or remarks about the question budget or whether the user needs to answer. Just ask the question naturally.\n"
    "- If priorities already feel clear before turn 3, you may write [RESOLVED] on its own line and add a brief warm closing "
    "with **no** question — otherwise keep probing until turn 3.\n\n"
    "Do not resolve tensions for them — let the user think through them.\n\n"
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
    "- The **very first line** of the profile (before any section headers) must be exactly this pattern: "
    "**Meet [FirstName].** — invent one plausible first name at random (vary style/culture). Add age in there too"
    "If the user already gave a name in their ideas, use that first name instead. "
    "Do not use the user's own name or a famous real person's name unless they asked.\n"
    "- After that opening line, add a blank line, then use clear headers for each section (you may still "
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
    "Keep the opening line **Meet [FirstName].** as the first line unless the user explicitly asks to rename them. "
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
    "into JSON format. The user confirmed one detailed numbered reflection (dimensions + connection themes) "
    "— represent that as one object in selected_dimensions "
    'with category \"Main reflection\" (or similar) and ranked_items in order (one entry per numbered item). '
    "Deal breakers are separate. "
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

def call_llm(messages, temperature=0.7, max_tokens=3000):
    if TEST_MODE:
        idx = st.session_state.get("test_llm_idx", 0)
        st.session_state.test_llm_idx = idx + 1
        if idx < len(_TEST_LLM_RESPONSES):
            return _TEST_LLM_RESPONSES[idx]
        return "I think we have everything we need."
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

# If user says "yes, these are right" we should confirm — not only exact "yes".
_CONFIRMATION_EXACT = frozenset({
    "yes", "yeah", "yep", "yup", "looks good", "that's right", "that is right",
    "correct", "good", "perfect", "spot on", "exactly", "sure", "ok", "okay",
    "that's it", "all good", "looks right", "looks great", "no changes", "fine",
    "done", "agreed", "sounds good", "works for me", "nothing to add",
    "feels good", "feel good", "feels great", "feels right", "that's great",
    "love it", "love this", "nailed it",
})
_CONFIRMATION_PHRASES = (
    "that's right", "that is right", "these are right", "this is right",
    "looks good", "sounds good", "works for me", "all good", "spot on",
    "no changes", "nothing to change", "happy with", "good with these",
    "i agree", "sounds right", "they're right", "they are right",
    "feels good", "feels right", "feels great", "sounds great",
)
# If message starts with yes but asks for edits, do not treat as pure confirmation.
_CONFIRMATION_NEGATORS = (
    " but ", " except ", " change", " remove", " add ", " wrong", " not quite",
    " actually", " instead", " don't want", " drop ", " replace", " edit ",
    " tweak", " adjust", " delete",
)


def _llm_classify_confirmation(user_input: str) -> bool:
    """Fallback: ask the LLM whether the user is confirming or giving feedback."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a classifier. The user just replied to a prompt asking them to confirm or suggest changes. "
                "Determine whether the user is CONFIRMING (accepting, agreeing, expressing satisfaction) "
                "or giving FEEDBACK (requesting changes, additions, or corrections).\n\n"
                "Reply with exactly one word: CONFIRM or FEEDBACK"
            ),
        },
        {"role": "user", "content": user_input},
    ]
    response = call_llm(messages, temperature=0.0, max_tokens=10)
    if response:
        return "confirm" in response.strip().lower()
    return False


def user_signals_confirmation(user_input: str) -> bool:
    """True when the user is accepting / confirming, including 'yes, these are right'."""
    t = (user_input or "").lower().strip()
    if not t:
        return False
    # Fast path: exact matches
    if t in _CONFIRMATION_EXACT:
        return True
    # Fast path: known phrases
    if any(p in t for p in _CONFIRMATION_PHRASES):
        if any(n in t for n in _CONFIRMATION_NEGATORS):
            return False
        return True
    # Fast path: starts with a confirmation word
    first_token = t.split(maxsplit=1)[0] if t else ""
    first_clean = first_token.strip(".,!?;:\"'")
    if first_clean in {"yes", "yeah", "yep", "yup", "correct", "perfect", "sure", "ok", "okay", "right", "fine"}:
        if any(n in t for n in _CONFIRMATION_NEGATORS):
            return False
        return True
    # Fast path: negators present means definitely feedback
    if any(n in t for n in _CONFIRMATION_NEGATORS):
        return False
    # Slow path: ambiguous input — ask the LLM
    return _llm_classify_confirmation(user_input)


# ── Live Trait Transparency ─────────────────────────────────────────────
# Dimensions backed by Big Five / MBTI frameworks.
DIMENSION_LABELS = {
    "social_energy":   "Social Energy",
    "thinking_style":  "Thinking Style",
    "decision_making": "Decision-Making",
    "structure":       "Structure",
    "openness":        "Openness",
    "emotional_tone":  "Emotional Tone",
    "communication":   "Communication",
    "values":          "Core Values",
}
ADD_THRESHOLD        = 0.75   # min confidence to add a card
ADD_MIN_EVIDENCE     = 2      # min distinct turns that must have signalled this dimension
REMOVE_THRESHOLD     = 0.40   # below this AND evidence_count drops → remove card
BIG5_TO_DIM = {
    "extraversion":      "social_energy",
    "openness":          "openness",
    "agreeableness":     "emotional_tone",
    "conscientiousness": "structure",
    "neuroticism":       "emotional_tone",
}

# TEST_MODE fixture — cards only appear after 2+ supporting turns.
# evidence_count tracks how many turns contributed to each dimension.
_PROGRESSIVE_LIVE_TRAITS = [
    {},   # turn 1: too early, gathering signals
    {},   # turn 2: still building — nothing confident enough yet
    {"social_energy":  {"label": "Introvert",  "confidence": 0.78, "evidence_count": 2}},
    {"social_energy":  {"label": "Introvert",  "confidence": 0.82, "evidence_count": 3},
     "thinking_style": {"label": "Analytical", "confidence": 0.76, "evidence_count": 2}},
    {"social_energy":  {"label": "Introvert",  "confidence": 0.84, "evidence_count": 4},
     "thinking_style": {"label": "Analytical", "confidence": 0.79, "evidence_count": 3},
     "values":         {"label": "Authentic",  "confidence": 0.76, "evidence_count": 2}},
    {"social_energy":  {"label": "Introvert",  "confidence": 0.87, "evidence_count": 5},
     "thinking_style": {"label": "Analytical", "confidence": 0.82, "evidence_count": 4},
     "values":         {"label": "Authentic",  "confidence": 0.79, "evidence_count": 3},
     "openness":       {"label": "Curious",    "confidence": 0.76, "evidence_count": 2}},
]


def extract_live_traits(stage_messages):
    """Return {dim: {label, confidence}} from conversation so far."""
    user_turns = sum(1 for m in stage_messages if m["role"] == "user")
    if user_turns == 0:
        return {}
    if TEST_MODE:
        idx = min(user_turns - 1, len(_PROGRESSIVE_LIVE_TRAITS) - 1)
        return _PROGRESSIVE_LIVE_TRAITS[idx]

    conversation_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in stage_messages
        if m["role"] in ("user", "assistant")
    )
    messages = [
        {"role": "system", "content": (
            "You are carefully analyzing a user's personality from a conversation. "
            "Be conservative — one mention of something is never enough to reach high confidence. "
            "A trait only earns high confidence when the user has said multiple things, "
            "across different messages, that consistently point to the same conclusion. "
            "Return ONLY a JSON object. Valid dimension IDs: social_energy, thinking_style, "
            "decision_making, structure, openness, emotional_tone, communication, values. "
            "Each value: {\"label\": \"short phrase (3 words max)\", \"confidence\": 0.0-1.0, "
            "\"evidence_count\": N} where evidence_count is the number of DISTINCT user "
            "statements that support this dimension. "
            "Only include a dimension if evidence_count >= 2 AND confidence >= 0.60. "
            "Prefer returning 1-2 well-supported traits over many guesses. "
            "No other text. Example: "
            "{\"social_energy\": {\"label\": \"Introvert\", \"confidence\": 0.82, \"evidence_count\": 3}}"
        )},
        {"role": "user", "content": f"Analyze:\n\n{conversation_text}"}
    ]
    response = call_llm(messages, temperature=0.1, max_tokens=400)
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        return json.loads(response[start:end])
    except Exception:
        return {}


def _merge_live_traits(new_traits):
    """Merge {dim: {label, confidence, evidence_count}} into live_traits with strict gating.

    A card is only added when BOTH confidence >= ADD_THRESHOLD AND
    evidence_count >= ADD_MIN_EVIDENCE. Once visible, a card is only
    removed if confidence drops below REMOVE_THRESHOLD. Evidence counts
    accumulate across calls (never decrease).
    """
    current = st.session_state.live_traits
    for dim, data in new_traits.items():
        conf = data.get("confidence", 0)
        ev   = data.get("evidence_count", 1)
        if dim in current:
            # Always accumulate evidence; keep the higher confidence
            prev_ev   = current[dim].get("evidence_count", 1)
            new_ev    = max(prev_ev, ev)
            new_conf  = max(current[dim].get("confidence", 0), conf)
            if conf < REMOVE_THRESHOLD and new_ev < ADD_MIN_EVIDENCE:
                del current[dim]
            else:
                current[dim] = {**data, "confidence": new_conf, "evidence_count": new_ev}
        elif conf >= ADD_THRESHOLD and ev >= ADD_MIN_EVIDENCE:
            current[dim] = data
    st.session_state.live_traits = current


def rank_live_traits():
    """Return top-6 trait dicts sorted by evidence density (evidence_count × confidence).

    Evidence density ranks traits that are backed by many independent signals
    above those that happen to have a single high-confidence inference.
    """
    live = st.session_state.get("live_traits", {})
    items = []
    for dim, data in live.items():
        conf = data.get("confidence", 0.5)
        ev   = data.get("evidence_count", 1)
        items.append({
            "id": dim,
            "dimension": DIMENSION_LABELS.get(dim, dim.replace("_", " ").title()),
            "label": data["label"],
            "confidence": conf,
            "_rank_score": ev * conf,   # internal — not sent to JS
        })
    items.sort(key=lambda x: x["_rank_score"], reverse=True)
    # Strip internal key before passing to JS
    for item in items:
        item.pop("_rank_score", None)
    return items[:6]


def _backfill_live_traits_from_portrait(portrait):
    """Seed live_traits from the extracted user_portrait after about_you ends."""
    portrait_dims = {
        "communication":   portrait.get("communication_style", ""),
        "values":          ", ".join(portrait.get("values", [])[:2]),
        "decision_making": portrait.get("decision_making", ""),
        "social_energy":   portrait.get("social_energy", ""),
        "thinking_style":  portrait.get("thinking_style", ""),
        "structure":       portrait.get("structure_vs_spontaneity", ""),
        "openness":        portrait.get("openness_to_experience", ""),
    }
    new_traits = {}
    for dim, val in portrait_dims.items():
        if val and str(val).strip():
            # Portrait is derived from the full conversation — treat as highly evidenced
            new_traits[dim] = {"label": str(val).strip(), "confidence": 0.80, "evidence_count": 5}
    big5 = portrait.get("big_five_estimates", {})
    for b5_key, dim in BIG5_TO_DIM.items():
        val = big5.get(b5_key, "")
        if val and str(val).strip() and dim not in new_traits:
            new_traits[dim] = {"label": str(val).strip(), "confidence": 0.75, "evidence_count": 4}
    _merge_live_traits(new_traits)
    _seed_match_priorities_from_portrait(portrait)


# ── Match Priority Seeding ───────────────────────────────────────────────
# What the user values in a partner — ranked by importance and editable.
PRIORITY_POOL = {
    "shared_values":           "Shared Values",
    "emotional_depth":         "Emotional Depth",
    "intellectual_connection": "Intellectual Connection",
    "space_independence":      "Space & Independence",
    "communication_style":     "Communication Style",
    "humor_warmth":            "Humor & Warmth",
    "life_goals":              "Life Goals",
    "reliability_trust":       "Reliability & Trust",
    "spontaneity":             "Spontaneity",
    "physical_chemistry":      "Physical Chemistry",
}

_TEST_PRIORITIES = [
    {"id": "shared_values",           "label": "Shared Values",           "reason": "You spoke a lot about authenticity and wanting someone whose principles align with yours."},
    {"id": "emotional_depth",         "label": "Emotional Depth",         "reason": "You value deep, meaningful conversations over surface-level connection."},
    {"id": "intellectual_connection", "label": "Intellectual Connection", "reason": "Curiosity and stimulating exchange came up consistently in how you described good relationships."},
    {"id": "space_independence",      "label": "Space & Independence",    "reason": "You described yourself as someone who needs room to recharge and pursue your own interests."},
]


def _seed_match_priorities_from_portrait(portrait):
    """Infer what the user values most in a partner and store as match_priorities.
    Only runs once — does not overwrite a list the user has already reordered.
    """
    if st.session_state.match_priorities:
        return  # already seeded or user has reordered

    if TEST_MODE:
        st.session_state.match_priorities = _TEST_PRIORITIES
        return

    portrait_text = json.dumps(portrait, indent=2)
    relationship_type = st.session_state.get("relationship_type", "relationship")
    messages = [
        {"role": "system", "content": (
            "Based on a user's personality portrait, infer 4-6 qualities they most "
            f"value in a partner for a {relationship_type}. "
            "Return ONLY a JSON array ordered from most to least important. "
            "Each item: {\"id\": \"...\", \"label\": \"short phrase (2-4 words)\", "
            "\"reason\": \"one sentence explaining why this matters given the portrait\"}. "
            "Prefer IDs from: shared_values, emotional_depth, intellectual_connection, "
            "space_independence, communication_style, humor_warmth, life_goals, "
            "reliability_trust, spontaneity, physical_chemistry. "
            "Create a new id+label+reason if none fit. No other text."
        )},
        {"role": "user", "content": f"Portrait:\n{portrait_text}"}
    ]
    response = call_llm(messages, temperature=0.2, max_tokens=600)
    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        st.session_state.match_priorities = json.loads(response[start:end])[:6]
    except Exception:
        st.session_state.match_priorities = [
            {"id": "shared_values",   "label": "Shared Values",   "reason": ""},
            {"id": "emotional_depth", "label": "Emotional Depth", "reason": ""},
        ]


_TEST_LOOKING_FOR = [
    {"id": "emotional_presence",     "label": "Emotional Presence",     "reason": "You want someone who is genuinely available — not just physically there, but emotionally tuned in."},
    {"id": "quiet_confidence",       "label": "Quiet Confidence",       "reason": "You're drawn to people who are sure of themselves without needing to prove it loudly."},
    {"id": "intellectual_curiosity", "label": "Intellectual Curiosity", "reason": "You want to be with someone who asks questions and finds the world genuinely interesting."},
    {"id": "steady_reliability",     "label": "Steady Reliability",     "reason": "Consistency matters to you — you want someone who shows up the same way every time."},
]


def _generate_looking_for_items(portrait, relationship_type):
    """Generate what the user is looking for in a partner — structured, ordered list.
    Only runs once — does not overwrite if already seeded.
    """
    if st.session_state.what_looking_for:
        return  # already seeded

    if TEST_MODE:
        st.session_state.what_looking_for = _TEST_LOOKING_FOR
        return

    portrait_text = json.dumps(portrait, indent=2)
    priorities_text = ""
    if st.session_state.match_priorities:
        priorities_text = "\n\nTheir confirmed match priorities (in order):\n" + "\n".join(
            f"{i+1}. {p['label']}" + (f" — {p['reason']}" if p.get("reason") else "")
            for i, p in enumerate(st.session_state.match_priorities)
        )
    messages = [
        {"role": "system", "content": (
            f"Based on the user's personality portrait and confirmed match priorities, "
            f"generate 4-6 specific qualities they are looking for in a {relationship_type}. "
            "These should be concrete traits or behaviours a partner should have. "
            "Return ONLY a JSON array ordered from most to least important. "
            "Each item: {\"id\": \"snake_case_id\", \"label\": \"short phrase (2-4 words)\", "
            "\"reason\": \"one sentence explaining why this fits given the portrait\"}. "
            "No other text."
        )},
        {"role": "user", "content": f"Portrait:\n{portrait_text}{priorities_text}"}
    ]
    response = call_llm(messages, temperature=0.2, max_tokens=600)
    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        st.session_state.what_looking_for = json.loads(response[start:end])[:6]
    except Exception:
        st.session_state.what_looking_for = [
            {"id": "emotional_presence", "label": "Emotional Presence", "reason": ""},
            {"id": "steady_reliability", "label": "Steady Reliability", "reason": ""},
        ]


_TEST_DEAL_BREAKERS = [
    {"id": "emotional_unavailability",  "label": "Emotional Unavailability",  "reason": "You lead with empathy and need that reciprocated — someone who shuts down would leave you feeling invisible."},
    {"id": "dismissiveness",            "label": "Dismissiveness",            "reason": "You think carefully before you speak; a partner who brushes off what you share erodes trust quickly."},
    {"id": "constant_unpredictability", "label": "Constant Unpredictability", "reason": "You need a stable foundation — ongoing chaos exhausts and unsettles you."},
]


def _generate_deal_breaker_items(portrait, relationship_type):
    """Generate the user's deal breakers as a structured, ordered list.
    Only runs once — does not overwrite if already seeded.
    """
    if st.session_state.deal_breaker_items:
        return  # already seeded

    if TEST_MODE:
        st.session_state.deal_breaker_items = _TEST_DEAL_BREAKERS
        return

    portrait_text = json.dumps(portrait, indent=2)
    priorities_text = ""
    if st.session_state.match_priorities:
        priorities_text = "\n\nConfirmed match priorities:\n" + "\n".join(
            f"{i+1}. {p['label']}" + (f" — {p['reason']}" if p.get("reason") else "")
            for i, p in enumerate(st.session_state.match_priorities)
        )
    looking_for_text = ""
    if st.session_state.what_looking_for:
        looking_for_text = "\n\nWhat they are looking for:\n" + "\n".join(
            f"{i+1}. {p['label']}" + (f" — {p['reason']}" if p.get("reason") else "")
            for i, p in enumerate(st.session_state.what_looking_for)
        )
    messages = [
        {"role": "system", "content": (
            f"Based on the user's personality portrait, confirmed priorities, and what they are looking for, "
            f"infer 2-4 genuine deal breakers for their {relationship_type}. "
            "These should be behaviours or traits that are non-negotiable given who this person is — "
            "things that would be genuinely incompatible with their personality and needs. "
            "Return ONLY a JSON array ordered from most to least critical. "
            "Each item: {\"id\": \"snake_case_id\", \"label\": \"short phrase (2-4 words)\", "
            "\"reason\": \"one sentence explaining why this would be incompatible\"}. "
            "No other text."
        )},
        {"role": "user", "content": f"Portrait:\n{portrait_text}{priorities_text}{looking_for_text}"}
    ]
    response = call_llm(messages, temperature=0.2, max_tokens=600)
    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        st.session_state.deal_breaker_items = json.loads(response[start:end])[:4]
    except Exception:
        st.session_state.deal_breaker_items = [
            {"id": "emotional_unavailability", "label": "Emotional Unavailability", "reason": ""},
            {"id": "dismissiveness",           "label": "Dismissiveness",           "reason": ""},
        ]


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
    if "big6_traits" not in st.session_state:
        st.session_state.big6_traits = []
    if "live_traits" not in st.session_state:
        st.session_state.live_traits = {}
    if "match_priorities" not in st.session_state:
        st.session_state.match_priorities = []
    if "awaiting_priority_ranking" not in st.session_state:
        st.session_state.awaiting_priority_ranking = False
    if "awaiting_looking_for_ranking" not in st.session_state:
        st.session_state.awaiting_looking_for_ranking = False
    if "what_looking_for" not in st.session_state:
        st.session_state.what_looking_for = []
    if "deal_breaker_items" not in st.session_state:
        st.session_state.deal_breaker_items = []
    if "awaiting_deal_breaker_ranking" not in st.session_state:
        st.session_state.awaiting_deal_breaker_ranking = False
    if "test_llm_idx" not in st.session_state:
        st.session_state.test_llm_idx = 0

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
        st.session_state.awaiting_deal_breaker_ranking = False

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

        # Progressively surface traits in Big 6 as answers come in (every 2nd turn)
        _ut = sum(1 for m in st.session_state.stage_messages if m["role"] == "user")
        if _ut % 2 == 0 or _ut <= 2:
            _new_live = extract_live_traits(st.session_state.stage_messages)
            if _new_live:
                _merge_live_traits(_new_live)

        if check_stage_completion("about_you", ai_response):
            st.session_state.messages.append({"role": "assistant", "content": "Does this capture you well? Feel free to correct anything or add something important I missed."})
            st.session_state.awaiting_summary_confirmation = True

def handle_summary_confirmation(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.awaiting_summary_confirmation = False

    if not user_signals_confirmation(user_input) and user_input.lower().strip() != "":
        st.session_state.stage_messages.append({"role": "assistant", "content": st.session_state.stage_messages[-2]["content"]})
        st.session_state.stage_messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": "Got it — thanks for the clarification. I'll make sure that's reflected."})
    # On confirm: no filler assistant line — trait map supplies the next message and its own intro.

    with st.spinner("Analyzing your response..."):
        st.session_state.user_portrait = extract_user_portrait(st.session_state.stage_messages)
    _backfill_live_traits_from_portrait(st.session_state.user_portrait)
    st.session_state.awaiting_priority_ranking = True

def start_proposition_stage(priorities_confirmed=False):
    st.session_state.trait_map_confirmed = priorities_confirmed
    st.session_state.proposition_categories = []
    st.session_state.current_category_index = 0
    st.session_state.awaiting_deal_breakers = False
    st.session_state.deal_breakers_confirmed = False

    user_context_text = (
        f"The user is looking for: {st.session_state.relationship_type}\n\n"
        f"Here is the structured portrait of who they are:\n"
        f"{json.dumps(st.session_state.user_portrait, indent=2)}"
    )
    system_msg = {
        "role": "system",
        "content": get_unified_proposition_system_prompt(st.session_state.relationship_type),
    }
    user_context = {"role": "user", "content": user_context_text}
    st.session_state.stage_messages = [system_msg, user_context]

    if priorities_confirmed:
        # Skip the unified LLM — show interactive "what you are looking for" ranking instead
        with st.spinner("Building your profile details..."):
            _generate_looking_for_items(
                st.session_state.user_portrait,
                st.session_state.relationship_type,
            )
        st.session_state.awaiting_looking_for_ranking = True
    else:
        # Original flow: fire the unified proposition LLM
        with st.spinner("Reflecting on what you're looking for..."):
            ai_response = call_llm(st.session_state.stage_messages, max_tokens=4000)

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
        # User is reacting to the unified reflection (one detailed list) — confirm or revise
        user_is_confirming = user_signals_confirmation(user_input)

        if not user_is_confirming:
            trait_messages = [
                {
                    "role": "system",
                    "content": get_unified_proposition_system_prompt(st.session_state.relationship_type),
                },
                {"role": "user", "content": user_context_text},
                {
                    "role": "assistant",
                    "content": (
                        st.session_state.stage_messages[-2]["content"]
                        if len(st.session_state.stage_messages) >= 2
                        else ""
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"The user's message: {user_input}\n\n"
                        "CRITICAL: You must copy EVERY unchanged item word-for-word from your previous response. "
                        "Only modify the specific item(s) the user mentioned. Do NOT rewrite, rephrase, reorder, "
                        "or regenerate items the user did not ask to change. "
                        "For example, if the user says 'I want someone who is a planner but also flexible', "
                        "only update the item about structure/spontaneity — every other item must appear exactly as before. "
                        "Keep the same shape (opening + one numbered list + same closing question). "
                        "Do NOT use [TRUST_RECOVERY:error1] unless there is genuine ambiguity or contradiction."
                    ),
                },
            ]

            with st.spinner("Updating reflection..."):
                ai_response = call_llm(trait_messages, max_tokens=4000)

            if ai_response:
                st.session_state.recovery_pending = trust_recovery.ai_signals_recovery(ai_response)
                ai_response = TrustRecoverySystem.strip_recovery_tag(ai_response)
                st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
        else:
            # User confirmed the unified list — go straight to deal breakers (no second list)
            st.session_state.trait_map_confirmed = True
            st.session_state.proposition_categories = []
            st.session_state.current_category_index = 0

            confirmed_text = _get_proposition_conversation_text()
            db_messages = [
                {"role": "system", "content": get_deal_breakers_system_prompt(st.session_state.relationship_type)},
                {
                    "role": "user",
                    "content": (
                        f"{user_context_text}\n\nFull confirmed conversation so far:\n{confirmed_text}"
                    ),
                },
            ]

            with st.spinner("Great! Now I'm thinking about your potential deal breakers..."):
                ai_response = call_llm(db_messages, max_tokens=3000)

            if ai_response:
                st.session_state.recovery_pending = trust_recovery.ai_signals_recovery(ai_response)
                ai_response = TrustRecoverySystem.strip_recovery_tag(ai_response)
                st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
                st.session_state.messages.append({"role": "assistant", "content": ai_response})

            st.session_state.awaiting_deal_breakers = True

    elif st.session_state.awaiting_deal_breakers:
        # User just reacted to deal breakers — check if confirming or giving feedback
        user_is_confirming = user_signals_confirmation(user_input)

        if not user_is_confirming:
            # User gave feedback on deal breakers — re-generate with their correction
            confirmed_text = _get_proposition_conversation_text()
            db_messages = [
                {"role": "system", "content": get_deal_breakers_system_prompt(st.session_state.relationship_type)},
                {"role": "user", "content": f"{user_context_text}\n\nFull confirmed conversation so far:\n{confirmed_text}"},
                {"role": "assistant", "content": st.session_state.stage_messages[-2]["content"] if len(st.session_state.stage_messages) >= 2 else ""},
                {
                    "role": "user",
                    "content": (
                        f"The user wants to adjust the deal breakers: {user_input}\n\n"
                        "CRITICAL: Distinguish between ADDING and REPLACING.\n"
                        "- If the user asks to ADD a deal breaker, keep ALL existing deal breakers exactly as they are "
                        "and append the new one.\n"
                        "- If the user asks to REMOVE a deal breaker, remove only that one and keep the rest unchanged.\n"
                        "- If the user asks to CHANGE a deal breaker, modify only that one and keep the rest word-for-word.\n"
                        "Never drop an existing deal breaker to make room for a new one.\n\n"
                        "Re-present the full list of deal breakers with their adjustments applied, then end with the same kind "
                        "of closing: ask them to confirm if these work, or to change anything."
                    ),
                },
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
    else:
        # Safety fallback: trait_map_confirmed=True but awaiting_deal_breakers=False
        # This state should not be reachable with the new ranking flow, but guard it anyway
        st.session_state.awaiting_looking_for_ranking = True


def tension_clarification_turn_user_message(clarification_num: int) -> str:
    """Inject before each tension LLM call so the model knows which of 3 clarifications it is."""
    if clarification_num == 3:
        return (
            "This is clarification **3 of 3** — ask exactly one final question. "
            "Do NOT add any disclaimers about this being the last question or whether they need to answer."
        )
    return f"This is clarification **{clarification_num} of 3** — ask exactly one question."


def start_tension_stage():
    system_msg = {"role": "system", "content": TENSION_SYSTEM_PROMPT}
    user_context = {"role": "user", "content": f"Here are the user's inferred priorities: {json.dumps(st.session_state.proposition_data)}"}
    turn_hint = {"role": "user", "content": tension_clarification_turn_user_message(1)}
    st.session_state.stage_messages = [system_msg, user_context, turn_hint]

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

    # round_count is now 2 or 3 = which clarification the assistant is about to produce
    st.session_state.stage_messages.append(
        {"role": "user", "content": tension_clarification_turn_user_message(st.session_state.round_count)}
    )

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
    prompt = "Does this feel right to you, or is something off?\n\n(Type **done** or **yes** when you're happy with it.)"
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

    # Treat confirmation phrases like "yes" / "looks good" / "that's right" as done — same as "done".
    # Keep exit / quit / finished for users who type those explicitly.
    t = (user_input or "").lower().strip()
    if user_signals_confirmation(user_input) or t in {"exit", "quit", "finished"}:
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
# -----------------------------------------------------------------------
# BIG 6 — Constants, HTML template, helpers
# -----------------------------------------------------------------------

CARD_COLORS = ["#4a7fa5", "#3d8b6e", "#7a5195", "#c05c7e", "#d4875a", "#557a95"]

_BIG6_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:transparent;padding:8px 6px 14px}}
.hd{{font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;
  letter-spacing:.7px;margin-bottom:4px}}
.sub{{font-size:11px;color:#9ca3af;margin-bottom:10px;line-height:1.4}}
.chips{{display:flex;flex-wrap:wrap;gap:8px}}
.chip{{background:#fff;border:1px solid #e5e7eb;border-left:3px solid #ccc;
  border-radius:9px;padding:7px 26px 7px 10px;position:relative;min-width:0}}
.chip-dim{{font-size:10px;color:#9ca3af;text-transform:uppercase;
  letter-spacing:.5px;margin-bottom:2px}}
.chip-label{{font-size:12px;font-weight:600;color:#111827;line-height:1.2}}
.chip-bar{{width:100%;height:2px;background:#f3f4f6;border-radius:1px;margin-top:5px}}
.chip-fill{{height:100%;border-radius:1px;transition:width .4s ease}}
.chip-x{{position:absolute;top:6px;right:5px;width:16px;height:16px;
  background:rgba(0,0,0,.06);border:none;border-radius:50%;cursor:pointer;
  font-size:11px;color:#9ca3af;display:flex;align-items:center;justify-content:center;
  padding:0;line-height:1;transition:background .1s,color .1s}}
.chip-x:hover{{background:rgba(0,0,0,.14);color:#374151}}
.empty{{font-size:12px;color:#d1d5db;font-style:italic}}
</style></head><body>
<div class="hd">About You</div>
<div class="sub">What the model has learned — tap \u00d7 to remove anything that\u2019s off.</div>
<div id="board"></div>
<script>
const COLORS = {colors};
const CHIPS = {chips};
const RK = 'removed_chips_v1';
let removed = loadRemoved();

function loadRemoved() {{
  try {{ return new Set(JSON.parse(sessionStorage.getItem(RK) || '[]')); }}
  catch(_) {{ return new Set(); }}
}}
function saveRemoved() {{ sessionStorage.setItem(RK, JSON.stringify([...removed])); }}
function removeChip(id) {{ removed.add(id); saveRemoved(); render(); }}
function mk(tag, cls) {{ const e = document.createElement(tag); if (cls) e.className = cls; return e; }}

function render() {{
  const board = document.getElementById('board');
  board.innerHTML = '';
  const visible = CHIPS.filter(c => !removed.has(c.id));
  if (!visible.length) {{
    const e = mk('div', 'empty');
    e.textContent = 'Building a picture of you\u2026';
    board.append(e); return;
  }}
  const wrap = mk('div', 'chips');
  visible.forEach((c, i) => {{
    const color = COLORS[i % COLORS.length];
    const chip = mk('div', 'chip');
    chip.style.borderLeftColor = color;
    const dim = mk('div', 'chip-dim'); dim.textContent = c.dimension;
    const lbl = mk('div', 'chip-label'); lbl.textContent = c.label;
    chip.append(dim, lbl);
    if (c.confidence > 0) {{
      const bar = mk('div', 'chip-bar');
      const fill = mk('div', 'chip-fill');
      fill.style.background = color;
      fill.style.width = Math.round(c.confidence * 100) + '%';
      bar.append(fill); chip.append(bar);
    }}
    const xb = mk('button', 'chip-x'); xb.innerHTML = '&times;';
    xb.addEventListener('click', () => removeChip(c.id));
    chip.append(xb);
    wrap.append(chip);
  }});
  board.append(wrap);
}}
render();
</script></body></html>"""



def render_big6_panel():
    chips = rank_live_traits()
    html = _BIG6_HTML.format(
        colors=json.dumps(CARD_COLORS),
        chips=json.dumps(chips),
    )
    components.html(html, height=300, scrolling=False)


# -----------------------------------------------------------------------
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
            background-color: #f9fafb !important;
            border-right: 1px solid #e5e7eb !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            background-color: #f9fafb !important;
        }

        /* Target the actual inner container Streamlit renders */
        [data-testid="stChatInput"] > div {
            border-radius: 20px !important;
            border: 1.5px solid #e5e7eb !important;
            background-color: #ffffff !important;
            box-shadow: 0 1px 6px rgba(0,0,0,0.06) !important;
            overflow: hidden !important;
        }
        [data-testid="stChatInput"] > div:focus-within {
            border-color: #d1d5db !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.10) !important;
        }

        /* Remove default border from the outer wrapper */
        [data-testid="stChatInput"] {
            border: none !important;
            background: transparent !important;
            border-radius: 9999px !important;
            
        }
        

        /* Textarea itself */
        [data-testid="stChatInput"] textarea {
            background-color: #ffffff !important;
            color: #111827 !important;
            border: none !important;
            box-shadow: none !important;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #9ca3af !important;
        }

        /* Rose pink pill send button */
        [data-testid="stChatInputSubmitButton"] button {
            background-color: #f43f5e !important;
            border-radius: 9999px !important;
            border: none !important;
            width: 36px !important;
            height: 36px !important;
            transition: background-color 0.15s ease !important;
        }
        [data-testid="stChatInputSubmitButton"] button:hover {
            background-color: #e11d48 !important;
        }

                /* Make sure all inner wrappers are white — but NOT the button */
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] > div > div {
            background-color: #ffffff !important;
        }

        /* Only target non-button children */
        [data-testid="stChatInput"] *:not(button):not(svg):not(path) {
            background-color: #ffffff !important;
        }

        /* Restore button color explicitly after the wildcard */
        [data-testid="stChatInputSubmitButton"] button {
            background-color: #f43f5e !important;
            border-radius: 9999px !important;
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
            width: 36px !important;
            height: 36px !important;
        }
        [data-testid="stChatInputSubmitButton"] button:hover {
            background-color: #e11d48 !important;
        }
        [data-testid="stChatInputSubmitButton"] button:focus {
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
        }
        
        

        /* Big 6 right column — fixed full-height right sidebar */
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
            position: fixed !important;
            right: 0 !important;
            top: 3.75rem !important;
            height: calc(100vh - 3.75rem) !important;
            width: 280px !important;
            overflow-y: auto !important;
            border-left: 1px solid #e5e7eb !important;
            background-color: #f9fafb !important;
            padding: 0 !important;
            z-index: 100 !important;
        }
        /* Prevent main content from sliding under the right sidebar */
        section[data-testid="stMain"] > div:first-child {
            padding-right: 296px !important;
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

    # Sidebar with progress (+ substeps on the active stage when applicable)
    with st.sidebar:
        st.title("Your journey")
        render_sidebar_timeline()
        st.divider()
        if st.sidebar.button("Start Over"):
            st.session_state.clear()
            st.rerun()

    # Main content — two-column layout: chat left, Big 6 right
    st.title("AI Relationship Profile Builder")
    left_col, right_col = st.columns([4, 1.5])

    with right_col:
        render_big6_panel()

    with left_col:
        render_chat_content()


def render_priority_ranking():
    """Inline chat widget: user sees and reorders match priorities before clarification starts."""
    priorities = st.session_state.match_priorities
    n = len(priorities)

    st.markdown(
        "Based on everything you've shared, here's what I think matters most to you "
        "in this connection. **Use ↑ ↓ to put them in your order** — #1 is what you "
        "value most. This ranking shapes your profile directly."
    )
    st.markdown("")

    for i, p in enumerate(priorities):
        col_num, col_card, col_up, col_dn = st.columns([0.08, 2.9, 0.2, 0.2])
        with col_num:
            st.markdown(
                f"<p style='margin:0;padding:14px 0 0;font-size:14px;"
                f"font-weight:700;color:#9ca3af;text-align:right'>{i + 1}</p>",
                unsafe_allow_html=True,
            )
        with col_card:
            color = CARD_COLORS[i % len(CARD_COLORS)]
            reason = p.get("reason", "")
            reason_html = (
                f"<div style='font-size:11px;color:#6b7280;margin-top:5px;"
                f"line-height:1.45;padding:0 2px'>{reason}</div>"
                if reason else ""
            )
            st.markdown(
                f"<div style='background:{color};color:#fff;border-radius:8px;"
                f"padding:9px 14px;font-weight:600;font-size:13px;"
                f"text-shadow:0 1px 2px rgba(0,0,0,.2)'>{p['label']}</div>"
                + reason_html,
                unsafe_allow_html=True,
            )
        with col_up:
            if i > 0 and st.button("↑", key=f"prio_up_{i}"):
                priorities[i], priorities[i - 1] = priorities[i - 1], priorities[i]
                st.session_state.match_priorities = priorities
                st.rerun()
        with col_dn:
            if i < n - 1 and st.button("↓", key=f"prio_dn_{i}"):
                priorities[i], priorities[i + 1] = priorities[i + 1], priorities[i]
                st.session_state.match_priorities = priorities
                st.rerun()

    st.markdown("")
    if st.button("This is my order — let's continue", type="primary"):
        st.session_state.awaiting_priority_ranking = False
        advance_stage()
        start_proposition_stage(priorities_confirmed=True)
        st.rerun()


def render_looking_for_ranking():
    """Inline chat widget: user reorders what they are looking for before deal breakers."""
    items = st.session_state.what_looking_for
    n = len(items)

    st.markdown(
        "Here's what I think you're looking for in this connection. "
        "**Use ↑ ↓ to put them in your order** — #1 matters most to you. "
        "This shapes how your profile is written."
    )
    st.markdown("")

    for i, p in enumerate(items):
        col_num, col_card, col_up, col_dn = st.columns([0.08, 2.9, 0.2, 0.2])
        with col_num:
            st.markdown(
                f"<p style='margin:0;padding:14px 0 0;font-size:14px;"
                f"font-weight:700;color:#9ca3af;text-align:right'>{i + 1}</p>",
                unsafe_allow_html=True,
            )
        with col_card:
            color = CARD_COLORS[i % len(CARD_COLORS)]
            reason = p.get("reason", "")
            reason_html = (
                f"<div style='font-size:11px;color:#6b7280;margin-top:5px;"
                f"line-height:1.45;padding:0 2px'>{reason}</div>"
                if reason else ""
            )
            st.markdown(
                f"<div style='background:{color};color:#fff;border-radius:8px;"
                f"padding:9px 14px;font-weight:600;font-size:13px;"
                f"text-shadow:0 1px 2px rgba(0,0,0,.2)'>{p['label']}</div>"
                + reason_html,
                unsafe_allow_html=True,
            )
        with col_up:
            if i > 0 and st.button("↑", key=f"lf_up_{i}"):
                items[i], items[i - 1] = items[i - 1], items[i]
                st.session_state.what_looking_for = items
                st.rerun()
        with col_dn:
            if i < n - 1 and st.button("↓", key=f"lf_dn_{i}"):
                items[i], items[i + 1] = items[i + 1], items[i]
                st.session_state.what_looking_for = items
                st.rerun()

    st.markdown("")
    if st.button("Looks right — let's continue", type="primary", key="confirm_looking_for"):
        st.session_state.awaiting_looking_for_ranking = False

        with st.spinner("Identifying your deal breakers..."):
            _generate_deal_breaker_items(
                st.session_state.user_portrait,
                st.session_state.relationship_type,
            )
        st.session_state.awaiting_deal_breaker_ranking = True
        st.rerun()


def render_deal_breaker_ranking():
    """Inline chat widget: user reviews and reorders deal breakers before moving to clarification."""
    items = st.session_state.deal_breaker_items
    n = len(items)

    st.markdown(
        "Based on everything you've shared, here are the things that would be genuine deal breakers for you. "
        "**Use ↑ ↓ to reorder** — #1 is your most non-negotiable."
    )
    st.markdown("")

    for i, p in enumerate(items):
        col_num, col_card, col_up, col_dn = st.columns([0.08, 2.9, 0.2, 0.2])
        with col_num:
            st.markdown(
                f"<p style='margin:0;padding:14px 0 0;font-size:14px;"
                f"font-weight:700;color:#9ca3af;text-align:right'>{i + 1}</p>",
                unsafe_allow_html=True,
            )
        with col_card:
            color = CARD_COLORS[i % len(CARD_COLORS)]
            reason = p.get("reason", "")
            reason_html = (
                f"<div style='font-size:11px;color:#6b7280;margin-top:5px;"
                f"line-height:1.45;padding:0 2px'>{reason}</div>"
                if reason else ""
            )
            st.markdown(
                f"<div style='background:{color};color:#fff;border-radius:8px;"
                f"padding:9px 14px;font-weight:600;font-size:13px;"
                f"text-shadow:0 1px 2px rgba(0,0,0,.2)'>{p['label']}</div>"
                + reason_html,
                unsafe_allow_html=True,
            )
        with col_up:
            if i > 0 and st.button("↑", key=f"db_up_{i}"):
                items[i], items[i - 1] = items[i - 1], items[i]
                st.session_state.deal_breaker_items = items
                st.rerun()
        with col_dn:
            if i < n - 1 and st.button("↓", key=f"db_dn_{i}"):
                items[i], items[i + 1] = items[i + 1], items[i]
                st.session_state.deal_breaker_items = items
                st.rerun()

    st.markdown("")
    if st.button("These are right — let's build my profile", type="primary", key="confirm_deal_breakers"):
        st.session_state.awaiting_deal_breaker_ranking = False
        st.session_state.deal_breakers_confirmed = True

        # Build proposition_data directly from session state (no extra LLM call needed)
        portrait = st.session_state.user_portrait
        trait_parts = []
        for field in ("personality_traits", "values", "social_energy", "thinking_style", "decision_making"):
            val = portrait.get(field, "")
            if isinstance(val, list):
                trait_parts.extend(val[:3])
            elif val:
                trait_parts.append(val)
        trait_summary = ". ".join(trait_parts[:6])

        st.session_state.proposition_data = {
            "relationship_type": st.session_state.relationship_type,
            "user_trait_summary": trait_summary,
            "selected_dimensions": [
                {
                    "category": "Match Priorities",
                    "ranked_items": [
                        {"item": p["label"], "reasoning": p.get("reason", "")}
                        for p in st.session_state.match_priorities
                    ],
                },
                {
                    "category": "What You're Looking For",
                    "ranked_items": [
                        {"item": p["label"], "reasoning": p.get("reason", "")}
                        for p in st.session_state.what_looking_for
                    ],
                },
            ],
            "deal_breakers": [p["label"] for p in st.session_state.deal_breaker_items],
        }

        confirmed_msg = "Great — I have everything I need. Let's build your profile."
        st.session_state.messages.append({"role": "assistant", "content": confirmed_msg})

        advance_stage()
        start_tension_stage()
        st.rerun()


def render_chat_content():
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
                response = intro_acknowledgment_message(user_input)
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
        if st.session_state.get("awaiting_priority_ranking", False):
            render_priority_ranking()
        elif st.session_state.get("awaiting_summary_confirmation", False):
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
                    # Progressively surface traits in Big 6 as answers come in (every 2nd turn)
                    user_turn_count = sum(
                        1 for m in st.session_state.stage_messages if m["role"] == "user"
                    )
                    if user_turn_count % 2 == 0 or user_turn_count <= 2:
                        new_live = extract_live_traits(st.session_state.stage_messages)
                        if new_live:
                            _merge_live_traits(new_live)
                    if check_stage_completion("about_you", ai_response):
                        st.session_state.messages.append({"role": "assistant", "content": "Does this capture you well? Feel free to correct anything or add something important I missed."})
                        st.session_state.awaiting_summary_confirmation = True
                st.rerun()

    elif st.session_state.stage == "proposition":
        if st.session_state.get("awaiting_looking_for_ranking", False):
            render_looking_for_ranking()
        elif st.session_state.get("awaiting_deal_breaker_ranking", False):
            render_deal_breaker_ranking()
        elif user_input := st.chat_input("Type 'yes' to confirm, or suggest changes..."):
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

                has_user_ideas = user_input.lower() not in {"surprise me", "surprise", "no", "nope", ""}
                if has_user_ideas:
                    messages.append({"role": "user", "content": f"The user wants to include these ideas: {user_input}"})

                name_instruction = (
                    "Start with one line only: Meet [FirstName], a [Age] year old [Gender]. "
                    "You MUST use the name the user provided above — do NOT invent a different name. "
                    "Invent a plausible age."
                ) if has_user_ideas else (
                    "Start with one line only: Meet [FirstName], a [Age] year old [Gender]. (invent a plausible first name and age). "
                )

                # Build comprehensive context from all confirmed rankings + about_you traits
                priorities_block = "\n".join(
                    f"{i+1}. {p['label']}" + (f" — {p.get('reason','')}" if p.get("reason") else "")
                    for i, p in enumerate(st.session_state.match_priorities)
                ) or "(not set)"
                looking_for_block = "\n".join(
                    f"{i+1}. {p['label']}" + (f" — {p.get('reason','')}" if p.get("reason") else "")
                    for i, p in enumerate(st.session_state.what_looking_for)
                ) or "(not set)"
                deal_breakers_block = "\n".join(
                    f"- {p['label']}" + (f" — {p.get('reason','')}" if p.get("reason") else "")
                    for p in st.session_state.deal_breaker_items
                ) or "(not set)"
                live_traits_block = "\n".join(
                    f"- {v['label']}: {int(v['confidence'] * 100)}% confidence"
                    for v in st.session_state.get("live_traits", {}).values()
                ) or "(not captured)"

                full_context = (
                    f"USER PORTRAIT (who they are):\n{json.dumps(st.session_state.user_portrait, indent=2)}\n\n"
                    f"ABOUT YOU — Personality dimensions observed during conversation:\n{live_traits_block}\n\n"
                    f"MATCH PRIORITIES (what they value most in a partner, ranked #1 = most important):\n{priorities_block}\n\n"
                    f"WHAT THEY'RE LOOKING FOR (specific partner qualities, ranked #1 = most important):\n{looking_for_block}\n\n"
                    f"DEAL BREAKERS (must NOT appear anywhere in the profile):\n{deal_breakers_block}"
                )

                messages.append({
                    "role": "user",
                    "content": (
                        f"Generate a complete profile using all of the following confirmed information:\n\n"
                        f"{full_context}\n\n"
                        f"{name_instruction}"
                        "Then a blank line, then the section headers and body. "
                        f"Select appropriate sections for this relationship type. "
                        f"The top-ranked match priorities and looking-for qualities should come through most prominently. "
                        f"The user's personality traits from 'About You' should be reflected in how the ideal person is described. "
                        f"End with a 'Why This Person Fits You' section that ties the profile back to the user's portrait. "
                        f"The profile must EXCLUDE all deal breakers entirely."
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
