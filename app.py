import re
import json
import streamlit as st
import streamlit.components.v1 as components

# -------------------------------
# CONFIG
# -------------------------------
MODEL_PATH = "llama-3.1-8b.gguf"
TEST_MODE = False  # Set to False to use the real model

# 25 mock LLM responses covering the full happy path + trust recovery.
# Ordered to match the call sequence:
#   R0-R4:  about_you questions (4 Qs + SUMMARY)
#   R5:     extract_user_portrait JSON (hidden)
#   R6-R7:  proposition trait map (first pass + revised after user tweak)
#   R8:     deal breakers
#   R9:     extract_proposition JSON (hidden)
#   R10-R12: tension clarifying questions
#   R13:    tension wrap-up
#   R14:    profile generation (Option A)
#   R14b:   profile generation (Option B — variant)
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
    # R14b — profile generation (variant B — different creative angle)
    "**Meet Lina, a 28 year old woman.**\n\n### Personality & Core Traits\nLina is perceptive in a way that catches you off guard — she'll notice the thing you didn't say and ask about it three days later. She's calm without being passive, warm without being overwhelming. She has a quiet confidence and a habit of collecting odd hobbies (currently: bookbinding and fermenting hot sauce).\n\n### Communication Style\nShe speaks carefully but not cautiously — she just doesn't waste words. She's the person who sends one perfect text instead of twelve. In person, she listens first, then responds with something that shows she was actually paying attention.\n\n### Emotional Style & Love Languages\nLina's love language is acts of care — she shows up with soup when you're sick, fixes the shelf you mentioned was broken, remembers your mother's birthday. She's emotionally steady and expressive in action more than words, though when she does say something vulnerable, she means every syllable.\n\n### A Typical Day in Her Life\nMornings start early with tea and journaling. She works as a UX researcher, spending her days listening to people talk about how they use things. Evenings are for cooking (always from scratch, always with music on), a long phone call with her sister, or reading on the couch with her cat.\n\n### Conflict Style\nLina needs a beat before she engages in conflict — not to avoid it, but to make sure she's responding to what's actually happening. She's direct when she's ready, and she doesn't hold grudges. She'd rather resolve it fully once than let it simmer.\n\n### Backstory\nLina grew up in a bilingual household, the youngest of two. Her parents ran a small bakery together, which taught her that love often looks like working side by side in comfortable silence. She moved cities once for a fresh start and built a close-knit circle from scratch.\n\n### Why This Person Fits You\nLina mirrors your depth and intentionality but brings a different texture — more action-oriented warmth, a groundedness that expresses itself through doing rather than saying. Where you're reflective, she's responsive. Together, the balance would feel effortless.",
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

# First profile for TEST_MODE dual-profile step — fixed full profile (do not use call_llm here:
# sequential index is often not exactly R14, which shows R13 tension wrap-up in Profile A by mistake).
_PROFILE_MOCK_VARIANT_A = (
    "**Meet Soren, a 31 year old man.**\n\n### Personality & Core Traits\n"
    "Soren is the kind of person who actually listens — not politely, but with the kind of attention that makes "
    "you feel like the most important person in the room. He's unhurried and deliberate, someone who thinks "
    "before he speaks and means what he says. He has a dry, quiet sense of humor that surfaces when you least "
   "expect it. He's not trying to impress anyone; he's just genuinely himself.\n\n"
    "### Communication Style\n"
    "Soren doesn't fill silence for the sake of it. When he does speak, it tends to be worth hearing — considered, "
    "specific, and warm. He's honest without being blunt.\n\n"
    "### A Typical Interaction\n"
    "A weekend might be a slow breakfast, a walk somewhere green, and a long conversation that wanders without "
    "needing a point. Weeknights are low-key — cooking together or parallel reading in the same room.\n\n"
    "### Emotional Style\n"
    "His primary orientation is quality time — undistracted, unhurried presence. He notices the small things and "
    "shows up with consistency.\n\n"
    "### Physical Description\n"
    "Tallish, unhurried in how he moves; soft-spoken; tends toward practical, comfortable clothes and an open, "
    "attentive posture.\n\n"
    "### Why This Person Fits You\n"
    "Soren offers emotional depth within a stable, consistent presence. His pace matches yours — unhurried and "
    "intentional."
)

# Second profile for TEST_MODE — same Meet line and section titles/order as A (flexible 5–8 list).
_PROFILE_MOCK_VARIANT_B = (
    "**Meet Soren, a 31 year old man.**\n\n### Personality & Core Traits\n"
    "Soren is quietly intense in the best way — he cares deeply but wears it lightly. He's observant without "
    "being judgmental, and he asks the one question that unlocks the real conversation.\n\n"
    "### Communication Style\n"
    "He texts in full sentences, uses voice memos when tone matters, and would rather have a hard conversation "
    "at the wrong time than a polite one that goes nowhere.\n\n"
    "### A Typical Interaction\n"
    "Time together feels unhurried: a shared errand turns into the best talk of the week; he remembers what "
    "you said last time and follows up.\n\n"
    "### Emotional Style\n"
    "Words of affirmation and quality time matter — he needs to feel seen, with unhurried time together without "
    "an agenda.\n\n"
    "### Physical Description\n"
    "Medium height, warm eyes, expressive hands; dresses in a way that's put-together but never performative.\n\n"
    "### Why This Person Fits You\n"
    "Soren matches the steadiness and depth you described: small-circle energy, meaning over noise, and someone who "
    "challenges you gently from a stable base."
)

# -------------------------------
# STAGE DEFINITIONS
# -------------------------------
STAGES = ["intro", "about_you", "tension", "proposition", "profile", "refinement", "complete"]
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
            color = "#fecdd3" if stage_idx <= current_idx else "transparent"
            st.markdown(f'<div style="width:2px;height:16px;background:{color};margin-left:9px;"></div>', unsafe_allow_html=True)

        # Dot styling
        if is_done:
            bg, border, text, lbl = "#f43f5e", "#f43f5e", "✓", "color:var(--text-color);"
        elif is_active:
            bg, border, text, lbl = "transparent", "#f43f5e", "", "color:var(--text-color);font-weight:700;"
        else:
            bg, border, text, lbl = "transparent", "rgba(128,128,128,0.4)", "", "color:var(--text-color);"

        st.markdown(f'<div style="display:flex;align-items:center;gap:12px;"><div style="width:20px;height:20px;border-radius:50%;background:{bg};border:2px solid {border};display:flex;align-items:center;justify-content:center;font-size:10px;color:white;font-weight:700;flex-shrink:0;">{text}</div><span style="font-size:14px;{lbl}">{STAGE_LABELS[stage_key]}</span></div>', unsafe_allow_html=True)

        if is_active:
            _render_substeps_inline(stage_key)


def _render_substeps_inline(stage_key: str):
    ss = st.session_state

    if stage_key == "about_you":
        steps = ["Chat about you", "Confirm summary"]
        active_i = 1 if ss.get("awaiting_summary_confirmation") else 0
    elif stage_key == "proposition":
        steps = ["Your priorities", "Deal breakers"]
        active_i = 1 if ss.get("awaiting_deal_breaker_ranking") else 0
    elif stage_key == "refinement":
        steps = ["Does it feel right?", "Fine-tune (optional)"]
        active_i = 0 if ss.get("awaiting_initial_refinement") else 1
    else:
        return

    for i, label in enumerate(steps):
        if i < active_i:
            txt = "color:var(--text-color);"
            line_color = "#f43f5e"
        elif i == active_i:
            txt = "color:var(--text-color);font-weight:700;"
            line_color = "#fecdd3"  # gray from active onward
        else:
            txt = "color:var(--text-color);"
            line_color = "transparent"

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
            print("[TRUST RECOVERY] 🔴 Detected: error3 (over-scope edit)")
            return "error3"
        if "[trust_recovery:error2]" in lowered:
            print("[TRUST RECOVERY] 🔴 Detected: error2 (false assumption in profile)")
            return "error2"
        if "[trust_recovery:error1]" in lowered:
            print("[TRUST RECOVERY] 🔴 Detected: error1 (AI confusion / clarifying question)")
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
        print(f"[TRUST RECOVERY] 🟡 Executing error1 recovery. User clarification: {user_clarification[:80]}...")
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
            print(f"[TRUST RECOVERY] ✅ error1 recovery complete. Alignment summary appended.")

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
        print("[TRUST RECOVERY] 🟡 Executing error2 recovery (assumption audit)...")
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
            print(f"[TRUST RECOVERY] ✅ error2 complete — {len(corrections)} correction(s) applied to profile.")
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
        print(f"[TRUST RECOVERY] 🟡 Executing error3 recovery (over-scope revert). Requested edit: {original_feedback[:80]}...")
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
            print("[TRUST RECOVERY] ✅ error3 complete — targeted edit applied, profile reverted to frozen state.")
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
    "You have a personality portrait of the user. Analyze it for any internal contradictions, "
    "tensions, or nuances in how they described themselves — things that could affect what kind "
    "of connection would truly work for them.\n\n"
    "For example: they might describe themselves as introverted but also say they love collaborating; "
    "or they might value structure but also say they're open to spontaneity. These are the kinds of "
    "tensions worth exploring — they help clarify what the user actually needs before we build priorities.\n\n"
    "Each time you reply, the user message will say which clarification this is (**1 of 3**, **2 of 3**, or **3 of 3**). "
    "Follow this structure strictly:\n"
    "1. **Acknowledge first** — Start with 1-2 sentences that reflect back what the user just said, "
    "showing you heard and understood their answer. This is essential so the user feels their input matters.\n"
    "2. **Then ask exactly ONE clarifying question** (one `?` only). "
    "Optional: one short transition sentence before the question.\n"
    "- Do **not** stack, number, or bullet multiple questions.\n"
    "- On **3 of 3**: acknowledge their answer, then ask one final clarifying question. "
    "Do NOT add any disclaimers, notes, or remarks about the question budget. Just ask naturally.\n"
    "- If things already feel clear before turn 3, you may write [RESOLVED] on its own line and add a brief warm closing "
    "with **no** question — otherwise keep probing until turn 3.\n\n"
    "Do not resolve tensions for them — let the user think through them.\n\n"
    f"{TRUST_RECOVERY_INSTRUCTIONS}"
)

# If Profile A's ### headings cannot be parsed (<5), Profile B falls back to this list (generic romantic-ish mix).
DEFAULT_PROFILE_SECTION_FALLBACK_ORDERED = [
    "Personality & Core Traits",
    "Communication Style",
    "A Typical Interaction",
    "Emotional Style",
    "Physical Description",
    "Why This Person Fits You",
]

PROFILE_SECTION_SELECTION_TEXT = (
    "SECTION SELECTION — Choose **5–8** sections that make sense for **{relationship_type}** "
    "(romantic vs activity-based vs other — pick from the buckets below). "
    "The two profiles will be compared side by side: **Profile B will mirror Profile A's exact section titles and order**, "
    "so choose one coherent set of sections only.\n\n"
    "Here are your options:\n\n"
    "FOR ANY RELATIONSHIP TYPE:\n"
    "- Personality & Core Traits\n"
    "- Communication Style\n"
    "- A Typical Interaction (what spending time together looks like)\n"
    "- Why This Person Fits You\n\n"
    "FOR ROMANTIC / CLOSE EMOTIONAL RELATIONSHIPS:\n"
    "- Physical Description\n"
    "- Emotional Style\n"
    "- Conflict Style\n"
    "- Backstory\n"
    "- A Typical Day in Their Life\n\n"
    "FOR ACTIVITY / CONTEXT-BASED RELATIONSHIPS:\n"
    "- Play Style or Work Style\n"
    "- Skill Level & Approach\n"
    "- Scheduling & Reliability\n"
    "- Growth Orientation\n\n"
    "RULES: Use **only** titles from the lists above (exact wording, including parentheticals where shown). "
    "Do not invent other top-level section names. "
    "Each section must begin with one Markdown H3 line on its own line: `### ` followed by that title **exactly**. "
    "Prefer including **Why This Person Fits You** as the **last** section when it fits. "
    "If a section feels thin, still include at least one short paragraph.\n\n"
)


def _extract_profile_section_headers(text: str) -> list[str]:
    """Section titles from `### Title` lines, in order (used to align profile B with profile A)."""
    if not text:
        return []
    titles = []
    for line in text.splitlines():
        m = re.match(r"^###\s+(.+)$", line.strip())
        if m:
            titles.append(m.group(1).strip())
    return titles


def _extract_profile_meet_line(text: str) -> str | None:
    """First non-empty line of the profile (the **Meet ...** line) — used so profile B matches profile A."""
    if not text:
        return None
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return None


PROFILE_SYSTEM_PROMPT = (
    "You are collaboratively building a profile of the user's ideal {relationship_type} "
    "based on everything you know about them.\n\n"
    "The user's trait summary: {trait_summary}\n\n"
    "Their confirmed priorities:\n{proposition_json}\n\n"
    f"{PROFILE_SECTION_SELECTION_TEXT}"
    "STRICT RULES:\n"
    "- The **very first line** of the profile (before any section headers) must be exactly this pattern: "
    "**Meet [FirstName], a [Age] year old [Gender].** — invent plausible details. "
    "If the user already gave a name in their ideas, use that first name instead. "
    "Do not use the user's own name or a famous real person's name unless they asked.\n"
    "- After that opening line, add a blank line, then start **every** section with a Markdown H3 heading on "
    "its own line: `### Section Name` (three hashes, space, title). "
    "Do not use bold-only lines (`**Title**`) as section headers — use ### consistently for all sections.\n"
    "- Write in warm, engaging prose under each heading.\n"
    "- Never show JSON to the user.\n"
    "- The top-ranked priorities from the proposition should come through clearly in the profile.\n"
    "- The profile must NOT include any of the user's stated deal breakers.\n"
    "- Keep the profile grounded and specific — this should feel like a real person, not a wish list.\n"
    "- When you include **Why This Person Fits You**, make it the **final** section and tie it to "
    "the user's personality and needs.\n\n"
    f"{TRUST_RECOVERY_INSTRUCTIONS}"
)

PROFILE_VARIANT_SYSTEM_PROMPT = (
    "You are collaboratively building a profile of the user's ideal {relationship_type} "
    "based on everything you know about them.\n\n"
    "IMPORTANT: Another version of this profile is being generated at the same time. "
    "Your job is to take a DIFFERENT creative angle **in the section bodies** — different stories, "
    "examples, career texture, and day-to-day specifics — while staying equally faithful to the "
    "user's confirmed priorities and deal breakers. "
    "The USER message gives the **exact** opening **Meet** line (same as the first profile); "
    "use it verbatim — do **not** change name, age, or gender in that line.\n\n"
    "The user's trait summary: {trait_summary}\n\n"
    "Their confirmed priorities:\n{proposition_json}\n\n"
    f"{PROFILE_SECTION_SELECTION_TEXT}"
    "CRITICAL FOR THIS TURN: The USER message lists the exact `###` section headings and order for this profile. "
    "Use **only** those headings — same titles, same order, no substitutions and no extra sections.\n\n"
    "STRICT RULES:\n"
    "- The **very first line** of the profile (before any section headers) is specified in the USER message "
    "(copy the **Meet** line exactly — same spelling, age, and gender as the first profile).\n"
    "- After that opening line, add a blank line, then start **every** section with a Markdown H3 heading on "
    "its own line: `### Section Name` (three hashes, space, title). "
    "Do not use bold-only lines (`**Title**`) as section headers — use ### consistently for all sections.\n"
    "- Write in warm, engaging prose under each heading.\n"
    "- Never show JSON to the user.\n"
    "- The top-ranked priorities from the proposition should come through clearly in the profile.\n"
    "- The profile must NOT include any of the user's stated deal breakers.\n"
    "- Keep the profile grounded and specific — this should feel like a real person, not a wish list.\n"
    "- If the last section is **Why This Person Fits You**, tie it to the user's personality and needs.\n\n"
    f"{TRUST_RECOVERY_INSTRUCTIONS}"
)

REFINEMENT_SYSTEM_PROMPT = (
    "You are helping the user refine a {relationship_type} profile through natural conversation. "
    "When the user gives feedback, update the profile and reprint it in full — same warm prose format, no JSON. "
    "Keep the opening line **Meet [FirstName], a [Age] year old [Gender].** as the first line. "
    "If the user asks to change the name, age, or gender, update the opening line AND every mention throughout the entire profile to match — "
    "including all pronouns (e.g. they/them → he/him, she/her, etc.) and any gendered language. "
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

def _strip_refinement_notes(text):
    """Remove trailing 'What changed:', '*Updated:', and suggestion notes from a profile."""
    lines = text.split("\n")
    cut_index = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if (stripped.startswith("what changed:") or
            stripped.startswith("*updated:") or
            stripped.startswith("one thing that might") or
            stripped.startswith("no further changes") or
            stripped.startswith("*no further changes")):
            cut_index = i
            break
    # Also strip trailing italic notes like "*Updated: ..."
    while cut_index > 0 and lines[cut_index - 1].strip().startswith("*"):
        cut_index -= 1
    return "\n".join(lines[:cut_index]).rstrip()

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
    # Start-of-sentence variants (no leading space) so they match at position 0
    "change ", "remove ", "replace ", "edit ", "tweak ", "adjust ", "delete ",
    "make ", "update ", "set ", "switch ",
    # Domain-specific terms for profile edits
    "gender", "age", "name",
)


def _llm_classify_confirmation(user_input: str) -> bool:
    """Fallback: ask the LLM whether the user is confirming or giving feedback."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict classifier. The user just replied to a prompt asking them to confirm a profile or suggest changes.\n\n"
                "Reply CONFIRM **only** if the user is purely accepting with zero requested edits.\n"
                "Reply FEEDBACK if the user asks to change, add, remove, or update ANYTHING — "
                "even if they also say something positive (e.g. 'looks great, just change the name').\n\n"
                "Examples that are FEEDBACK:\n"
                "- 'change the name to Alex'\n"
                "- 'make me 25'\n"
                "- 'can you update the age'\n"
                "- 'looks good but make the name different'\n"
                "- 'set the gender to male'\n"
                "- 'I want a different name'\n\n"
                "Examples that are CONFIRM:\n"
                "- 'yes'\n"
                "- 'looks good'\n"
                "- 'perfect, love it'\n"
                "- 'no changes needed'\n\n"
                "When in doubt, reply FEEDBACK. Reply with exactly one word: CONFIRM or FEEDBACK"
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
    # Don't seed priorities here — wait for start_proposition_stage which has
    # the full conversation + tension context for better priority generation.


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


def _seed_match_priorities_from_portrait(portrait, conversation_context="", tension_context=""):
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

    extra_context = ""
    if conversation_context:
        extra_context += f"\n\nCONVERSATION (the user's own words — use specific details they mentioned):\n{conversation_context}"
    if tension_context:
        extra_context += f"\n\nCLARIFICATIONS (important nuances the user explained):\n{tension_context}"

    messages = [
        {"role": "system", "content": (
            f"Based on a user's personality portrait and their conversation, infer 4-6 qualities they most "
            f"value in a partner for a {relationship_type}. "
            "Pay close attention to SPECIFIC things the user mentioned — activities, traits, behaviors, "
            "preferences. If they mentioned specific activities (e.g. badminton, hiking) or specific "
            "qualities (e.g. honest feedback, pushing them to improve), create priorities that reflect "
            "those specifics rather than generic categories.\n\n"
            "Return ONLY a JSON array ordered from most to least important. "
            "Each item: {\"id\": \"snake_case_id\", \"label\": \"short phrase (2-4 words)\", "
            "\"reason\": \"one sentence explaining why this matters given what the user said\"}. "
            "You may use IDs from this list if they fit: shared_values, emotional_depth, "
            "intellectual_connection, space_independence, communication_style, humor_warmth, "
            "life_goals, reliability_trust, spontaneity, physical_chemistry. "
            "But CREATE new specific id+label+reason whenever the user's actual words suggest "
            "something more precise. No other text."
        )},
        {"role": "user", "content": f"Portrait:\n{portrait_text}{extra_context}"}
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


def _generate_deal_breaker_items(portrait, relationship_type, conversation_context="", tension_context=""):
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
    extra_context = ""
    if conversation_context:
        extra_context += f"\n\nCONVERSATION (the user's own words):\n{conversation_context}"
    if tension_context:
        extra_context += f"\n\nCLARIFICATIONS (important nuances):\n{tension_context}"
    messages = [
        {"role": "system", "content": (
            f"Based on the user's personality portrait, confirmed priorities, and their conversation, "
            f"infer 2-4 genuine deal breakers for their {relationship_type}. "
            "These should be behaviours or traits that are non-negotiable given who this person is — "
            "things that would be genuinely incompatible with their personality and needs. "
            "Pay attention to specific things the user mentioned in conversation. "
            "Return ONLY a JSON array ordered from most to least critical. "
            "Each item: {\"id\": \"snake_case_id\", \"label\": \"short phrase (2-4 words)\", "
            "\"reason\": \"one sentence explaining why this would be incompatible\"}. "
            "No other text."
        )},
        {"role": "user", "content": f"Portrait:\n{portrait_text}{priorities_text}{extra_context}"}
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
    if "profile_a" not in st.session_state:
        st.session_state.profile_a = ""
    if "profile_b" not in st.session_state:
        st.session_state.profile_b = ""
    if "awaiting_profile_choice" not in st.session_state:
        st.session_state.awaiting_profile_choice = False
    if "test_llm_idx" not in st.session_state:
        st.session_state.test_llm_idx = 0
    if "awaiting_profile_choice" not in st.session_state:
        st.session_state.awaiting_profile_choice = False
    if "profile_candidate_a" not in st.session_state:
        st.session_state.profile_candidate_a = None
    if "profile_candidate_b" not in st.session_state:
        st.session_state.profile_candidate_b = None

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
        st.session_state.awaiting_profile_choice = False
        st.session_state.profile_candidate_a = None
        st.session_state.profile_candidate_b = None
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
        st.session_state.profile_a = ""
        st.session_state.profile_b = ""
        st.session_state.awaiting_profile_choice = False

# -------------------------------
# STAGE HANDLERS
# -------------------------------
def handle_about_you(user_input):
    # Append user message first so recovery has full context
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.stage_messages.append({"role": "user", "content": user_input})

    if st.session_state.recovery_pending == "error1":
        print("[TRUST RECOVERY] 🟡 error1 pending in about_you — running alignment recovery...")
        trust_recovery.recover_error1(user_input, st.session_state.stage_messages, st.session_state.user_portrait)
        st.session_state.recovery_pending = None
        print("[TRUST RECOVERY] ✅ error1 complete — pipeline will continue on next user turn")
        return  # Do not fire another LLM call this turn; let the user drive next

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
            # Show only the summary portion, stripping any preamble
            summary_idx = ai_response.find("SUMMARY:")
            if summary_idx != -1:
                summary_only = ai_response[summary_idx + len("SUMMARY:"):].strip()
                st.session_state.messages[-1]["content"] = summary_only
                st.session_state.stage_messages[-1]["content"] = summary_only
            st.session_state.messages.append({"role": "assistant", "content": "Does this capture you well? Feel free to correct anything or add something important I missed."})
            st.session_state.awaiting_summary_confirmation = True

def handle_summary_confirmation(user_input):
    st.session_state.messages.append({"role": "user", "content": user_input})

    if not user_signals_confirmation(user_input) and user_input.lower().strip() != "":
        # User gave a correction — regenerate the summary incorporating their feedback
        st.session_state.stage_messages.append({"role": "user", "content": user_input})

        revision_messages = list(st.session_state.stage_messages) + [
            {
                "role": "system",
                "content": (
                    "The user has corrected or added to your summary of who they are. "
                    "You MUST meaningfully integrate their feedback — not just append a sentence at the end. "
                    "Weave their correction naturally throughout the summary where it fits best. "
                    "If they said something is wrong, remove or replace it. "
                    "If they added something new, give it real weight in the summary — "
                    "it matters enough that they brought it up.\n\n"
                    "Output ONLY the revised summary in paragraph form. "
                    "No preamble, no apology, no 'SUMMARY:' prefix, no 'Here is the updated summary'. "
                    "Just the summary text itself."
                ),
            },
            {
                "role": "user",
                "content": f"The user's correction: {user_input}",
            }
        ]

        with st.spinner("Updating summary..."):
            revised = call_llm(revision_messages, max_tokens=3000)

        if revised:
            # Strip any SUMMARY: prefix the LLM might add anyway
            if "SUMMARY:" in revised:
                revised = revised[revised.find("SUMMARY:") + len("SUMMARY:"):].strip()
            st.session_state.stage_messages.append({"role": "assistant", "content": revised})
            st.session_state.messages.append({"role": "assistant", "content": revised})
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Does this capture you well now? Feel free to correct anything else or confirm to continue."
            })
        # Stay in awaiting_summary_confirmation so the user can confirm or correct again
        return

    # User confirmed — extract portrait and advance to tension
    st.session_state.awaiting_summary_confirmation = False
    with st.spinner("Analyzing your response..."):
        st.session_state.user_portrait = extract_user_portrait(st.session_state.stage_messages)
    _backfill_live_traits_from_portrait(st.session_state.user_portrait)
    advance_stage()
    start_tension_stage()

def start_proposition_stage():
    st.session_state.awaiting_deal_breakers = False
    st.session_state.deal_breakers_confirmed = False

    # Build conversation context from chat history
    conversation_context = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    )

    # Extract tension clarifications specifically
    tension_context = ""
    in_tension = False
    tension_exchanges = []
    for msg in st.session_state.messages:
        content = msg.get("content", "")
        if "Let me think through a couple of things" in content:
            in_tension = True
            continue
        if "Thanks for working through that with me" in content:
            in_tension = False
            continue
        if in_tension and msg["role"] in ("user", "assistant"):
            tension_exchanges.append(f"{msg['role'].upper()}: {content}")
    if tension_exchanges:
        tension_context = "\n".join(tension_exchanges)

    # Seed match priorities using portrait + conversation + tension context
    if not st.session_state.match_priorities:
        with st.spinner("Analyzing your priorities..."):
            _seed_match_priorities_from_portrait(
                st.session_state.user_portrait,
                conversation_context=conversation_context,
                tension_context=tension_context,
            )
    st.session_state.awaiting_priority_ranking = True

def _get_proposition_conversation_text():
    """Build the confirmed conversation text from stage_messages for context."""
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in st.session_state.stage_messages
        if m['role'] in ('assistant', 'user')
    )

def handle_proposition(user_input):
    # Append user message first so recovery has full context
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.stage_messages.append({"role": "user", "content": user_input})

    if st.session_state.recovery_pending == "error1":
        print("[TRUST RECOVERY] 🟡 error1 pending in proposition — running alignment recovery...")
        trust_recovery.recover_error1(user_input, st.session_state.stage_messages, st.session_state.user_portrait)
        st.session_state.recovery_pending = None
        print("[TRUST RECOVERY] ✅ error1 complete — pipeline will continue on next user turn")
        return  # Do not fire another LLM call this turn; let the user drive next

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
    user_context = {"role": "user", "content": f"Here is the user's personality portrait: {json.dumps(st.session_state.user_portrait)}"}
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

    # If a trust recovery was signalled on the previous turn, resolve it first
    if st.session_state.recovery_pending == "error1":
        print("[TRUST RECOVERY] 🟡 error1 pending in tension — running alignment recovery...")
        trust_recovery.recover_error1(user_input, st.session_state.stage_messages, st.session_state.user_portrait)
        st.session_state.recovery_pending = None
        print("[TRUST RECOVERY] ✅ error1 complete — pipeline will continue on next user turn")
        return  # Do not fire another LLM call this turn; let the user drive next

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
        start_proposition_stage()
        return

    # round_count is now 2 or 3 = which clarification the assistant is about to produce
    st.session_state.stage_messages.append(
        {"role": "user", "content": tension_clarification_turn_user_message(st.session_state.round_count)}
    )

    with st.spinner("Thinking..."):
        ai_response = call_llm(st.session_state.stage_messages, max_tokens=3000)

    if ai_response:
        # Check for trust recovery signals before displaying
        st.session_state.recovery_pending = trust_recovery.ai_signals_recovery(ai_response)
        ai_response = TrustRecoverySystem.strip_recovery_tag(ai_response)
        st.session_state.stage_messages.append({"role": "assistant", "content": ai_response})
        st.session_state.messages.append({"role": "assistant", "content": ai_response})

        if "resolved" in ai_response.lower():
            advance_stage()
            start_proposition_stage()

def start_profile_stage():
    prompt_msg = "Before I build the profile — do you have anything specific in mind? A name, a vibe, a detail you definitely want included? Or should I surprise you?"
    st.session_state.messages.append({"role": "assistant", "content": prompt_msg})
    st.session_state.awaiting_profile_ideas = True



def apply_profile_choice(which: str):
    """User picked profile A or B; commit to chat history and move to refinement."""
    a = st.session_state.get("profile_candidate_a")
    b = st.session_state.get("profile_candidate_b")
    chosen = a if which == "a" else b
    if not chosen:
        return
    label = "Profile A" if which == "a" else "Profile B"
    st.session_state.profile_text = chosen
    st.session_state.frozen_profile = chosen
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"You chose **{label}**. Here it is — we can refine from here.\n\n{chosen}",
    })
    st.session_state.awaiting_profile_choice = False
    st.session_state.profile_candidate_a = None
    st.session_state.profile_candidate_b = None
    advance_stage()
    start_refinement_stage()

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
            "IMPORTANT: Reprint the COMPLETE updated profile with ALL requested changes applied directly in the text. "
            "Do NOT just describe the changes — actually make them in the reprinted profile. "
            "If the user changed the gender, update the opening line, ALL pronouns, and ALL gendered language throughout. "
            "After the full updated profile, briefly note what changed and suggest "
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
            # Store only the profile itself — strip "What changed:" / suggestions / "*Updated:" notes
            profile_only = _strip_refinement_notes(clean_response)
            st.session_state.profile_text = profile_only
            st.session_state.frozen_profile = profile_only
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
function removeChip(id) {{
  removed.add(id); saveRemoved(); render();
  // Notify Streamlit so the trait is removed from session state
  const url = new URL(window.parent.location);
  url.searchParams.set('remove_trait', id);
  window.parent.location.href = url.toString();
}}
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
    # Estimate height: header (~70px) + each chip (~130px for label, text, bar, gap) + footer padding.
    # Chips stack vertically in the narrow right panel, so count each chip as its own row.
    n = len(chips)
    estimated_height = 70 + max(n, 1) * 130 + 20
    components.html(html, height=estimated_height, scrolling=True)


_AB_PROFILE_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:transparent;padding:0}}

.ab-header{{font-size:15px;color:#374151;margin-bottom:16px;line-height:1.5}}

.ab-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}

.ab-card{{
  border:2px solid #e5e7eb;border-radius:14px;padding:20px;
  cursor:pointer;background:#fff;transition:border-color .15s, box-shadow .15s;
  position:relative;
}}
.ab-card:hover{{border-color:#d1d5db;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.ab-card.selected{{border-color:#f43f5e;box-shadow:0 0 0 3px rgba(244,63,94,.15)}}

.ab-card-label{{
  font-size:13px;font-weight:600;color:#f43f5e;
  margin-bottom:12px;text-transform:uppercase;letter-spacing:.5px
}}

.ab-card-body{{font-size:14px;color:#1f2937;line-height:1.65}}
.ab-card-body h3{{font-size:15px;font-weight:600;margin:14px 0 6px;color:#111827}}
.ab-card-body p{{margin:0 0 10px}}
.ab-card-body strong{{font-weight:600}}

.ab-btn-row{{display:flex;justify-content:center;margin-top:18px}}
.ab-btn{{
  background:#f43f5e;color:#fff;border:none;border-radius:9999px;
  padding:10px 32px;font-size:14px;font-weight:600;cursor:pointer;
  transition:background .15s,transform .08s;
}}
.ab-btn:hover{{background:#e11d48}}
.ab-btn:active{{transform:scale(.97)}}
.ab-btn:disabled{{background:#d1d5db;cursor:default;transform:none}}
</style></head><body>

<p class="ab-header">I generated two different profiles based on your priorities.<br>
Click one to select it, then confirm your choice.</p>

<div class="ab-grid">
  <div class="ab-card" id="card-a" onclick="pick('A')">
    <div class="ab-card-label">Option A</div>
    <div class="ab-card-body" id="body-a"></div>
  </div>
  <div class="ab-card" id="card-b" onclick="pick('B')">
    <div class="ab-card-label">Option B</div>
    <div class="ab-card-body" id="body-b"></div>
  </div>
</div>

<div class="ab-btn-row">
  <button class="ab-btn" id="ab-submit" disabled onclick="submit()">Select a profile</button>
</div>

<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script>
const profileA = {profile_a_json};
const profileB = {profile_b_json};
let selected = null;

document.getElementById('body-a').innerHTML = marked.parse(profileA);
document.getElementById('body-b').innerHTML = marked.parse(profileB);

function pick(choice) {{
  selected = choice;
  document.getElementById('card-a').classList.toggle('selected', choice === 'A');
  document.getElementById('card-b').classList.toggle('selected', choice === 'B');
  const btn = document.getElementById('ab-submit');
  btn.disabled = false;
  btn.textContent = 'Choose Option ' + choice;
}}

function submit() {{
  if (!selected) return;
  // Navigate parent to same page with query param — triggers Streamlit rerun
  const url = new URL(window.parent.location);
  url.searchParams.set('profile_choice', selected);
  window.parent.location.href = url.toString();
}}
</script>
</body></html>"""


def render_profile_choice():
    """Render the AB profile comparison as an HTML component inside the chat."""
    html = _AB_PROFILE_HTML.format(
        profile_a_json=json.dumps(st.session_state.profile_a),
        profile_b_json=json.dumps(st.session_state.profile_b),
    )
    # Estimate height: roughly 1 line per 80 chars, 24px per line, plus padding
    max_len = max(len(st.session_state.profile_a), len(st.session_state.profile_b))
    est_height = max(600, min(1600, int(max_len / 80 * 24) + 200))
    components.html(html, height=est_height, scrolling=True)


def render_autoscroll():
        components.html(
                """
                <script>
                (function () {
                    const parentDoc = window.parent.document;
                    const mainSection = parentDoc.querySelector('section[data-testid="stMain"]');
                    const mainInner = parentDoc.querySelector('section[data-testid="stMain"] > div:first-child');

                    function scrollBottom() {
                        try {
                            if (mainInner) {
                                mainInner.scrollTo({ top: mainInner.scrollHeight, behavior: 'smooth' });
                            }
                            if (mainSection) {
                                mainSection.scrollTo({ top: mainSection.scrollHeight, behavior: 'smooth' });
                            }
                            window.parent.scrollTo({ top: parentDoc.body.scrollHeight, behavior: 'smooth' });
                        } catch (_) {}
                    }

                    setTimeout(scrollBottom, 0);
                    setTimeout(scrollBottom, 120);
                    setTimeout(scrollBottom, 300);

                    const observerTarget = mainInner || mainSection;
                    if (!observerTarget) return;

                    const observer = new MutationObserver(() => {
                        scrollBottom();
                    });
                    observer.observe(observerTarget, { childList: true, subtree: true });
                    setTimeout(() => observer.disconnect(), 1800);
                })();
                </script>
                """,
                height=0,
                scrolling=False,
        )


def render_scroll_to_top():
    """Scroll the main app view to the top (final profile page after chat autoscroll)."""
    components.html(
        """
        <script>
        (function () {
            const parentDoc = window.parent.document;
            const mainSection = parentDoc.querySelector('section[data-testid="stMain"]');
            const mainInner = parentDoc.querySelector('section[data-testid="stMain"] > div:first-child');
            function scrollTopNow() {
                try {
                    if (mainInner) {
                        mainInner.scrollTop = 0;
                        mainInner.scrollTo({ top: 0, left: 0, behavior: 'auto' });
                    }
                    if (mainSection) {
                        mainSection.scrollTop = 0;
                        mainSection.scrollTo({ top: 0, left: 0, behavior: 'auto' });
                    }
                    if (parentDoc.documentElement) parentDoc.documentElement.scrollTop = 0;
                    if (parentDoc.body) parentDoc.body.scrollTop = 0;
                    window.parent.scrollTo(0, 0);
                } catch (_) {}
            }
            scrollTopNow();
            setTimeout(scrollTopNow, 0);
            setTimeout(scrollTopNow, 100);
            setTimeout(scrollTopNow, 300);
        })();
        </script>
        """,
        height=0,
        scrolling=False,
    )


def render_scroll_to_profile_choice():
    """
    Scroll the main column so the view starts at the last assistant message (the
    'two possible profiles' line), then the A/B comparison below — not the bottom of the page.
    """
    components.html(
        """
        <script>
        (function () {
            const parentDoc = window.parent.document;
            const mainSection = parentDoc.querySelector('section[data-testid="stMain"]');
            const mainInner = parentDoc.querySelector('section[data-testid="stMain"] > div:first-child');
            function scrollToLastAssistant() {
                try {
                    const scope = mainInner || mainSection || parentDoc.body;
                    const msgs = scope.querySelectorAll('[data-testid="stChatMessage"]');
                    if (msgs.length === 0) return;
                    let target = null;
                    for (let i = msgs.length - 1; i >= 0; i--) {
                        const txt = (msgs[i].innerText || msgs[i].textContent || '').toLowerCase();
                        if (txt.includes('two possible profiles')) {
                            target = msgs[i];
                            break;
                        }
                    }
                    if (!target) target = msgs[msgs.length - 1];
                    target.scrollIntoView({ block: 'start', behavior: 'smooth' });
                } catch (_) {}
            }
            scrollToLastAssistant();
            setTimeout(scrollToLastAssistant, 0);
            setTimeout(scrollToLastAssistant, 120);
            setTimeout(scrollToLastAssistant, 350);
            setTimeout(scrollToLastAssistant, 700);
        })();
        </script>
        """,
        height=0,
        scrolling=False,
    )


# -----------------------------------------------------------------------



# -----------------------------------------------------------------------
@st.dialog("Welcome to the Relationship Profile Builder", width="large")
def onboarding_modal():
    """Onboarding modal shown on every app load before the user can interact."""

    st.markdown(
        """
        <style>
        /* Ensure modal sits above all other UI (Big 6 panel, etc.) */
        div[data-testid="stModal"] {
            z-index: 9999 !important;
        }
        /* Hide the dialog close (X) button */
        [data-testid="stDialog"] button[aria-label="Close"],
        [data-testid="stDialog"] button[kind="header"] {
            display: none !important;
        }
        /* Prevent closing by clicking the backdrop overlay */
        div[data-testid="stModal"]::before,
        div[data-testid="stModal"] > div:first-child {
            pointer-events: none !important;
        }
        div[data-testid="stModal"] [data-testid="stDialog"] {
            pointer-events: auto !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    st.markdown("#### ✅ What this tool does")
    st.markdown(
        """
- Learns about you through a guided conversation
- Builds a detailed profile of your ideal connection
- Works for any type of relationship — romantic, friendship, study partner, and more
- Assumes the age and gender for your profile. If we make a mistake, feel free to correct it after profile generation!
        """
    )

    st.markdown("")

    st.markdown("#### 🚫 What this tool doesn't do")
    st.markdown(
        """
- Match you with real people
- Store your data after your session
        """
    )

    st.markdown("")

    # Centered "Let's go" button
    _col_l, _col_c, _col_r = st.columns([1, 1, 1])
    with _col_c:
        if st.button("Let's go", key="_onboarding_go", type="primary", use_container_width=True):
            st.session_state._onboarding_dismissed = True
            st.rerun()


def main():
    st.set_page_config(
        page_title="AI Relationship Profile Builder",
        page_icon="💝",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()

    # Show onboarding modal on every fresh load (before any other UI renders)
    if not st.session_state.get("_onboarding_dismissed", False):
        onboarding_modal()
        st.stop()

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
            background-color: var(--secondary-background-color) !important;
            border-right: 1px solid rgba(128,128,128,0.2) !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            background-color: var(--secondary-background-color) !important;
        }

        /* Target the actual inner container Streamlit renders */
        [data-testid="stChatInput"] > div {
            border-radius: 20px !important;
            border: 1.5px solid rgba(128,128,128,0.25) !important;
            background-color: var(--background-color) !important;
            box-shadow: 0 1px 6px rgba(0,0,0,0.06) !important;
            overflow: hidden !important;
        }
        [data-testid="stChatInput"] > div:focus-within {
            border-color: rgba(128,128,128,0.45) !important;
            box-shadow: 0 2px 10px rgba(0,0,0,0.10) !important;
        }

        /* Remove default border from the outer wrapper */
        [data-testid="stChatInput"] {
            border: none !important;
            background: var(--background-color) !important;
            border-radius: 9999px !important;
            position: fixed !important;
            bottom: 16px !important;
            left: 5% !important;
            right: calc(296px + 1%) !important;  /* right panel reserve + gutter */
            width: auto !important;
            z-index: 100 !important;
        }

        /* When sidebar is expanded, keep chat input inside main content area */
        section[data-testid="stSidebar"][aria-expanded="true"] ~ section[data-testid="stMain"] [data-testid="stChatInput"],
        section[data-testid="stSidebar"][aria-expanded="true"] ~ div [data-testid="stChatInput"] {
            left: calc(320px + 16px) !important;
        }
            
        
        

        /* Textarea itself */
        [data-testid="stChatInput"] textarea {
            background-color: var(--background-color) !important;
            color: var(--text-color) !important;
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

        /* Make sure all inner wrappers follow theme — but NOT the button */
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] > div > div {
            background-color: var(--background-color) !important;
        }

        /* Only target non-button children */
        [data-testid="stChatInput"] *:not(button):not(svg):not(path) {
            background-color: var(--background-color) !important;
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
        
        

        /* Ensure onboarding modal sits above the Big 6 fixed panel */
        div[data-testid="stModal"] {
            z-index: 9999 !important;
        }

        /* Big 6 right column — fixed full-height right sidebar */
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
            position: fixed !important;
            right: 0 !important;
            top: 3.75rem !important;
            height: calc(100vh - 3.75rem) !important;
            width: 280px !important;
            overflow-y: auto !important;
            border-left: 1px solid rgba(128,128,128,0.2) !important;
            background-color: var(--secondary-background-color) !important;
            padding: 0 !important;
            z-index: 100 !important;
        }
        /*
         * Nested st.columns (e.g. Profile A vs B) also render stHorizontalBlock + stColumn:last-child.
         * Without this override, the *second* profile column incorrectly receives the Big 6 fixed-panel
         * CSS — it stacks under the real Big 6 panel so only Profile A is visible.
         */
        [data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
            position: static !important;
            right: auto !important;
            top: auto !important;
            height: auto !important;
            width: auto !important;
            min-width: 0 !important;
            overflow: visible !important;
            overflow-y: visible !important;
            border-left: none !important;
            background-color: transparent !important;
            padding: 0 !important;
            z-index: auto !important;
        }
        /* Prevent main content from sliding under the right sidebar */
        section[data-testid="stMain"] > div:first-child {
            padding-right: 296px !important;
        }

        @media (max-width: 768px) {
            [data-testid="stSidebar"] > div:first-child {
                width: 180px !important;
            }
            [data-testid="stChatInput"] {
                left: 16px !important;
                right: 16px !important;
                width: auto !important;
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
            [data-testid="stChatInput"] {
            left: 16px !important;
            right: 16px !important;
            width: auto !important;
            }
        }
        </style>
    """, unsafe_allow_html=True)
    

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
    
    # Add CSS styling for buttons - borderless and evenly spaced
    st.markdown("""
    <style>
    /* Remove borders and style ranking buttons */
    [data-testid="stButton"] button {
        border: none !important;
        background-color: transparent !important;
        padding: 8px 6px !important;
        min-width: 28px !important;
        min-height: 28px !important;
        font-size: 16px !important;
        line-height: 1 !important;
        color: #9ca3af !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="stButton"] button:hover {
        color: #374151 !important;
        background-color: rgba(0, 0, 0, 0.04) !important;
        border-radius: 4px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    priorities = st.session_state.match_priorities
    n = len(priorities)

    # Build a stable id→color map so colors follow the trait, not its rank position
    if "priority_color_map" not in st.session_state:
        st.session_state.priority_color_map = {
            p["id"]: CARD_COLORS[i % len(CARD_COLORS)] for i, p in enumerate(priorities)
        }
    color_map = st.session_state.priority_color_map

    st.markdown(
        "Based on everything you've shared, here's what I think matters most to you "
        "in this connection. **Use ↑ ↓ to reorder** — #1 is what you "
        "value most. **Use ✕ to remove** anything that doesn't fit. This ranking shapes your profile directly."
    )
    st.markdown("")

    for i, p in enumerate(priorities):
        col_num, col_card, col_up, col_dn, col_rm = st.columns([0.06, 2.75, 0.13, 0.13, 0.13])
        with col_num:
            st.markdown(
                f"<p style='margin:0;padding:14px 0 0;font-size:14px;"
                f"font-weight:700;color:#9ca3af;text-align:right'>{i + 1}</p>",
                unsafe_allow_html=True,
            )
        with col_card:
            color = color_map.get(p["id"], CARD_COLORS[i % len(CARD_COLORS)])
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
        with col_rm:
            if n > 1 and st.button("×", key=f"prio_rm_{i}"):
                priorities.pop(i)
                st.session_state.match_priorities = priorities
                st.rerun()

    st.markdown("")
    
    # Style primary buttons to rose pink
    st.markdown("""
    <style>
    button[kind="primary"] {
        background-color: #f43f5e !important;
        color: white !important;
        border: none !important;
        padding: 14px 18px !important;
    }
    button[kind="primary"]:hover {
        background-color: #e11d48 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("This is my order — let's continue", type="primary"):
        st.session_state.awaiting_priority_ranking = False

        # Display confirmed priorities in chat
        priority_summary = "**Your priorities (in order):**\n\n" + "\n".join(
            f"{i+1}. **{p['label']}** — {p.get('reason', '')}" if p.get("reason")
            else f"{i+1}. **{p['label']}**"
            for i, p in enumerate(st.session_state.match_priorities)
        )
        st.session_state.messages.append({"role": "assistant", "content": priority_summary})

        with st.spinner("Moving on to deal breakers..."):
            # Build conversation + tension context for deal breaker generation
            _db_conversation = "\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in st.session_state.messages
                if m["role"] in ("user", "assistant")
            )
            _db_tension = ""
            _in_t = False
            _t_exchanges = []
            for _msg in st.session_state.messages:
                _content = _msg.get("content", "")
                if "Let me think through a couple of things" in _content:
                    _in_t = True
                    continue
                if "Thanks for working through that with me" in _content:
                    _in_t = False
                    continue
                if _in_t and _msg["role"] in ("user", "assistant"):
                    _t_exchanges.append(f"{_msg['role'].upper()}: {_content}")
            if _t_exchanges:
                _db_tension = "\n".join(_t_exchanges)
            _generate_deal_breaker_items(
                st.session_state.user_portrait,
                st.session_state.relationship_type,
                conversation_context=_db_conversation,
                tension_context=_db_tension,
            )
        st.session_state.awaiting_deal_breaker_ranking = True
        st.rerun()


def render_looking_for_ranking():
    """Inline chat widget: user reorders what they are looking for before deal breakers."""
    items = st.session_state.what_looking_for
    n = len(items)

    # Build a stable id→color map so colors follow the trait, not its rank position
    if "looking_for_color_map" not in st.session_state:
        st.session_state.looking_for_color_map = {
            p["id"]: CARD_COLORS[i % len(CARD_COLORS)] for i, p in enumerate(items)
        }
    color_map = st.session_state.looking_for_color_map

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
            color = color_map.get(p["id"], CARD_COLORS[i % len(CARD_COLORS)])
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
    
    # Add CSS styling for buttons - borderless and evenly spaced
    st.markdown("""
    <style>
    /* Remove borders and style ranking buttons */
    [data-testid="stButton"] button {
        border: none !important;
        background-color: transparent !important;
        padding: 8px 6px !important;
        min-width: 28px !important;
        min-height: 28px !important;
        font-size: 16px !important;
        line-height: 1 !important;
        color: #9ca3af !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="stButton"] button:hover {
        color: #374151 !important;
        background-color: rgba(0, 0, 0, 0.04) !important;
        border-radius: 4px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    items = st.session_state.deal_breaker_items
    n = len(items)

    # Build a stable id→color map so colors follow the trait, not its rank position
    if "deal_breaker_color_map" not in st.session_state:
        st.session_state.deal_breaker_color_map = {
            p["id"]: CARD_COLORS[i % len(CARD_COLORS)] for i, p in enumerate(items)
        }
    color_map = st.session_state.deal_breaker_color_map

    st.markdown(
        "Based on everything you've shared, here are the things that would be genuine deal breakers for you. "
        "**Use ↑ ↓ to reorder** — #1 is your most non-negotiable. **Use ✕ to remove** anything that doesn't apply."
    )
    st.markdown("")

    for i, p in enumerate(items):
        col_num, col_card, col_up, col_dn, col_rm = st.columns([0.06, 2.75, 0.13, 0.13, 0.13])
        with col_num:
            st.markdown(
                f"<p style='margin:0;padding:14px 0 0;font-size:14px;"
                f"font-weight:700;color:#9ca3af;text-align:right'>{i + 1}</p>",
                unsafe_allow_html=True,
            )
        with col_card:
            color = color_map.get(p["id"], CARD_COLORS[i % len(CARD_COLORS)])
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
        with col_rm:
            if n > 1 and st.button("✕", key=f"db_rm_{i}"):
                items.pop(i)
                st.session_state.deal_breaker_items = items
                st.rerun()

    st.markdown("")
    
    # Style primary buttons to rose pink
    st.markdown("""
    <style>
    button[kind="primary"] {
        background-color: #f43f5e !important;
        color: white !important;
        border: none !important;
    }
    button[kind="primary"]:hover {
        background-color: #e11d48 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
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
            ],
            "deal_breakers": [p["label"] for p in st.session_state.deal_breaker_items],
        }

        # Display confirmed deal breakers in chat
        db_summary = "**Your deal breakers (in order):**\n\n" + "\n".join(
            f"{i+1}. **{p['label']}** — {p.get('reason', '')}" if p.get("reason")
            else f"{i+1}. **{p['label']}**"
            for i, p in enumerate(st.session_state.deal_breaker_items)
        )
        st.session_state.messages.append({"role": "assistant", "content": db_summary})

        confirmed_msg = "Great — I have everything I need. Let's build your profile."
        st.session_state.messages.append({"role": "assistant", "content": confirmed_msg})

        advance_stage()
        start_profile_stage()
        st.rerun()


def _normalize_profile_section_markdown(text: str) -> str:
    """Turn standalone **Section Title** lines into ### headings so A/B columns render consistently."""
    if not text:
        return text
    lines = text.splitlines()
    out = []
    for line in lines:
        m = re.match(r"^\s*\*\*([^*]+)\*\*\s*$", line)
        if m:
            title = m.group(1).strip()
            if len(title) <= 120 and not title.lower().startswith("meet "):
                out.append(f"### {title}")
                continue
        out.append(line)
    return "\n".join(out)


def render_profile_comparison():
    """Side-by-side profile candidates with continue buttons (profile stage)."""
    st.divider()
    st.subheader("Which profile do you prefer?")
    st.caption(
        "Read both versions, then continue with the one that feels closer to what you're looking for."
    )
    a = st.session_state.get("profile_candidate_a")
    b = st.session_state.get("profile_candidate_b")
    if not a or not b:
        st.warning("Profile options are not available.")
        return
    c1, c2 = st.columns(2, gap="large")
    a_norm = _normalize_profile_section_markdown(a)
    b_norm = _normalize_profile_section_markdown(b)
    with c1:
        st.markdown("##### Profile A")
        st.markdown(a_norm)
        if st.button("Continue with Profile A", key="profile_pick_a", type="primary", use_container_width=True):
            apply_profile_choice("a")
            st.rerun()
    with c2:
        st.markdown("##### Profile B")
        st.markdown(b_norm)
        if st.button("Continue with Profile B", key="profile_pick_b", type="primary", use_container_width=True):
            apply_profile_choice("b")
            st.rerun()


def render_chat_content():
    # "Skip Question" handler (callback-based, set from button on_click below)

    # Trait removal: user clicked × on a chip in the Big 6 panel
    _removed_trait = st.query_params.get("remove_trait")
    if _removed_trait:
        st.query_params.clear()
        if _removed_trait in st.session_state.live_traits:
            del st.session_state.live_traits[_removed_trait]
        st.rerun()


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

    # "Skip Question" link — reserve a placeholder here, render after chat_input
    # so we know whether the user just submitted a response on this render pass.
    _skippable_stages = ("about_you", "tension")
    _last_msg = st.session_state.messages[-1] if st.session_state.messages else None
    _skip_eligible = (
        _last_msg is not None
        and _last_msg["role"] == "assistant"
        and "?" in _last_msg["content"]
        and st.session_state.stage in _skippable_stages
        and not st.session_state.get("awaiting_summary_confirmation")
        and not st.session_state.get("awaiting_priority_ranking")
    )
    _skip_placeholder = st.empty() if _skip_eligible else None

    def _render_skip_button(placeholder):
        """Render the skip button inside the given placeholder."""
        with placeholder.container():
            _skip_col1, _skip_col2 = st.columns([3, 1])
            with _skip_col2:
                st.button(
                    "Skip Question",
                    key="_skip_q_btn",
                    on_click=lambda: st.session_state.__setitem__("_skip_question", True),
                    type="tertiary",
                )
            st.markdown("""
            <style>
            /* Style the Skip Question button as underlined text */
            button[data-testid="stBaseButton-tertiary"][kind="tertiary"] {
                color: #888 !important;
                font-size: 13px !important;
                text-decoration: underline !important;
                background: none !important;
                border: none !important;
                box-shadow: none !important;
                padding: 0 !important;
                margin-top: -12px !important;
                font-family: "Source Sans Pro", sans-serif !important;
            }
            button[data-testid="stBaseButton-tertiary"][kind="tertiary"]:hover {
                color: #555 !important;
            }
            </style>
            """, unsafe_allow_html=True)

    if (
        st.session_state.stage == "profile"
        and st.session_state.get("awaiting_profile_choice")
    ):
        render_profile_comparison()
        render_scroll_to_profile_choice()

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
        if st.session_state.get("awaiting_summary_confirmation", False):
            if user_input := st.chat_input("Your response..."):
                with st.chat_message("user"):
                    st.markdown(user_input)
                handle_summary_confirmation(user_input)
                st.rerun()
        else:
            _skip_active = st.session_state.pop("_skip_question", False)
            user_input = st.chat_input("Your response...")
            if _skip_active:
                user_input = "I'd rather not answer this one — let's move on."
            # Render skip button only if user hasn't submitted on this pass
            if not user_input and _skip_placeholder is not None:
                _render_skip_button(_skip_placeholder)
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.stage_messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)

                # If a trust recovery was signalled on the previous turn, resolve it now
                # and return — do not fire another LLM call on the same turn.
                if st.session_state.get("recovery_pending") == "error1":
                    print("[TRUST RECOVERY] 🟡 error1 pending in about_you (inline) — running alignment recovery...")
                    trust_recovery.recover_error1(user_input, st.session_state.stage_messages, st.session_state.user_portrait)
                    st.session_state.recovery_pending = None
                    print("[TRUST RECOVERY] ✅ error1 complete — pipeline will continue on next user turn")
                    st.rerun()

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
                        # Show only the summary portion, stripping any preamble
                        summary_idx = ai_response.find("SUMMARY:")
                        if summary_idx != -1:
                            summary_only = ai_response[summary_idx + len("SUMMARY:"):].strip()
                            st.session_state.messages[-1]["content"] = summary_only
                            st.session_state.stage_messages[-1]["content"] = summary_only
                        st.session_state.messages.append({"role": "assistant", "content": "Does this capture you well? Feel free to correct anything or add something important I missed."})
                        st.session_state.awaiting_summary_confirmation = True
                st.rerun()

    elif st.session_state.stage == "tension":
        _skip_active = st.session_state.pop("_skip_question", False)
        user_input = st.chat_input("Your response...")
        if _skip_active:
            user_input = "I'm not sure — let's move on to the next one."
        # Render skip button only if user hasn't submitted on this pass
        if not user_input and _skip_placeholder is not None:
            _render_skip_button(_skip_placeholder)
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            handle_tension(user_input)
            st.rerun()

    elif st.session_state.stage == "proposition":
        if st.session_state.get("awaiting_priority_ranking", False):
            render_priority_ranking()
        elif st.session_state.get("awaiting_deal_breaker_ranking", False):
            render_deal_breaker_ranking()

    elif st.session_state.stage == "profile":
        if st.session_state.awaiting_profile_choice:
            # Check if the user already made a choice via the HTML component
            choice = st.query_params.get("profile_choice")
            if choice in ("A", "B"):
                st.query_params.clear()
                chosen = st.session_state.profile_a if choice == "A" else st.session_state.profile_b
                st.session_state.profile_text = chosen
                st.session_state.frozen_profile = chosen
                st.session_state.messages.append({"role": "assistant", "content": chosen})
                st.session_state.awaiting_profile_choice = False
                advance_stage()
                start_refinement_stage()
                st.rerun()
            # Profile comparison is rendered above by render_profile_comparison()

        elif st.session_state.awaiting_profile_ideas:
            if user_input := st.chat_input("Your response..."):
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.awaiting_profile_ideas = False
                with st.chat_message("user"):
                    st.markdown(user_input)

                relationship_type = st.session_state.proposition_data.get("relationship_type", "connection")
                trait_summary = st.session_state.proposition_data.get("user_trait_summary", "")
                proposition_json = json.dumps(st.session_state.proposition_data.get("selected_dimensions", []), indent=2)

                has_user_ideas = user_input.lower() not in {"surprise me", "surprise", "no", "nope", ""}

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
                deal_breakers_block = "\n".join(
                    f"- {p['label']}" + (f" — {p.get('reason','')}" if p.get("reason") else "")
                    for p in st.session_state.deal_breaker_items
                ) or "(not set)"
                live_traits_block = "\n".join(
                    f"- {v['label']}: {int(v['confidence'] * 100)}% confidence"
                    for v in st.session_state.get("live_traits", {}).values()
                ) or "(not captured)"

                # Extract tension clarifications from chat history
                tension_block = ""
                in_tension = False
                tension_exchanges = []
                for msg in st.session_state.messages:
                    content = msg.get("content", "")
                    if "Let me think through a couple of things" in content:
                        in_tension = True
                        continue
                    if "Thanks for working through that with me" in content:
                        in_tension = False
                        continue
                    if in_tension and msg["role"] in ("user", "assistant"):
                        tension_exchanges.append(f"{msg['role'].upper()}: {content}")
                if tension_exchanges:
                    tension_block = f"\n\nCLARIFICATIONS (important nuances the user explained about themselves):\n" + "\n".join(tension_exchanges)

                full_context = (
                    f"USER PORTRAIT (who they are):\n{json.dumps(st.session_state.user_portrait, indent=2)}\n\n"
                    f"ABOUT YOU — Personality dimensions observed during conversation:\n{live_traits_block}\n\n"
                    f"MATCH PRIORITIES (what they value most in a partner, ranked #1 = most important):\n{priorities_block}\n\n"
                    f"DEAL BREAKERS (must NOT appear anywhere in the profile):\n{deal_breakers_block}"
                    f"{tension_block}"
                )

                profile_prompt = PROFILE_SYSTEM_PROMPT.format(
                    relationship_type=relationship_type,
                    trait_summary=trait_summary,
                    proposition_json=proposition_json
                )
                messages = [{"role": "system", "content": profile_prompt}]

                user_ideas_block = ""
                if has_user_ideas:
                    user_ideas_block = (
                        f"\n\nUSER'S SPECIFIC REQUESTS (HIGHEST PRIORITY — you MUST incorporate all of these):\n"
                        f"{user_input}\n"
                        f"These requests override any default choices you would make. "
                        f"If the user specified a name, use that name. If they specified a job, use that job. "
                        f"If they specified a vibe or personality detail, make it central to the profile."
                    )

                messages.append({
                    "role": "user",
                    "content": (
                        f"Generate a complete profile using all of the following confirmed information:\n\n"
                        f"{full_context}"
                        f"{user_ideas_block}\n\n"
                        f"{name_instruction}"
                        "Then a blank line, then **5–8** sections chosen from the SECTION SELECTION options in the system prompt — "
                        "pick the set that best fits this relationship type; use `###` headings with titles **exactly** as in the options. "
                        "Profile B will match your section titles and order, so choose one clear structure. "
                        f"The top-ranked match priorities should come through most prominently. "
                        f"The user's personality traits from 'About You' should be reflected in how the ideal person is described. "
                        f"The profile must EXCLUDE all deal breakers entirely."
                    )
                })

                with st.spinner("Generating profile..."):
                    if TEST_MODE:
                        # Fixed mocks: avoids wrong sequential slot (e.g. R13 tension line in Profile A) and
                        # matches R15+ for refinement — advance index as if one R14 call_llm had run.
                        profile_a = _PROFILE_MOCK_VARIANT_A
                        profile_b = _PROFILE_MOCK_VARIANT_B
                        st.session_state.test_llm_idx = st.session_state.get("test_llm_idx", 0) + 1
                    else:
                        profile_a = call_llm(messages, temperature=0.72, max_tokens=3000)
                        if not profile_a:
                            st.error("Could not generate profiles. Please try again.")
                            st.session_state.awaiting_profile_ideas = True
                            st.rerun()
                        variant_instruction = (
                            "Now generate a second, clearly different alternative profile for the same "
                            "priorities and constraints. Vary the **content under each section** — different "
                            "stories, examples, and specifics — not the opening line or section titles.\n\n"
                        )
                        if has_user_ideas:
                            variant_instruction += (
                                f"IMPORTANT: The user specifically requested these details: {user_input}. "
                                "You MUST keep all of the user's requested details (name, age, job, vibe, etc.) "
                                "exactly as they asked in the section bodies. Only vary the things the user did NOT specify — "
                                "backstory, personality texture, day-to-day details, hobbies, etc. "
                            )
                        meet_line = _extract_profile_meet_line(profile_a)
                        if meet_line:
                            variant_instruction += (
                                "OPENING LINE — Use this **exact** first line (verbatim — same name, age, gender, punctuation, "
                                "including `**` if present), then one blank line before the first `###` section:\n"
                                f"{meet_line}\n\n"
                            )
                        section_order = _extract_profile_section_headers(profile_a)
                        if len(section_order) < 5:
                            section_order = list(DEFAULT_PROFILE_SECTION_FALLBACK_ORDERED)
                        variant_instruction += (
                            "Keep the same deal breakers and ranked priorities. "
                            "Output the full profile only.\n\n"
                            "STRUCTURE — Profile B must use the **exact same** section titles and **exact same order** "
                            "as listed below (mirror Profile A; do not rename or swap sections — e.g. if the list says "
                            "'A Typical Interaction', do not write 'A Typical Day in Their Life' instead, and vice versa). "
                            "Required `###` headings, in order:\n"
                        )
                        for i, title in enumerate(section_order, 1):
                            variant_instruction += f"{i}. ### {title}\n"
                        variant_instruction += (
                            "\nFORMAT: One `### Title` line per section, then paragraphs. "
                            "Do not use bold-only (**Title**) lines as section headers."
                        )
                        variant_system = PROFILE_VARIANT_SYSTEM_PROMPT.format(
                            relationship_type=relationship_type,
                            trait_summary=trait_summary,
                            proposition_json=proposition_json,
                        )
                        messages_alt = [
                            {"role": "system", "content": variant_system},
                            messages[1],
                            {"role": "user", "content": variant_instruction},
                        ]
                        profile_b = call_llm(messages_alt, temperature=0.92, max_tokens=3000)
                        if not profile_b:
                            st.error("Could not generate a second profile. Please try again.")
                            st.session_state.awaiting_profile_ideas = True
                            st.rerun()

                st.session_state.profile_candidate_a = profile_a
                st.session_state.profile_candidate_b = profile_b
                st.session_state.awaiting_profile_choice = True
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        "Here are **two possible profiles** based on what you shared. "
                        "Compare them side by side below, then pick the one you want to refine."
                    ),
                })
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
        render_scroll_to_top()

    if st.session_state.stage != "complete":
        # Don't fight profile A/B scroll: autoscroll jumps to bottom and hides the last AI line + cards
        _profile_choice = (
            st.session_state.stage == "profile"
            and st.session_state.get("awaiting_profile_choice")
        )
        if not _profile_choice:
            render_autoscroll()

if __name__ == "__main__":
    main()