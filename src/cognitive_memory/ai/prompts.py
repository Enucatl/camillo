VALENCE_PROMPT = """You are scoring the long-term memory value of an AI interaction.

Return only a single float between 0.1 and 1.0.

Score high for:
- stable user preferences
- important project decisions
- architecture decisions
- emotional salience
- repeated patterns
- commitments or constraints

Score low for:
- trivial acknowledgements
- temporary chatter
- one-off small talk

Interaction:
{raw_content}
"""
