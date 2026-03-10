# Simple symptom detection using keyword matching
COMMON_SYMPTOMS = [
    "fever", "vomiting", "nausea", "headache", "cough",
    "sore throat", "diarrhea", "stomach ache", "fatigue"
]

def detect_symptoms(user_message):
    """
    Returns only actual symptoms mentioned in user input.
    """
    message_words = user_message.lower().split()
    detected = [sym for sym in COMMON_SYMPTOMS if sym in message_words]
    return detected
