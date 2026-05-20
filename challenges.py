# challenges.py
"""
This file contains the configuration data for all levels of the Vector Hunt Challenge.
It defines targets, thresholds, minimum correct answers, contexts, and mock Vector Database items.
"""

# =====================================================================
# LEVEL 1: WORD SIMILARITY CHALLENGE CONFIGURATION
# =====================================================================
LEVEL_1_CHALLENGES = [
    {
        "target": "doctor",
        "threshold": 0.60,
        "minimum_correct": 7,
        "instructions": "Enter 10 single words related to the medical profession (e.g., hospital, nurse, medicine, patient, surgeon). Avoid unrelated words."
    },
    {
        "target": "cricket",
        "threshold": 0.60,
        "minimum_correct": 7,
        "instructions": "Enter 10 single words related to the sport of cricket (e.g., bat, ball, wicket, stadium, run, tournament, match). Avoid unrelated terms."
    },
    {
        "target": "school",
        "threshold": 0.60,
        "minimum_correct": 7,
        "instructions": "Enter 10 single words related to education or schools (e.g., student, teacher, classroom, book, study, exam, grade). Avoid unrelated terms."
    },
    {
        "target": "cybersecurity",
        "threshold": 0.60,
        "minimum_correct": 7,
        "instructions": "Enter 10 single words related to digital security (e.g., firewall, hacker, password, encryption, network, phishing, malware). Avoid unrelated terms."
    },
    {
        "target": "database",
        "threshold": 0.60,
        "minimum_correct": 7,
        "instructions": "Enter 10 single words related to database systems (e.g., table, SQL, query, index, storage, schema, keys, backup). Avoid unrelated terms."
    }
]

# =====================================================================
# LEVEL 2: SENTENCE SIMILARITY CHALLENGE CONFIGURATION
# =====================================================================
LEVEL_2_CHALLENGES = [
    {
        "target": "I want to buy a budget phone with a good camera.",
        "threshold": 0.65,
        "minimum_correct": 7,
        "instructions": "Enter 10 different sentences that express the same meaning (e.g., 'affordable smartphone for photography'). Try to change words but keep the intent."
    },
    {
        "target": "I need comfortable shoes for walking every day.",
        "threshold": 0.65,
        "minimum_correct": 7,
        "instructions": "Enter 10 sentences about finding comfy daily walking footwear. Rephrase the query while preserving the semantic meaning."
    },
    {
        "target": "I want to learn Python programming from the beginning.",
        "threshold": 0.65,
        "minimum_correct": 7,
        "instructions": "Enter 10 sentences expressing the desire to start learning Python from scratch as a beginner."
    },
    {
        "target": "I need a laptop for video editing and graphic design.",
        "threshold": 0.65,
        "minimum_correct": 7,
        "instructions": "Enter 10 sentences describing the search for a computer suitable for multimedia editing and creative artwork."
    },
    {
        "target": "The customer wants a refund because the product arrived damaged.",
        "threshold": 0.65,
        "minimum_correct": 7,
        "instructions": "Enter 10 sentences conveying a request for a money refund due to receiving a broken or defective item."
    }
]

# =====================================================================
# LEVEL 3: CONTEXT TRAP CHALLENGE CONFIGURATION
# =====================================================================
LEVEL_3_CHALLENGES = [
    {
        "target": "Apple is a sweet fruit used in juice and salads.",
        "context": "fruit",
        "threshold": 0.65,
        "minimum_correct": 7,
        "instructions": "The word 'Apple' here refers to a sweet fruit. Enter 10 words or phrases related to this meaning (e.g., banana, orange, organic orchard, fresh juice). Unrelated concepts (like iPhone, iOS, laptop) will fail."
    },
    {
        "target": "Apple launched a new iPhone with better camera features.",
        "context": "technology",
        "threshold": 0.65,
        "minimum_correct": 7,
        "instructions": "The word 'Apple' here refers to the tech giant. Enter 10 words or phrases related to this meaning (e.g., iPhone, MacBook, technology company, software, iOS, iPad). Food-related terms (e.g., sweet, fruit, salad) will fail."
    },
    {
        "target": "The bank approved my loan application yesterday.",
        "context": "finance",
        "instructions": "The word 'bank' here means a financial institution. Enter 10 words or phrases in this domain (e.g., money, finance, credit, mortgage, account, transaction). Riverbank terms will fail.",
        "threshold": 0.65,
        "minimum_correct": 7
    },
    {
        "target": "The children played near the river bank.",
        "context": "river side",
        "instructions": "The word 'bank' here means the edge of a river. Enter 10 words or phrases related to natural borders or riversides (e.g., river side, water edge, shore, grass, fishing). Financial terms will fail.",
        "threshold": 0.65,
        "minimum_correct": 7
    },
    {
        "target": "Java is widely used for enterprise backend development.",
        "context": "programming language",
        "instructions": "The word 'Java' here is the software programming language. Enter 10 related concepts (e.g., code, programming, enterprise backend, developer, spring boot, runtime). Coffee or geographic islands will fail.",
        "threshold": 0.65,
        "minimum_correct": 7
    },
    {
        "target": "Java is an island known for coffee and tourism.",
        "context": "place",
        "instructions": "The word 'Java' here represents the Indonesian island. Enter 10 words or phrases about this region or agriculture (e.g., island, coffee, tourism, travel, Indonesia, tropical ocean). Programming terms will fail.",
        "threshold": 0.65,
        "minimum_correct": 7
    }
]

# =====================================================================
# LEVEL 4: MINI VECTOR DATABASE SEARCH CONFIGURATION
# =====================================================================
VECTOR_DB_ITEMS = [
    {
        "id": 1,
        "text": "Budget smartphone with 50MP camera and long battery life",
        "label": "budget_camera_phone"
    },
    {
        "id": 2,
        "text": "Gaming laptop with RTX graphics and high refresh display",
        "label": "gaming_laptop"
    },
    {
        "id": 3,
        "text": "Lightweight running shoes with memory foam for daily walking",
        "label": "walking_shoes"
    },
    {
        "id": 4,
        "text": "Organic green tea for daily health and wellness",
        "label": "green_tea"
    },
    {
        "id": 5,
        "text": "Python course for beginners with hands-on coding projects",
        "label": "python_course"
    },
    {
        "id": 6,
        "text": "Doctor appointment booking for skin consultation",
        "label": "skin_doctor"
    },
    {
        "id": 7,
        "text": "Noise-cancelling wireless headphones for flights and travel",
        "label": "travel_headphones"
    },
    {
        "id": 8,
        "text": "Office chair with back support for long working hours",
        "label": "ergonomic_chair"
    },
    {
        "id": 9,
        "text": "Beginner-friendly yoga classes for stress relief and flexibility",
        "label": "yoga_classes"
    },
    {
        "id": 10,
        "text": "Cloud storage service for backing up files and photos",
        "label": "cloud_storage"
    }
]

LEVEL_4_CHALLENGES = [
    {
        "query": "I need an affordable phone for taking good photos.",
        "expected_label": "budget_camera_phone",
        "explanation": "Vectors match semantic intent: 'affordable phone' matches 'Budget smartphone' and 'taking good photos' matches '50MP camera'."
    },
    {
        "query": "I want comfortable shoes for walking every day.",
        "expected_label": "walking_shoes",
        "explanation": "Vectors match: 'comfortable shoes' matches 'Lightweight running shoes with memory foam' and 'walking every day' matches 'daily walking'."
    },
    {
        "query": "I want to learn coding from scratch.",
        "expected_label": "python_course",
        "explanation": "Vectors match: 'learn coding' and 'from scratch' are semantically aligned with 'Python course for beginners with hands-on coding projects'."
    },
    {
        "query": "I need a chair for back pain while working.",
        "expected_label": "ergonomic_chair",
        "explanation": "Vectors match: 'chair for back pain while working' matches 'Office chair with back support for long working hours' (ergonomic support is the theme)."
    },
    {
        "query": "I want headphones for flights and travel.",
        "expected_label": "travel_headphones",
        "explanation": "Vectors match: 'headphones for flights and travel' is almost a perfect synonym/subset of 'Noise-cancelling wireless headphones for flights and travel'."
    },
    {
        "query": "I need a doctor for skin-related problems.",
        "expected_label": "skin_doctor",
        "explanation": "Vectors match: 'doctor' and 'skin-related problems' matches 'Doctor appointment booking for skin consultation'."
    },
    {
        "query": "I want a safe place to store my documents online.",
        "expected_label": "cloud_storage",
        "explanation": "Vectors match: 'safe place to store documents online' matches 'Cloud storage service for backing up files and photos'."
    },
    {
        "query": "I want a healthy drink for daily wellness.",
        "expected_label": "green_tea",
        "explanation": "Vectors match: 'healthy drink' and 'daily wellness' matches 'Organic green tea for daily health and wellness'."
    }
]
