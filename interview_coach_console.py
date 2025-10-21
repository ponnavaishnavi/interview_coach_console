# interview_coach_console.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

qa_pairs = [
    ("What is photosynthesis?", "Photosynthesis converts CO2 and water into glucose and oxygen using sunlight."),
    ("Explain Newton's second law.", "Force equals mass times acceleration (F=ma)."),
    ("Define machine learning.", "Machine learning is AI where computers learn patterns from data."),
]

answer = input("Enter your answer to an interview question:\n")

questions = [q for q,_ in qa_pairs]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions + [answer])
sims = cosine_similarity(X[-1], X[:-1])
best_idx = sims.argmax()
expected = qa_pairs[best_idx][1]
score = sims[0,best_idx]*100

print("\nClosest Question:", questions[best_idx])
print("Expected Answer:", expected)
print(f"Similarity Score: {score:.2f}%")
