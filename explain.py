import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")

HF_MODEL_URL = "https://router.huggingface.co/hf-inference/models/google/flan-t5-large"

headers = {
    "Authorization": f"Bearer {HF_API_KEY}"
}


def explain_topic(topic: str, level: str) -> str:
    if not topic:
        return "Please provide a topic."

    # Level based prompt (ChatGPT style)
    if level == "std1":
        prompt = f"""
Explain the topic "{topic}" for a Class 1 student.

Rules:
- Use very easy words
- Use very small sentences
- Use simple daily life example
- Use 5 to 7 lines only
"""
    elif level == "std5":
        prompt = f"""
Explain the topic "{topic}" for a Class 5 student.

Rules:
- Use simple language
- Use bullet points
- Give one small example
- 8 to 12 lines
"""
    elif level == "std10":
        prompt = f"""
Explain the topic "{topic}" for a Class 10 student.

Rules:
- Use scientific terms
- Explain step-by-step
- Add real life example
- Add short conclusion
- Use bullet points + paragraphs
"""
    else:  # diploma
        prompt = f"""
Explain the topic "{topic}" for Diploma/College students.

Write in professional educational format:

1. Definition
2. Working / Explanation (Step-by-step)
3. Key Points
4. Applications / Uses
5. Advantages (if applicable)
6. Conclusion

Make it clear and easy to understand.
Use bullet points and short paragraphs.
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7
        }
    }

    # Retry system
    for _ in range(3):
        try:
            response = requests.post(HF_MODEL_URL, headers=headers, json=payload)

            if response.status_code == 200:
                data = response.json()

                if isinstance(data, list) and "generated_text" in data[0]:
                    return data[0]["generated_text"].strip()

            # If model is loading
            if response.status_code == 503:
                time.sleep(5)
                continue

            break

        except Exception:
            break

    # If HuggingFace fails, return fallback
    return _local_fallback_explanation(topic, level)


def _local_fallback_explanation(topic, level):

    # Fallback should also be detailed and topic based
    if level == "std1":
        return f"""
{topic.title()} (Class 1 Explanation)

- {topic.title()} is something we learn about.
- It is useful in our daily life.
- It helps us understand the world.
- It is simple and interesting.

Example:
- We can see {topic.lower()} around us.
"""

    elif level == "std5":
        return f"""
{topic.title()} (Class 5 Explanation)

{topic.title()} is an important topic.

Key Points:
- It explains how something works.
- It is related to our environment or science.
- It is useful in our daily life.

Example:
- We can understand {topic.lower()} using real life situations.

Conclusion:
{topic.title()} is a useful and interesting topic to learn.
"""

    elif level == "std10":
        return f"""
{topic.title()} (Class 10 Explanation)

Definition:
{topic.title()} is an important scientific topic or concept.

Explanation:
- It helps us understand how a process/system works.
- It has different steps and stages.
- It is used in many real-life applications.

Example:
- We can observe the effects of {topic.lower()} in daily life.

Conclusion:
{topic.title()} is important for understanding science and technology.
"""

    else:
        return f"""
{topic.title()} (Diploma Level Explanation)

Definition:
{topic.title()} is an important concept in science/technology.

Working / Explanation:
1. It starts with the basic idea or process.
2. Then it goes through intermediate steps.
3. Finally, it gives the final output/result.

Applications:
- Used in practical systems
- Useful in engineering and science
- Helps in understanding real-world processes

Conclusion:
{topic.title()} is a useful topic for technical learning.
"""
