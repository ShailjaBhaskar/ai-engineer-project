from openai import OpenAI
from app.core.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def get_ai_response(user_input: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI tutor. Answer clearly, concisely, and in bullet points. Don't exceed more than 3 points."},
            {"role": "user", "content": user_input}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content