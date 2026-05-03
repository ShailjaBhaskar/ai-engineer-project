import json
from openai import OpenAI
from app.core.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def get_ai_response(user_input: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
You are a JSON generator.

Rules:
1. Always return valid JSON
2. Do NOT include any extra text
3. Format:
{
  "answer": "..."
}
""",
            },
            {"role": "user", "content": user_input},
        ],
        temperature=0.3,
    )

    content = response.choices[0].message.content

    # 🔥 Step 2: Parse JSON safely
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response", "raw": content}
