from openai import OpenAI

client = OpenAI()

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # lightweight + cheap model
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("✅ Success! Response:", response.choices[0].message.content)
except Exception as e:
    print("❌ Error:", e)
