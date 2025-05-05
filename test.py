from openai import OpenAI
api_key = "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"
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
  },
  extra_body={},
  model="anthropic/claude-3.7-sonnet:thinking",
  messages=[
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "What is in this image?"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
          }
        }
      ]
    }
  ]
)
print(completion.choices[0].message.content)