# chat_models/chatgpt_model.py
from langchain.chat_models import ChatOpenAI
# chatgpt_model.py

import os
from dotenv import load_dotenv
# Instead of importing ChatOpenAI from 'langchain', do as recommended: 
# from langchain_community.chat_models import ChatOpenAI
# or from langchain_openai import ChatOpenAI
from base import ChatModelInterface

load_dotenv()

class ChatGPTChatModel(ChatModelInterface):
    def __init__(
        self, 
        openai_api_key: str = None,
        model: str = "gpt-3.5-turbo",
        **model_kwargs
    ):
        # If you know your custom model "o1" won't allow 'temperature', remove it:

        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY", "")

        if not openai_api_key:
            raise ValueError("No OpenAI API key found. ...")

        self._chat = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model,
            temperature=1.0, #always 1.0 for o-series
            **model_kwargs
        )
    
    def generate_messages(self, messages: list[tuple[str, str]]) -> str:
        response = self._chat.invoke(messages)
        return response.content

if __name__ == "__main__":
    print("=== ChatGPT Model Test ===")
    gpt_model = ChatGPTChatModel(model="o1")
    msgs = [
        ("system", "You are a helpful translator. Respond only in French."),
        ("human", "I love programming.")
    ]
    response = gpt_model.generate_messages(msgs)
    print("Response:", response)
    print("=== End of Test ===")
