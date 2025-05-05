# adapters.py
import logging
import os
from openai import OpenAI
from dotenv import load_dotenv


from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class LangChainAdapter:
    def __init__(self, model):
        self.model = model

    def generate_messages(self, tuples):               # tuples=[("user", "..."), ...]
        # ── convert tuples → LangChain Message objects
        role_map = {
            "system": SystemMessage,
            "user":   HumanMessage,
            "assistant": AIMessage,
        }
        lc_msgs = [role_map[r](content=c) for r, c in tuples]

        # ── call the underlying LC chat model; no try/except needed
        resp = self.model.invoke(lc_msgs)
        return resp.content
# class LangChainAdapter:
#     """Adapter for LangChain models to provide a consistent interface."""
    
#     def __init__(self, model):
#         self.model = model

#     def generate_messages(self, messages):
#         """Generate response from model using a list of (role, content) tuples."""
#         try:
#             resp = self.model.invoke(messages)
#             print(resp.content)
#             return resp.content
#         except Exception:
#             pass

#         load_dotenv()  # ensure OPENAI_API_KEY is loaded
#         client = OpenAI()


#         # resp = client.responses.create(
#         #     model=self.model.model_name,
#         #     input=messages,
#         #     text={
#         #         "format": {
#         #           "type": "text"
#         #         }
#         #       },
#         #       reasoning={
#         #         "effort": "medium"
#         #       },
#         #       tools=[],
#         #       store=True
#         # )
#         # print(messages[0][1])
#         resp = client.responses.create(
#             model=self.model.model_name,

#             input=[
#                 {"role": "system", "content": messages[0][1]},
#                 {"role": "user", "content": messages[1][1]},
#             ],
#             text={
#                     "format": {
#                       "type": "text"
#                     }
#                   },
#                   reasoning={
#                     "effort": "medium"
#                   },
#                   tools=[],
#                   store=True
#         )
#         # print(resp.output[1].content[0])
#         # print(resp.output[1].content[0].text)

#         # Wrap it so .content still works
#         # class FakeMsg:
#         #     def __init__(self, text): self.content = text

#         return resp.output[1].content[0].text

class LocalModelHandler:
    """Handles local models with vLLM preference and HuggingFace fallback."""
    
    def __init__(self, model_path=None, model_name=None, temperature=0.7, 
                 max_tokens=512, try_vllm_first=True, **kwargs):
        self.model_path = model_path or model_name
        self.model_name = model_name or os.path.basename(model_path) if model_path else None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self.model = None
        
        # Try vLLM first if preferred
        if try_vllm_first:
            try:
                self._init_vllm()
                logging.info(f"Successfully initialized vLLM with model: {self.model_path}")
            except (ImportError, Exception) as e:
                logging.warning(f"Failed to initialize vLLM: {str(e)}")
                logging.info("Falling back to HuggingFace Transformers")
                self._init_huggingface()
        else:
            # Directly use HuggingFace
            self._init_huggingface()
    
    def _init_vllm(self):
        """Initialize using vLLM."""
        try:
            from langchain_community.llms import VLLM
            
            self.model = VLLM(
                model=self.model_path,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tensor_parallel_size=8,  # Use all 8 GPUs
                gpu_memory_utilization=0.85,  # Control memory usage
                **self.kwargs
            )
        except ImportError:
            raise ImportError("vLLM package not found. Install with 'pip install vllm'.")
        except Exception as e:
            raise Exception(f"Error initializing vLLM: {str(e)}")
    
    def _init_huggingface(self):
        """Initialize using HuggingFace Transformers."""
        try:
            from langchain_community.chat_models import ChatHuggingFace
                # Initialize the HuggingFace model
            self.model = ChatHuggingFace(
                model_path=self.model_path,
                temperature=self.temperature,
                max_length=self.max_tokens,
                model_kwargs={
                    "device_map": "auto",
                    "torch_dtype": "auto"
                },
                **self.kwargs
            )
        except ImportError:
            raise ImportError("Transformers package not found. Install with 'pip install transformers'.")
        except Exception as e:
            raise Exception(f"Error initializing HuggingFace model: {str(e)}")
    
    def invoke(self, messages):
        """Invoke the model with messages."""
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        # Convert tuple format to LangChain message format
        langchain_messages = []
        for role, content in messages:
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
        
        # Return the response
        return self.model.invoke(langchain_messages)
