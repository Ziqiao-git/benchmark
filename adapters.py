# adapters.py
class LangChainAdapter:
    """Adapter for LangChain models to provide a consistent interface."""
    
    def __init__(self, model):
        self.model = model
    
    def generate_messages(self, messages):
        """Generate response from model using a list of (role, content) tuples."""
        response = self.model.invoke(messages)
        return response.content