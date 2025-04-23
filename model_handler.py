# model_handler.py
import os
import json
import importlib
from dotenv import load_dotenv
import logging

load_dotenv()

# In model_handler.py
def get_chat_model(model_id, **kwargs):
    """
    Create a chat model instance based on configuration.
    
    Args:
        model_id: The ID of the model in the config file
        **kwargs: Additional parameters to pass to the model
    
    Returns:
        An initialized chat model instance
    """
    # Load the configuration
    with open("models_config.json", 'r') as f:
        config = json.load(f)
    
    if model_id not in config:
        raise ValueError(f"Model ID '{model_id}' not found in config")
    
    model_config = config[model_id]
    
    # Get model module and class from config
    module_path = model_config["module"]
    class_name = model_config["class"]
    
    # Import the module and get the class
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    
    # Get API key from environment or config
    api_key_env = model_config.get("api_key_env")
    api_key = None
    if api_key_env:
        api_key = os.getenv(api_key_env)
    
    # Prepare parameters
    params = model_config.get("params", {}).copy()
    
    # Replace placeholders with actual values
    model_path = kwargs.get('model_path')
    if model_path and "model_path" in params and params["model_path"] == "${MODEL_PATH}":
        params["model_path"] = model_path
    
    model_name = kwargs.get('model_name')
    if model_name and "model_name" in params and params["model_name"] == "${MODEL_NAME}":
        params["model_name"] = model_name
    
    # Add API key to parameters if available
    if api_key:
        api_key_param = model_config.get("api_key_param", "api_key")
        params[api_key_param] = api_key
    
    try:
        # Initialize the model
        model = model_class(**params)
        
        # Wrap model with common interface if needed
        if "wrapper" in model_config:
            wrapper_module_path = model_config["wrapper"]["module"]
            wrapper_class_name = model_config["wrapper"]["class"]
            
            wrapper_module = importlib.import_module(wrapper_module_path)
            wrapper_class = getattr(wrapper_module, wrapper_class_name)
            
            return wrapper_class(model)
        
        return model
    except Exception as e:
        logging.error(f"Error initializing model {model_id}: {str(e)}")
        raise