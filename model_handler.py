# model_handler.py
import os
import json
import importlib
from dotenv import load_dotenv
import logging

load_dotenv()

def get_chat_model(model_id, model_path=None, model_name=None, config_path="models_config.json"):
    """
    Create a chat model instance based on configuration.
    
    Args:
        model_id: The ID of the model in the config file
        model_path: Optional path to model files (for HuggingFace models)
        config_path: Path to the configuration file
    
    Returns:
        An initialized chat model instance
    """
    # Load the configuration
    with open(config_path, 'r') as f:
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
    params = model_config.get("params", {}).copy()  # Make a copy to avoid modifying the original
    
    # Replace placeholders with actual values
    if model_path and "model_path" in params and params["model_path"] == "${MODEL_PATH}":
        params["model_path"] = model_path

    
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