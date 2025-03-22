"""
Utilities for handling asynchronous operations safely within Streamlit.
"""

import asyncio
import functools
import time
from typing import Any, Callable

def run_async(coroutine_func):
    """
    Decorator to safely run async functions in Streamlit environment.
    
    This decorator handles the event loop properly regardless of the current context
    (whether there's a running event loop or not) and prevents issues with
    Streamlit's thread context.
    
    Args:
        coroutine_func: The async function to wrap
        
    Returns:
        A synchronous function that wraps the async function
    """
    @functools.wraps(coroutine_func)
    def wrapper(*args, **kwargs):
        result = None
        
        # Try to get the current event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create a new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the coroutine in the event loop
        if loop.is_running():
            # Create a future and schedule it in the running loop
            future = asyncio.run_coroutine_threadsafe(coroutine_func(*args, **kwargs), loop)
            # Wait for the result with a timeout (30 seconds)
            result = future.result(30)
        else:
            # Run the coroutine directly in the loop
            result = loop.run_until_complete(coroutine_func(*args, **kwargs))
        
        return result
    
    return wrapper

async def async_timeout(coroutine, timeout=30):
    """
    Run a coroutine with a timeout.
    
    Args:
        coroutine: The coroutine to run
        timeout: Timeout in seconds
        
    Returns:
        The result of the coroutine or raises asyncio.TimeoutError
    """
    return await asyncio.wait_for(coroutine, timeout)
