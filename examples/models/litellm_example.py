"""
Example demonstrating ChatLiteLLM usage with browser-use.

ChatLiteLLM provides access to 100+ LLM providers through a unified interface,
making it easy to switch between different models and providers without 
changing your browser-use code.

This example shows how to use ChatLiteLLM with various providers:
- OpenAI (GPT models)
- Anthropic (Claude models)
- Google (Gemini models)
- Cohere (Command models)
- And many more...

To run this example:
1. Install LiteLLM: pip install litellm
2. Set your API keys as environment variables
3. Run: python examples/models/litellm_example.py
"""

import asyncio
import os
from browser_use import Agent
from browser_use.llm import ChatLiteLLM


async def main():
    """Demonstrate ChatLiteLLM with different providers."""
    
    # Example 1: OpenAI GPT-4
    print("=== OpenAI GPT-4 Example ===")
    if os.getenv('OPENAI_API_KEY'):
        llm_openai = ChatLiteLLM(
            model='openai/gpt-4',
            temperature=0.1,
            max_tokens=1000,
        )
        
        agent = Agent(
            task='Go to google.com and search for "browser automation with AI"',
            llm=llm_openai,
        )
        
        try:
            result = await agent.run()
            print(f"OpenAI result: {result}")
        except Exception as e:
            print(f"OpenAI example failed: {e}")
    else:
        print("Skipping OpenAI example - OPENAI_API_KEY not set")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Anthropic Claude
    print("=== Anthropic Claude Example ===")
    if os.getenv('ANTHROPIC_API_KEY'):
        llm_claude = ChatLiteLLM(
            model='anthropic/claude-3-5-sonnet-20241022',
            temperature=0.1,
            max_tokens=1000,
        )
        
        agent = Agent(
            task='Navigate to python.org and find information about Python 3.11',
            llm=llm_claude,
        )
        
        try:
            result = await agent.run()
            print(f"Claude result: {result}")
        except Exception as e:
            print(f"Claude example failed: {e}")
    else:
        print("Skipping Claude example - ANTHROPIC_API_KEY not set")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Google Gemini
    print("=== Google Gemini Example ===")
    if os.getenv('GOOGLE_API_KEY'):
        llm_gemini = ChatLiteLLM(
            model='google/gemini-pro',
            temperature=0.1,
            max_tokens=1000,
        )
        
        agent = Agent(
            task='Visit github.com and find trending Python repositories',
            llm=llm_gemini,
        )
        
        try:
            result = await agent.run()
            print(f"Gemini result: {result}")
        except Exception as e:
            print(f"Gemini example failed: {e}")
    else:
        print("Skipping Gemini example - GOOGLE_API_KEY not set")
    
    print("\n" + "="*50 + "\n")
    
    # Example 4: Using custom provider settings
    print("=== Custom Provider Example ===")
    if os.getenv('OPENAI_API_KEY'):
        llm_custom = ChatLiteLLM(
            model='openai/gpt-3.5-turbo',
            temperature=0.2,
            max_tokens=500,
            # Custom LiteLLM parameters
            litellm_params={
                'drop_params': True,  # Drop unsupported parameters
                'timeout': 30,        # Custom timeout
            }
        )
        
        agent = Agent(
            task='Go to news.ycombinator.com and find the top story title',
            llm=llm_custom,
        )
        
        try:
            result = await agent.run()
            print(f"Custom settings result: {result}")
        except Exception as e:
            print(f"Custom example failed: {e}")
    else:
        print("Skipping custom example - OPENAI_API_KEY not set")
    
    print("\n" + "="*50 + "\n")
    
    # Example 5: Using different providers with the same code
    print("=== Multi-Provider Comparison ===")
    
    providers = []
    
    if os.getenv('OPENAI_API_KEY'):
        providers.append(('OpenAI GPT-3.5', ChatLiteLLM(model='openai/gpt-3.5-turbo', temperature=0.1)))
    
    if os.getenv('ANTHROPIC_API_KEY'):
        providers.append(('Claude Haiku', ChatLiteLLM(model='anthropic/claude-3-haiku-20240307', temperature=0.1)))
    
    if os.getenv('GOOGLE_API_KEY'):
        providers.append(('Gemini Pro', ChatLiteLLM(model='google/gemini-pro', temperature=0.1)))
    
    task = 'Visit example.com and describe what you see'
    
    for provider_name, llm in providers:
        print(f"--- Testing {provider_name} ---")
        agent = Agent(task=task, llm=llm)
        
        try:
            result = await agent.run()
            print(f"{provider_name} completed successfully")
        except Exception as e:
            print(f"{provider_name} failed: {e}")
        
        print()


def print_setup_instructions():
    """Print setup instructions for the example."""
    print("ChatLiteLLM Setup Instructions:")
    print("=" * 40)
    print("1. Install LiteLLM:")
    print("   pip install litellm")
    print()
    print("2. Set up API keys (choose one or more):")
    print("   export OPENAI_API_KEY='your-openai-key'")
    print("   export ANTHROPIC_API_KEY='your-anthropic-key'")
    print("   export GOOGLE_API_KEY='your-google-key'")
    print("   export COHERE_API_KEY='your-cohere-key'")
    print()
    print("3. Run the example:")
    print("   python examples/models/litellm_example.py")
    print()
    print("Supported providers include:")
    print("- OpenAI (openai/gpt-4, openai/gpt-3.5-turbo, etc.)")
    print("- Anthropic (anthropic/claude-3-5-sonnet-20241022, etc.)")
    print("- Google (google/gemini-pro, google/gemini-pro-vision, etc.)")
    print("- Cohere (cohere/command-r, cohere/command-r-plus, etc.)")
    print("- Meta (meta-llama/llama-2-70b-chat, etc.)")
    print("- Mistral (mistral/mistral-large, mistral/mistral-medium, etc.)")
    print("- And 100+ more providers!")
    print("=" * 40)


if __name__ == '__main__':
    print_setup_instructions()
    print()
    
    # Check if any API keys are available
    api_keys = [
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY', 
        'GOOGLE_API_KEY',
        'COHERE_API_KEY'
    ]
    
    available_keys = [key for key in api_keys if os.getenv(key)]
    
    if not available_keys:
        print("No API keys found in environment variables.")
        print("Please set at least one API key to run the examples.")
        print("Available keys:", ', '.join(api_keys))
    else:
        print(f"Found API keys: {', '.join(available_keys)}")
        print("Running examples...\n")
        asyncio.run(main()) 