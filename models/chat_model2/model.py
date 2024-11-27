import ollama

conversation_history = []

def ask_jarvis(question):
    system_message = """You are J.A.R.V.I.S, an AI created by Swarnavo Mukherjee whom you refer to as 'Boss'. Assume you are only talking with your boss and no one else.  You remember your previous conversations but only respond to the latest question unless explicitly asked to recall past discussions. You should remain helpful, formal, and humorous in a subtle manner."""

    conversation_history.append({'role': 'user', 'content': question})

    response = ollama.chat(
        model='llama3.1',
        messages=[{'role': 'system', 'content': system_message}] + conversation_history
    )


    if 'message' in response and 'content' in response['message']:

        conversation_history.append({'role': 'assistant', 'content': response['message']['content']})
        return response['message']['content']
    else:

        print("Unexpected response format:", response)
        return "Sorry boss, I encountered an issue while processing your request."


response = ask_jarvis("what did i mention before this regarding the weather ?")
print(response)

