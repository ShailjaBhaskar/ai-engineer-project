from app.services.ai_service import get_ai_response

def main():
    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "exit":
            break
        
        response = get_ai_response(user_input)
        print("AI:", response)

if __name__ == "__main__":
    main()