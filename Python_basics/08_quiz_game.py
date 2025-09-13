def quiz_game():
    score = 0
    print("Welcome to the quiz!")

    q1 = input("Q1: What is the capital of India?")
    if q1.lower() == "delhi":
        score += 1

    q2 = input("Q2: What is 5+7?")
    if q2 == "12":
        score += 1
    
    q3 = input("Q3: Who is know as the father of computers?")
    if "babbage" in q3.lower():
        score += 1
    
    print(f"Your Score: {score}/3")

quiz_game()