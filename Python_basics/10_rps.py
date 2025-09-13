import random

def rps():
    choices = ["rock", "paper", "scissors"]
    user = input("Choose rock/paper/scissors: ").lower()
    comp = random.choice(choices)

    print(f"Computer choice: {comp}")

    if user == comp:
        print("It is tie!")
    elif (user == "rock" and comp == "scissors") or (user == "paper" and comp == "rock") or (user == "scissors" and comp == "paper"):
        print("You win!")
    
    else:
        print("You lose")
rps()