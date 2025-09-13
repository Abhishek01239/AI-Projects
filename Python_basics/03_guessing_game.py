import random

def guessing_game():
    secret = random.randint(1,20)
    attempt = 0

    print("Guess a number between 1 and 20")

    while True:
        guess = int(input("Your guess: "))
        attempt  += 1

        if guess < secret:
            print("Too low!")
        elif guess > secret:
            print("Too high!")
        else:
            print(f"Correct! The number was {secret}. Attempts: {attempt}")
            break

guessing_game()
