import random

def dice_simulator():
    while True:
        roll = input("Roll the dicec? (y/n):").lower()
        if roll == 'y':
            print("You rolled: ", random.randint(1,6))
        elif roll == 'n':
            print("Exiting...")
            break
        else:
            print("Invalid input")
    
dice_simulator()