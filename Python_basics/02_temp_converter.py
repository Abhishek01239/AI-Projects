def temp_Converter():
    choice = input("Conver (C->F) or (F->C)").strip().upper()
    temp = float(input("Enter temprature:"))

    if choice == "C->F":
        print(f"{temp}°C = {(temp*9/5)+32: .2f}°F")
    elif choice == "F->C":
        print(f"{temp}°F = {((temp-32)*5/9):.2f}°C")
    else:
        print("Invalid choice")

temp_Converter()