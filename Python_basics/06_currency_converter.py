def currency_converter():
   
    USD_TO_INR = 83.0  

    print("Choose conversion:")
    print("1. USD → INR")
    print("2. INR → USD")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        usd = float(input("Enter amount in USD: "))
        inr = usd * USD_TO_INR
        print(f"{usd:.2f} USD = {inr:.2f} INR")

    elif choice == "2":
        inr = float(input("Enter amount in INR: "))
        usd = inr / USD_TO_INR
        print(f"{inr:.2f} INR = {usd:.2f} USD")

    else:
        print("❌ Invalid choice. Please enter 1 or 2.")

currency_converter()
