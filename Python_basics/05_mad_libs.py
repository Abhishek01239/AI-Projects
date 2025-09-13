def mad_libs():
    noun = input("Enter a noun:")
    adj = input("Enter an adjective:")
    verb = input("Enter a verb:")

    story = f"One day, a {adj} {noun} decide to {verb} all day long!"
    print("\nYour story:")
    print(story)

mad_libs()