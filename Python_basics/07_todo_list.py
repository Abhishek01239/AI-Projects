def todo_list():
    tasks = []

    while True:
        print("\n1. Add Task\n2. View Task\n3. Remove Task\n4. Exit")
        choice = input("Enter choice:")

        if choice == "1":
            task = input("Enter task: ")
            tasks.append(task)
            print("Task Added!")
        elif choice == "2":
            print("\nYour Tasks:")
            for i, t in enumerate(tasks, 1):
                print(f"{i}. {t}")
        elif choice == "3":
            num = int(input("Enter task number to remove: "))
            if 1 <= num <= len(tasks):
                tasks.pop(num-1)
                print("Task Removed!")
        elif choice == "4":
            print("Exitinf TO-DO List...")
            break
        else:
            print("Invalid Choice")

todo_list()