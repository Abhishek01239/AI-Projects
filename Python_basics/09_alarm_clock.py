import time 
import winsound

def alarm_clock():
    alarm_time = input("Enter alarm time (HH:MM in 24 hr format): ")

    print(f"Alarm set for {alarm_time}")
    while True:
        current_time = time.strftime("%H:%M")
        if current_time == alarm_time:
            print("Wake up!")
            try:
                winsound.Beep(1000, 2000)
            except:
                print("Beep! Veep! (sound not supported here)")
            break

alarm_clock()