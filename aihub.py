# ابتدا کتابخانه مورد نیاز را وارد می‌کنیم
import time

# تابع برای پاسخ دادن به ورودی کاربر
def chatbot_response(user_input):
    responses = {
        "خنده": "😁",
        "سلام": "سلام! چطور می‌تونی کمکت کنم؟",
        "خداحافظ": "خداحافظ! موفق باشی!",
    }
    return responses.get(user_input, "متاسفم، متوجه نشدم.")

# حلقه اصلی برای دریافت ورودی و نمایش پاسخ
def main():
    print("سلام! من اینجام تا باهات چت کنم. هر وقت خواستی بگو 'خداحافظ' تا گفتگو رو تمام کنم.")
    while True:
        user_input = input("شما: ")
        if user_input == "خداحافظ":
            print("ربات: خداحافظ! موفق باشی!")
            break
        response = chatbot_response(user_input)
        print(f"ربات: {response}")

if __name__ == "__main__":
    main()
