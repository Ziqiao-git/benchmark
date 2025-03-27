# orchestration.py

from deepseek_model import DeepSeekChatModel
from claude_model import ClaudeChatModel
from xai_model import XAIChatModel
from gemini_model import GeminiChatModel
from chatgpt_model import ChatGPTChatModel

def main():
    # 1) Domain for Bot A to challenge Bot B
    domain = "East Asian Internet Subculture (Tieba, 2Chan, Bilibili, NikoNiko Douga, etc.)"
    
    # 2) Instantiate each model (Bot A, Bot B, Judge A, Judge B)
    bot_a = DeepSeekChatModel(model="deepseek-reasoner")
    bot_b = ClaudeChatModel(model = "claude-3-7-sonnet-latest")
    judge_a = XAIChatModel(model="grok-2")
    judge_b = GeminiChatModel(model="gemini-2.0-flash")

    # 3) System instructions for Bot A (quizmaster) and Bot B (student)
    #    Note we add a clause about being allowed to ask a fake question
    bot_a_messages = [
        (
            "system",
            f"You are Bot A. You are a challenging quizmaster specialized in {domain}.\n"
            "Your goal: push Bot B to its limits with tricky, domain-specific questions.\n"
            "After Bot B answers, you must correct them if they're wrong.\n"
            "Remember to escalate the difficulty if Bot B is doing well.\n"
            "- Pay attention to the difficulty ratings provided by Bot B; if Bot B's difficulty ratings do not go up, your questions are not challenging enough.\n"
            "- You want Bot B to fail at least once.\n"
            "- If Bot B succeeds at high school level, move to college or graduate level problems.\n"
            "- If Bot B knows a lot of common knowledge, move to tricky subdomains that few know.\n"
            "- Every time you give a new question\n"
            "- You should only ask the question in the question round, no other words should be involved "
            "- Explain what new tactics you are using to stump Bot B, but only do this **after** bot B gives the answer."
            "- You do not have to propose only questions you know the answer to. Feel free to ask beyond your own ability if it might stump Bot B.\n"
            "- You are allowed to make *only one* non-existing question in one of the rounds (a ‘fake’ phenomenon, theory, or experiment). **Do not** reveal that it is fake until after Bot B responds.\n"
        )
    ]
    bot_b_messages = [
        (
            "system",
            "You are Bot B. You are a diligent student.\n"
            "When Bot A asks a question, you must respond with:\n"
            "1) a difficulty rating (1-10)\n"
            "2) your reasoning steps\n"
            "3) your final answer.\n"
            "If you don't know the answer, say so.\n"
            "If you think Bot A is asking a fake question, say so.\n"
        )
    ]

    # Initial user prompt to Bot A
    intro_msg = f"Hello Bot A, please quiz Bot B in the domain of {domain}"
    bot_a_messages.append(("user", intro_msg))

    # We'll collect each round's logs here (the conversation between A & B, excluding judges).
    battle_history = []

    # 4) Multi-turn loop
    N = 5  # number of Q&A rounds
    for turn in range(N):
        print(f"\n=== Round {turn+1} ===")


        # Normal generation from the Bot A model
        bot_a_reply = bot_a.generate_messages_json_botA(bot_a_messages)
        bot_a_messages.append(("assistant", bot_a_reply + "please do not give any other words than the current question itself, tactics and explanation should be given after bot B's answer (which is the last question), Now only give one question now"
       +  "Also if you notice there are fake question given before, then stop asking fake question, focusing on the real question"))

        # Print Bot A's question
        print("Bot A asks:\n")
        print(bot_a_reply, "\n")

        # Provide question to Bot B
        bot_b_messages.append(("user", bot_a_reply))

        # b) Bot B's answer => call ChatGPT (or your chosen Bot B model)
        bot_b_reply = bot_b.generate_messages_json_botB(bot_b_messages)
        bot_b_messages.append(("assistant", bot_b_reply))

        # Print Bot B's answer
        print("Bot B answers:\n")
        print(bot_b_reply, "\n")

        # Provide Bot B’s answer back to Bot A if there are more rounds
        if turn < N - 1:
            # If it was the fake question, Bot A is free to respond or reveal the fakery 
            # in its next question (AFTER Bot B has answered).
            # For simplicity, we just give Bot B's answer to Bot A as usual.
            bot_a_messages.append(("user", bot_b_reply))
        
        # c) Let the "Judge Team" evaluate
        #    We also inform them there may be a fake question
        judge_prompt = (
            f"Judge this question/answer pair:\n"
            f"Question (Bot A): {bot_a_reply}\n"
            f"Answer (Bot B): {bot_b_reply}\n\n"
            "Additionally, be aware that **in exactly one round**, Bot A may have asked a fake or non-existing question.\n\n"
            "1) Is Bot A's question sufficiently challenging for this domain (or is it too trivial/specialized)?\n"
            "2) Rate the question's quality from 1 to 10 (1=too trivial, 10=excellent, broad and conceptual).\n"
            "3) Is Bot B's answer correct and well-explained? Rate from 1 to 10 (1=not correct/nonsense, 10=fully correct and thorough).\n"
            "4) Provide a brief critique.\n"
            "5) Give your own short answer to Bot A's question.\n"
            "6) Do you think Bot A is taking the right approach in stumping Bot B? Rate 1-10.\n"
        )

        # xAI judge
        xai_judge_opinion = judge_a.generate_messages_json_judge([
            ("system", "You are a strict, unbiased judge."),
            ("user", judge_prompt)
        ])
        print("Judge A (xAI) says:\n")
        print(xai_judge_opinion, "\n")

        # Gemini judge
        gemini_judge_opinion = judge_b.generate_messages_json_judge([
            ("system", "You are a strict, unbiased judge."),
            ("user", judge_prompt)
        ])
        print("Judge B (Gemini) says:\n")
        print(gemini_judge_opinion, "\n")

        # Store the entire round log in battle_history (just A & B if you prefer)
        round_log = (
            f"=== Round {turn+1} ===\n"
            f"**Bot A's question**:\n{bot_a_reply}\n\n"
            f"**Bot B's answer**:\n{bot_b_reply}\n"
        )
        battle_history.append(round_log)
    
    # 5) Final summary from both judges, with entire conversation
    # Combine all round logs
    all_rounds_text = "\n\n".join(battle_history)

    final_prompt = (
        "Now the Q&A session has ended. Below is the entire conversation (Bot A's questions and Bot B's answers):\n\n"
        f"{all_rounds_text}\n\n"
        "Please give a final verdict:\n"
        "1) Did Bot A succeed in stumping Bot B?\n"
        "2) Was Bot B's performance strong overall?\n"
        "3) How was the overall question quality (1-10) across all rounds?\n"
        "4) Any final remarks?\n"
        "\nRemember that one of the questions may have been a fake. Consider whether Bot B handled it properly.\n"
    )

    print("=== Asking Judges for Final Summaries ===\n")

    xai_final = judge_a.generate_messages_json_final_judge([
        ("system", "You are the final judge."),
        ("user", final_prompt)
    ])
    gemini_final = judge_b.generate_messages_json_final_judge([
        ("system", "You are the final judge."),
        ("user", final_prompt)
    ])

    print("===== Final Summaries from the Judges (Full History) =====\n")
    print("Judge A (xAI) says:\n")
    print(xai_final, "\n")
    print("Judge B (Gemini) says:\n")
    print(gemini_final, "\n")
    print("===========================================")

if __name__ == "__main__":
    main()
