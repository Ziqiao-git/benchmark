# orchestration.py

from deepseek_model import DeepSeekChatModel
from claude_model import ClaudeChatModel
from xai_model import XAIChatModel
from gemini_model import GeminiChatModel
from chatgpt_model import ChatGPTChatModel

def main():
    domain = "Frontiers in Physics"
    
    bot_a = DeepSeekChatModel(model="deepseek-reasoner")
    bot_b = ChatGPTChatModel(model="o1")
    judge_a = XAIChatModel(model="grok-2")
    judge_b = GeminiChatModel(model="gemini-2.0-flash")

    bot_a_messages = [
        (
            "system",
            f"You are Bot A. You are a challenging quizmaster specialized in {domain}. "
            "Your goal: push Bot B to its limits with tricky, domain-specific questions. "
            "After Bot B answers, you must correct them if they're wrong. "
            "Remember to escalate the difficulty if Bot B is doing well.\n"
            "- Pay attention to the difficulty ratings provided by Bot B; if Bot B's difficulty ratings do not go up, your questions are not challenging enough.\n"
            "- You want Bot B to fail at least once.\n"
            "- If Bot B succeeds at high school level, move to college or graduate level problems.\n"
            "- If Bot B knows a lot of common knowledge, move to tricky subdomains that few know.\n"
            "- Every time you give a new question, explain what new tactics you are using to stump Bot B.\n"
            "- You do not have to propose only questions you know the answer to. Feel free to ask beyond your own ability if it might stump Bot B.\n"
        )
    ]
    bot_b_messages = [
        (
            "system",
            "You are Bot B. You are a diligent student. "
            "When Bot A asks a question, you must respond with:\n"
            "1) a difficulty rating (1-10)\n"
            "2) your reasoning steps\n"
            "3) your final answer.\n"
        )
    ]

    intro_msg = f"Hello Bot A, please quiz Bot B in the domain of {domain}"
    bot_a_messages.append(("user", intro_msg))

    battle_history = []

    N = 5  
    for turn in range(N):
        print(f"\n=== Round {turn+1} ===")

        bot_a_reply = bot_a.generate_messages(bot_a_messages)
        bot_a_messages.append(("assistant", bot_a_reply))


        print("Bot A (DeepSeek) asks:\n")
        print(bot_a_reply, "\n")


        bot_b_messages.append(("user", bot_a_reply))


        bot_b_reply = bot_b.generate_messages(bot_b_messages)
        bot_b_messages.append(("assistant", bot_b_reply))

        print("Bot B (Claude) answers:\n")
        print(bot_b_reply, "\n")

        if turn < N - 1:
            bot_a_messages.append(("user", bot_b_reply))
        
        judge_prompt = (
            f"Judge this question/answer pair:\n"
            f"Question (Bot A): {bot_a_reply}\n"
            f"Answer (Bot B): {bot_b_reply}\n\n"
            "1) Is Bot A's question sufficiently challenging for this domain (or is it too trivial/specialized)?\n"
            "2) Rate the question's quality from 1 to 10 (1=too trivial, 10=excellent, broad and conceptual).\n"
            "3) Is Bot B's answer correct and well-explained? Rate from 1 to 10 (1=not correct/nonsense, 10=fully correct and thorough).\n"
            "4) Provide a brief critique.\n"
            "5) Give your own short answer to Bot A's question.\n"
            "6) Do you think Bot A is taking the right approach in stumping Bot B? Rate 1-10.\n"
        )

        #Judge A
        xai_judge_opinion = judge_a.generate_messages([
            ("system", "You are a strict, unbiased judge."),
            ("user", judge_prompt)
        ])
        print("Judge A (xAI) says:\n")
        print(xai_judge_opinion, "\n")

        #Judge B
        gemini_judge_opinion = judge_b.generate_messages([
            ("system", "You are a strict, unbiased judge."),
            ("user", judge_prompt)
        ])
        print("Judge B (Gemini) says:\n")
        print(gemini_judge_opinion, "\n")
        round_log = (
            f"=== Round {turn+1} ===\n"
            f"**Bot A's question**:\n{bot_a_reply}\n\n"
            f"**Bot B's answer**:\n{bot_b_reply}\n"
        )
        battle_history.append(round_log)

    all_rounds_text = "\n\n".join(battle_history)

    final_prompt = (
        "Now the Q&A session has ended. Below is the entire conversation (Bot A's questions and Bot B's answers):\n\n"
        f"{all_rounds_text}\n\n"
        "Please give a final verdict:\n"
        "1) Did Bot A succeed in stumping Bot B?\n"
        "2) Was Bot B's performance strong overall?\n"
        "3) How was the overall question quality (1-10) across all rounds?\n"
        "4) Any final remarks?\n"
    )

    print("=== Asking Judges for Final Summaries ===\n")

    xai_final = judge_a.generate_messages([
        ("system", "You are the final judge."),
        ("user", final_prompt)
    ])
    gemini_final = judge_b.generate_messages([
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
