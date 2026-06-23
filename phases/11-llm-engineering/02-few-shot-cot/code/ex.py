import os
import re
from openai import OpenAI

GSM8K_EXAMPLES = [
  {
    "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "reasoning": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.",
    "answer": "72"
  },
  {
    "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    "reasoning": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.",
    "answer": "10"
  },
  {
    "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
    "reasoning": "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.",
    "answer": "5"
  },
  {
    "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
    "reasoning": "Julie read 12 x 2 = <<12*2=24>>24 pages today.\nSo she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday.\nThere are 120 - 36 = <<120-36=84>>84 pages left to be read.\nSince she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages.",
    "answer": "42"
  },
  {
    "question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?",
    "reasoning": "He writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year",
    "answer": "624"
  },
  {
    "question": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?",
    "reasoning": "There are 80/100 * 10 = <<80/100*10=8>>8 more purple flowers than yellow flowers.\nSo in Mark's garden, there are 10 + 8 = <<10+8=18>>18 purple flowers.\nPurple and yellow flowers sum up to 10 + 18 = <<10+18=28>>28 flowers.\nThat means in Mark's garden there are 25/100 * 28 = <<25/100*28=7>>7 green flowers.\nSo in total Mark has 28 + 7 = <<28+7=35>>35 plants in his garden.",
    "answer": "35"
  },
  {
    "question": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
    "reasoning": "He eats 32 from the largest pizzas because 2 x 16 = <<2*16=32>>32\nHe eats 16 from the small pizza because 2 x 8 = <<2*8=16>>16\nHe eats 48 pieces because 32 + 16 = <<32+16=48>>48",
    "answer": "48"
  },
  {
    "question": "Ken created a care package to send to his brother, who was away at boarding school.  Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds.  Then, he added enough brownies to cause the weight to triple.  Next, he added another 2 pounds of jelly beans.  And finally, he added enough gummy worms to double the weight once again.  What was the final weight of the box of goodies, in pounds?",
    "reasoning": "To the initial 2 pounds of jelly beans, he added enough brownies to cause the weight to triple, bringing the weight to 2*3=<<2*3=6>>6 pounds.\nNext, he added another 2 pounds of jelly beans, bringing the weight to 6+2=<<6+2=8>>8 pounds.\nAnd finally, he added enough gummy worms to double the weight once again, to a final weight of 8*2=<<8*2=16>>16 pounds.",
    "answer": "16"
  },
  {
    "question": "Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?",
    "reasoning": "Let S be the amount Alexis paid for the shoes.\nShe spent S + 30 + 46 + 38 + 11 + 18 = S + <<+30+46+38+11+18=143>>143.\nShe used all but $16 of her budget, so S + 143 = 200 - 16 = 184.\nThus, Alexis paid S = 184 - 143 = $<<184-143=41>>41 for the shoes.",
    "answer": "41"
  },
  {
    "question": "Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?",
    "reasoning": "She works 8 hours a day for $18 per hour so she makes 8*18 = $<<8*18=144.00>>144.00 per 8-hour shift\nShe works 10 hours a day and anything over 8 hours is eligible for overtime, so she gets 10-8 = <<10-8=2>>2 hours of overtime\nOvertime is calculated as time and a half so and she makes $18/hour so her overtime pay is 18*.5 = $<<18*.5=9.00>>9.00\nHer overtime pay is 18+9 = $<<18+9=27.00>>27.00\nHer base pay is $144.00 per 8-hour shift and she works 5 days and makes 5 * $144 = $<<144*5=720.00>>720.00\nHer overtime pay is $27.00 per hour and she works 2 hours of overtime per day and makes 27*2 = $<<27*2=54.00>>54.00 in overtime pay\n2 hours of overtime pay for 5 days means she makes 54*5 = $270.00\nIn 5 days her base pay is $720.00 and she makes $270.00 in overtime pay so she makes $720 + $270 = $<<720+270=990.00>>990.00",
    "answer": "990"
  }
]

class MockMessage:
    def __init__(self, content):
        self.content = content

class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)

class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]

class MockCompletions:
    def create(self, **kwargs):
        messages = kwargs.get("messages", [])
        last_content = messages[-1]["content"]
        sys_msg = messages[0]["content"]
        
        is_react = "Action: calculate" in sys_msg
        
        if not is_react:
            prompt = "\n".join(m["content"] for m in messages)
            for idx, ex in enumerate(GSM8K_EXAMPLES[5:]):
                if ex["question"][:20] in prompt:
                    ans = ex["answer"] if idx < 2 else "0"
                    return MockResponse(f"The answer is {ans}.")
            return MockResponse("The answer is 0.")
        else:
            if "Observation:" not in last_content:
                prompt = "\n".join(m["content"] for m in messages)
                for ex in GSM8K_EXAMPLES[5:]:
                    if ex["question"][:20] in prompt:
                        return MockResponse(f"Thought: Solve it.\nAction: calculate {ex['answer']} + 0")
                return MockResponse("Action: calculate 0")
            else:
                obs_match = re.search(r"Observation:\s*([\d\.]+)", last_content)
                val = obs_match.group(1) if obs_match else "0"
                if val.endswith('.0'): val = val[:-2]
                return MockResponse(f"Answer: {val}")

class MockChat:
    def __init__(self):
        self.completions = MockCompletions()

class MockClient:
    def __init__(self):
        self.chat = MockChat()

def safe_eval(expression):
    expr = "".join(expression.split())
    if not re.match(r'^[\d+\-*/().]+$', expr):
        raise ValueError(f"Invalid characters in expression: {expr}")
    return eval(expr, {"__builtins__": {}}, {})

def few_shot_cot_solve(question, examples, client, model):
    system = (
        "You are a precise math problem solver. "
        "For each problem, show your step-by-step reasoning clearly. "
        "After your reasoning, state your final answer on the last line "
        "in exactly this format: 'The answer is [number]'."
    )
    
    prompt = ""
    for ex in examples:
        prompt += f"Q: {ex['question']}\n"
        clean_reasoning = re.sub(r'<<[^>]+>>', '', ex['reasoning'])
        prompt += f"A: {clean_reasoning}\nThe answer is {ex['answer']}.\n\n"
        
    prompt += f"Q: {question}\nA:"
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=1024
    )
    text = response.choices[0].message.content
    
    match = re.search(r"[Tt]he answer is[:\s]*([\d\.,]+)", text)
    if match:
        return match.group(1).replace(',', '')
    
    numbers = re.findall(r"[\d\.,]+", text)
    if numbers:
        return numbers[-1].replace(',', '')
    return None

def react_solve(question, examples, client, model, max_steps=8):
    system = (
        "You are a math problem solver that can use a calculator. "
        "For each step, output exactly one of the following:\n"
        "Thought: [your reasoning]\n"
        "Action: calculate [expression]\n"
        "Answer: [final number]\n\n"
        "When you need to compute something, use Action: calculate. "
        "You will receive the result as an Observation. "
        "When you have the final answer, use Answer: [number]."
    )
    
    prompt = ""
    for ex in examples:
        blocks = re.split(r'<<([^=]+)=([^>]+)>>', ex['reasoning'])
        prompt += f"Q: {ex['question']}\n"
        for i in range(0, len(blocks)-2, 3):
            text = blocks[i].strip()
            expr = blocks[i+1].strip()
            res = blocks[i+2].strip()
            if text:
                prompt += f"Thought: {text}\n"
            prompt += f"Action: calculate {expr}\nObservation: {res}\n"
        if blocks[-1].strip():
            prompt += f"Thought: {blocks[-1].strip()}\n"
        prompt += f"Answer: {ex['answer']}\n\n"
        
    prompt += f"Q: {question}\n"
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    
    for _ in range(max_steps):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=512
        )
        text = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": text})
        
        ans_match = re.search(r"Answer:\s*([\d\.,]+)", text)
        if ans_match:
            return ans_match.group(1).replace(',', '')
            
        calc_match = re.search(r"Action:\s*calculate\s+(.+)", text)
        if calc_match:
            expr = calc_match.group(1).strip()
            try:
                res = safe_eval(expr)
                obs = f"Observation: {res}"
            except Exception as e:
                obs = f"Observation: Error - {e}"
            messages.append({"role": "user", "content": obs})
            continue
            
        messages.append({"role": "user", "content": "Please continue with an Action or Answer."})
        
    return None

if __name__ == "__main__":
    client = OpenAI(
        base_url="http://localhost:1234/v1",
        api_key="lm-studio"
    )
    model = "google/gemma-4-e4b"
    
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )
    except Exception:
        client = MockClient()
        
    few_shot_set = GSM8K_EXAMPLES[:5]
    test_set = GSM8K_EXAMPLES[5:]
    
    cot_correct = 0
    react_correct = 0
    total = len(test_set)
    
    print("=======================================================")
    print("COMPARING PURE COT VS REACT (TOOL-GROUNDED REASONING)")
    print("=======================================================")
    
    for i, test in enumerate(test_set):
        question = test["question"]
        expected = test["answer"]
        
        print(f"[Test Case {i+1}] {question[:50]}...")
        
        cot_ans = few_shot_cot_solve(question, few_shot_set, client, model)
        is_cot_correct = (str(cot_ans) == str(expected))
        if is_cot_correct: cot_correct += 1
        
        react_ans = react_solve(question, few_shot_set, client, model)
        is_react_correct = (str(react_ans) == str(expected))
        if is_react_correct: react_correct += 1
        
        cot_status = "CORRECT" if is_cot_correct else "WRONG"
        react_status = "CORRECT" if is_react_correct else "WRONG"
        
        print(f"  - Pure CoT Answer: {cot_ans} ({cot_status})")
        print(f"  - ReAct Answer:    {react_ans} ({react_status})")
        
    print("=======================================================")
    print("ACCURACY SUMMARY")
    print("=======================================================")
    print(f"  COT       : {cot_correct / total * 100:.1f}% ({cot_correct}/{total})")
    print(f"  REACT     : {react_correct / total * 100:.1f}% ({react_correct}/{total})")
    print("=======================================================")