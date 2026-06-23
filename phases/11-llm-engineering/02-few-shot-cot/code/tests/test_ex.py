# phases/11-llm-engineering/02-few-shot-cot/code/tests/test_ex.py
# Reference: Lesson 11.02 docs/en.md

import os
import sys
import unittest

# Adjust path to import from the parent code directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ex import (
    extract_answer,
    build_few_shot_cot_prompt,
    build_react_prompt,
    safe_eval,
    few_shot_cot_solve,
    react_solve,
    MockOpenAI,
    GSM8K_EXAMPLES
)

class TestFewShotCoTAndReAct(unittest.TestCase):
    
    def test_extract_answer(self):
        self.assertEqual(extract_answer("The answer is 72"), "72")
        self.assertEqual(extract_answer("The answer is: 10.5"), "10.5")
        self.assertEqual(extract_answer("Answer: 10"), "10")
        self.assertEqual(extract_answer("#### 42"), "42")
        self.assertEqual(extract_answer("Therefore, x = 3.14"), "3.14")
        self.assertEqual(extract_answer("Some text with no pattern but ends with 123"), "123")
        self.assertIsNone(extract_answer("No numbers here"))
        
    def test_build_few_shot_cot_prompt(self):
        examples = GSM8K_EXAMPLES[:2]
        system, user = build_few_shot_cot_prompt("How much is 1+1?", examples)
        
        self.assertIn("precise math problem solver", system)
        self.assertIn("Show your step-by-step reasoning", system)
        self.assertIn("How much is 1+1?", user)
        self.assertIn(examples[0]["question"], user)
        self.assertIn("Let's think step by step.", user)
        
    def test_build_react_prompt(self):
        system, user = build_react_prompt("How much is 1+1?")
        
        self.assertIn("alternates between thoughts and actions", system)
        self.assertIn("calculate", system)
        self.assertIn("Observation:", system)
        self.assertIn("How much is 1+1?", user)
        
    def test_safe_eval_success(self):
        self.assertEqual(safe_eval("3 * 2"), 6)
        self.assertEqual(safe_eval("100 - 50 - 30"), 20)
        self.assertEqual(safe_eval("48 / (2 + 2)"), 12.0)
        self.assertEqual(safe_eval(" 0.2 * 50 "), 10.0)
        
    def test_safe_eval_unsafe(self):
        # Alphabetic characters (names, variables, imports) should be banned
        with self.assertRaises(ValueError):
            safe_eval("x + 1")
        with self.assertRaises(ValueError):
            safe_eval("__import__('os').system('ls')")
        with self.assertRaises(ValueError):
            safe_eval("print(1)")
        with self.assertRaises(ValueError):
            safe_eval("1; import sys")
            
    def test_few_shot_cot_solve_mock(self):
        mock_client = MockOpenAI()
        few_shot_examples = GSM8K_EXAMPLES[:2]
        # Test Case: Mark has a garden (index 5)
        question = GSM8K_EXAMPLES[5]["question"]
        expected = GSM8K_EXAMPLES[5]["answer"]
        
        ans, trace = few_shot_cot_solve(question, few_shot_examples, mock_client, "mock-model")
        self.assertEqual(ans, expected)
        self.assertIn("Let's think step by step", trace)
        
    def test_react_solve_mock(self):
        mock_client = MockOpenAI()
        # Test Case: Albert is wondering (index 6)
        question = GSM8K_EXAMPLES[6]["question"]
        expected = GSM8K_EXAMPLES[6]["answer"]
        
        ans, trace = react_solve(question, mock_client, "mock-model")
        self.assertEqual(ans, expected)
        self.assertIn("Action: calculate 2 * 16", trace)
        self.assertIn("Observation: 32", trace)
        self.assertIn("Answer: 48", trace)

if __name__ == "__main__":
    unittest.main()
