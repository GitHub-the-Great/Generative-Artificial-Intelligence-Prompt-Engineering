import pandas as pd
import os
import time
import google.generativeai as genai
import groq
import json
from concurrent.futures import ThreadPoolExecutor
import re
from datetime import datetime


# Load the CSV files
def load_data():
    sample_df = pd.read_csv('mmlu_sample.csv')
    submit_df = pd.read_csv('mmlu_submit.csv')
    submit_format_df = pd.read_csv('submit_format.csv')
    return sample_df, submit_df, submit_format_df

# Setup API clients
def setup_apis(gemini_api_key, groq_api_key):
    # Setup Gemini
    genai.configure(api_key=gemini_api_key)
    
    # Setup Groq
    groq_client = groq.Client(api_key=groq_api_key)
    
    return groq_client

# Prompt Engineering Techniques

# Comprehensive Prompting Strategy
def comprehensive_prompt(question, options, task, sample_data):
    prompt = "Answer the following multiple-choice question about {task}. You are an expert exam taker. You are an expert in {task}. Using your expertise in {task}, analyze each option carefully and determine the correct answer. Choose only A, B, C, or D.\n\n"

    # Get examples from the same task/subject
    examples = sample_data[sample_data['task'] == task].head(10)
    
    # Add examples
    for _, example in examples.iterrows():
        prompt += f"Question: {example['input']}\n"
        prompt += f"A: {example['A']}\n"
        prompt += f"B: {example['B']}\n"
        prompt += f"C: {example['C']}\n"
        prompt += f"D: {example['D']}\n"
        prompt += f"Answer: {example['target']}\n\n"
    
    # Add our question
    prompt += f"Question: {question}\n"
    prompt += f"A: {options['A']}\n"
    prompt += f"B: {options['B']}\n"
    prompt += f"C: {options['C']}\n"
    prompt += f"D: {options['D']}\n"
    prompt += "Answer:"

    prompt += """Let's think step by step to find the correct answer:
    1. First, understand what the question is asking.
    2. Consider each option carefully.
    3. Eliminate incorrect options.
    4. Determine the most accurate answer.

    Explain your reasoning for each option:
    Option A reasoning:
    Option B reasoning:
    Option C reasoning:
    Option D reasoning:
    
    Write your answer in the following pattern:
    Final answer is: A, B, C or D.
    """
    
    return prompt

# 1. Zero-shot prompting
def zero_shot_prompt(question, options, task):
    prompt = f"""
    Answer the following multiple-choice question. Choose only A, B, C, or D.
    
    Question: {question}
    A: {options['A']}
    B: {options['B']}
    C: {options['C']}
    D: {options['D']}
    
    Answer with just the letter.
    """
    return prompt

#def preprompt(task):
 #   prompt = f"""
  #  You are a contestant in a millionaire quiz, and the next question is the grand finale. If you answer correctly, you will win a million dollars.
   # Now, the host presents the final million-dollar question, which is a single-choice multiple-choice question in {task}. You are an expert in {task}.
    #"""
    #return prompt

# 2. Few-shot learning with examples from the sample data
def few_shot_prompt(question, options, task, sample_data):
    # Get examples from the same task/subject
    examples = sample_data[sample_data['task'] == task].head(3)
    
    prompt = "Answer the following multiple-choice questions. Choose only A, B, C, or D.\n\n"
    
    # Add examples
    for _, example in examples.iterrows():
        prompt += f"Question: {example['input']}\n"
        prompt += f"A: {example['A']}\n"
        prompt += f"B: {example['B']}\n"
        prompt += f"C: {example['C']}\n"
        prompt += f"D: {example['D']}\n"
        prompt += f"Answer: {example['target']}\n\n"
    
    # Add our question
    prompt += f"Question: {question}\n"
    prompt += f"A: {options['A']}\n"
    prompt += f"B: {options['B']}\n"
    prompt += f"C: {options['C']}\n"
    prompt += f"D: {options['D']}\n"
    prompt += "Answer:"
    
    return prompt

# 3. Chain-of-thought reasoning
def cot_prompt(question, options, task):
    prompt = f"""
    Answer the following multiple-choice question about {task}. Choose only A, B, C, or D.
    
    Question: {question}
    A: {options['A']}
    B: {options['B']}
    C: {options['C']}
    D: {options['D']}
    
    Let's think step by step to find the correct answer:
    1. First, understand what the question is asking.
    2. Consider each option carefully.
    3. Eliminate incorrect options.
    4. Determine the most accurate answer.
    
    Final answer (just the letter A, B, C, or D):
    """
    return prompt

# 4. Self-consistency checking
def self_consistency_prompt(question, options):
    prompt = f"""
    You are an expert exam taker. Answer the following multiple-choice question. Choose only A, B, C, or D.
    
    Question: {question}
    A: {options['A']}
    B: {options['B']}
    C: {options['C']}
    D: {options['D']}
    
    First, explain your reasoning for each option:
    Option A reasoning:
    Option B reasoning:
    Option C reasoning:
    Option D reasoning:
    
    Based on the above reasoning, the correct answer is (A, B, C, or D):
    """
    return prompt

# 5. Domain-specific prompt
def domain_specific_prompt(question, options, task):
    task_prompts = {
        "high_school_biology": "You are a high school biology teacher with a talent for explaining biological concepts.",
        "high_school_computer_science": "You are a high school computer science teacher specializing in programming fundamentals.",
        "high_school_european_history": "You are a high school history teacher specializing in European history.",
        "high_school_geography": "You are a high school geography teacher with expertise in physical and human geography.",
        "high_school_government_and_politics": "You are a high school government teacher with knowledge of political systems.",
        "high_school_macroeconomics": "You are a high school economics teacher specializing in macroeconomic principles.",
        "high_school_microeconomics": "You are a high school economics teacher specializing in microeconomic principles.",
        "high_school_psychology": "You are a high school psychology teacher with knowledge of psychological principles.",        
        "high_school_us_history": "You are a high school history teacher specializing in United States history.",
        "high_school_world_history": "You are a high school history teacher with expertise in world history.",
    }
    
    role = task_prompts.get(task.lower().replace(" ", "_"), f"You are an expert in {task}.")
    
    prompt = f"""
    {role}
    
    Answer the following multiple-choice question. Choose only A, B, C, or D.
    
    Question: {question}
    A: {options['A']}
    B: {options['B']}
    C: {options['C']}
    D: {options['D']}
    
    Using your expertise in {task}, analyze each option carefully and determine the correct answer.
    
    The final answer is: 
    """
    return prompt

# API calling functions
def call_gemini_api(prompt, ai_model):
    try:
        model = genai.GenerativeModel(ai_model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        time.sleep(2)  # Back off a bit
        return None

def call_groq_api(groq_client, prompt, ai_model):
    try:
        completion = groq_client.chat.completions.create(
            model=ai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Groq API error: {e}")
        time.sleep(2)  # Back off a bit
        return None

# Parse the response to extract the answer (A, B, C, or D)
def extract_answer(response):
    if not response:
        return None

    # Find all occurrences of A, B, C, or D, including bold formatting (**A**, A)
    matches = re.findall(r'\*\*?([A-D])\*\*?', response)

    # Look for "Answer: A, B, C, or D" pattern, including bold text (**A**)
    answer_pattern = re.search(r'[Aa]nswer[:\s]+(?:\*\*)?([A-D])(?:\*\*)?', response)

    # Look for "**Answer:** A, B, C, or D" (bold Answer)
    bold_answer_pattern = re.search(r'\*\*Answer:\*\*\s*(?:\*\*)?([A-D])(?:\*\*)?', response)

    # 1. Look for fully bolded "Answer: A" (**Answer: A** or **Answer: B**)
    fully_bold_answer_pattern = re.search(r'\*\*Answer:\s*([A-D])\*\*', response)
    
    # Look for direct answer pattern (A, B, C, or D)
    direct_match = re.search(r'\b([A-D])\b', response)
    
    # Look for "The answer is X" pattern
    answer_match = re.search(r'[Aa]nswer(?:\s+is)?[:\s]+([A-D])', response)
    
    # Look for "Option X" pattern
    option_match = re.search(r'[Oo]ption\s+([A-D])', response)
    
    # Look for last occurrence of A, B, C, or D with period or parenthesis
    last_match = re.search(r'([A-D])[\.\)](?!.*[A-D][\.\)])', response)
    
    # Check for any capital letter A, B, C, or D
    any_match = re.search(r'\b([A-D])\b', response)
    
    # Use the first match found in this order of preference
    if last_match:
        return last_match.group(1)
    elif fully_bold_answer_pattern:
        return fully_bold_answer_pattern.group(1)
    elif answer_pattern:
        return answer_pattern.group(1)
    elif bold_answer_pattern:
        return bold_answer_pattern.group(1)
    elif matches:
        return matches[-1]
    elif answer_match:
        return answer_match.group(1)
    elif option_match:
        return option_match.group(1)
    elif direct_match:
        return direct_match.group(1)
    elif any_match:
        return any_match.group(1)
    
    # If no clear match, try to find any A, B, C, or D in the text
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response:
            return letter
            
    # Default to None if no answer can be extracted
    return None

# Ensemble the results from different prompts and models
def get_ensemble_answer(question, options, task, sample_df, groq_client):
    answers = []
    
    # Create different prompts
    prompts = [
        comprehensive_prompt(question, options, task, sample_df),
        zero_shot_prompt(question, options, task),
        few_shot_prompt(question, options, task, sample_df),
        cot_prompt(question, options, task),
        self_consistency_prompt(question, options),
        domain_specific_prompt(question, options, task)
    ]
    
    # Call Gemini API for each prompt
    AI_models = [
        'gemini-2.0-flash',
        #'gemini-2.0-flash-lite',
        #'gemini-2.0-pro-exp-02-05',
        #'gemini-2.0-flash-thinking-exp-01-21',
        'gemini-2.0-flash-exp',
        # 'learnlm-1.5-pro-experimental',
        # 'gemini-1.5-pro',
        'gemini-1.5-flash',
        'gemini-1.5-flash-8b',
        #'gemma-3-27b-it'
        #'gemma-2-2b-it',
        #'gemma-2-9b-it',
        #'gemma-2-27b-it'
    ]
    for ai_model in AI_models[:4]:
        for prompt in prompts[:1]:
            # time.sleep(4)
            # response = call_gemini_api(preprompt(task), ai_model)
            # print("Pre ", end=" ")
            response = call_gemini_api(prompt, ai_model)
            print(response)
            answer = extract_answer(response)
            print(f"extract_answer = {answer}")
            # print(f"{answer} of {ai_model}.")
            if answer:
                answers.append(answer)

    # print("\n")
    # Call Groq API for each prompt
    AI_models = [
        "allam-2-7b", 
        "deepseek-r1-distill-llama-70b", #
        "deepseek-r1-distill-qwen-32b", #
        "gemma2-9b-it",
        "llama-3.1-8b-instant",
        "llama-3.2-11b-vision-preview",
        "llama-3.2-1b-preview", #
        "llama-3.2-3b-preview",
        "llama-3.2-90b-vision-preview", #
        "llama-3.3-70b-specdec", #
        "llama-3.3-70b-versatile", #
        "llama-guard-3-8b", #
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mistral-saba-24b", #
        "mixtral-8x7b-32768",
        "qwen-2.5-32b",
        "qwen-2.5-coder-32b",
        "qwen-qwq-32b" #
    ]
    for ai_model in AI_models[:0]:
        for prompt in prompts[:0]:  # Use fewer prompts for Groq to manage rate limits
            response = call_groq_api(groq_client, prompt, ai_model)
            answer = extract_answer(response)
            print(f"extract_answer = {answer}")
            if answer:
                answers.append(answer)
    
    # If we have answers, return the most common one
    if answers:
        # Get the most common answer
        answer_counts = {}
        for ans in answers:
            answer_counts[ans] = answer_counts.get(ans, 0) + 1
        
        # Sort by count (descending)
        sorted_answers = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_answers[0][0]
    
    # Default to 'A' if no answers could be extracted
    return 'A'

# Process all questions and generate answers
def process_questions(submit_df, sample_df, groq_client):
    results = []
    
    for index, row in submit_df.iterrows():
        question_id = row['Unnamed: 0']
        question = row['input']
        options = {
            'A': row['A'],
            'B': row['B'],
            'C': row['C'],
            'D': row['D']
        }
        task = row['task']

        start_time = datetime.now()  # 記錄起始時間
        # Get ensemble answer
        answer = get_ensemble_answer(question, options, task, sample_df, groq_client)
        end_time = datetime.now()  # 記錄結束時間
        elapsed_time = end_time - start_time  # 計算執行時間

        print(f"final {answer} 執行時間: {elapsed_time}", end=" ")
        
        results.append({
            'ID': question_id,
            'target': answer
        })
        
        # Print progress
        ## if (index + 1) % 10 == 0:
        print(f"Processed {index + 1}/{len(submit_df)} questions")
        
        # Add small delay to avoid rate limits
        time.sleep(1)
    
    return pd.DataFrame(results)

# Main function
def main():
    # Load your API keys from environment variables
    gemini_api_key = os.environ.get('GEMINI_API_KEY')
    groq_api_key = os.environ.get('GROQ_API_KEY')
    
    if not gemini_api_key or not groq_api_key:
        print("Please set GEMINI_API_KEY and GROQ_API_KEY environment variables")
        return
    
    # Load data
    print("Load data...")
    sample_df, submit_df, submit_format_df = load_data()
    print("Load data done.")
    
    # Setup APIs
    print("Setup APIs...")
    groq_client = setup_apis(gemini_api_key, groq_api_key)
    print("Setup APIs done.")
    
    # Process questions and get results
    print("Process questions and get results...")
    results_df = process_questions(submit_df, sample_df, groq_client)
    print("Process questions and get results done.")
    
    # Format results according to submission format
    print("Format results according to submission format...")
    submission_df = pd.DataFrame({
        'ID': results_df['ID'],
        'target': results_df['target']
    })
    print("Format results according to submission format done.")
    
    # Save results
    print("Save results...")
    submission_df.to_csv('mmlu_submission.csv', index=False)
    print("Submission file generated: mmlu_submission.csv")

if __name__ == "__main__":
    main()


