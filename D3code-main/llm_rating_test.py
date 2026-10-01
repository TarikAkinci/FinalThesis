import os
import pandas as pd
from openai import OpenAI


API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj-cg_WOemImDE-O9l3EZh4L8gZp1PjbxOqgb7Gyfc3-n_3KXH3MeZHyg-")

if API_KEY == "sk-proj-cg_WOemImDE-O9l3EZh4L8gZp1PjbxOqgb7Gyfc3-n_3KXH3MeZHyg-E":
    print("WARNING: Please insert your OpenAI API Key into the script before running!")
    exit(1)

client = OpenAI(api_key=API_KEY)

try:
    items_df = pd.read_csv('dataset/d3-items.csv')
    
    sample_items = items_df.sample(5, random_state=42)
    
except FileNotFoundError:
    print("Error: Could not find 'dataset/d3-items.csv'. Make sure you run this script from the root of the project (D3code-main).")
    exit(1)

print("Starting LLM Rating Test for 5 sample items...\n" + "-"*50)


for index, row in sample_items.iterrows():
    text = row['text']
    item_id = row['item_id']
    category = row['category']
    
    # Prepare the prompt
    system_prompt = "You are annotating social media comments for offensiveness."
    user_prompt = f"Rate this comment's offensiveness from 0 (not offensive) to 4 (extremely offensive). Only output the number:\n\n'{text}'"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,  # Deterministic output
            max_tokens=10   # We only expect a single number
        )
        
        rating = response.choices[0].message.content.strip()
        
        print(f"Item ID: {item_id} | Category: {category}")
        print(f"Text: {text}")
        print(f"LLM Rating (0-4): {rating}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Failed to get rating for Item {item_id}. Error: {e}")
