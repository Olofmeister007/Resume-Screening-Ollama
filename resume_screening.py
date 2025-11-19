import os
import re
import pandas as pd
from PyPDF2 import PdfReader
from ollama import Client


MODEL_NAME = "deepseek-r1:1.5b"   # Make sure this model is installed via Ollama
RESUME_FOLDER = "resumes/"        # Folder where PDF resumes are stored
OUTPUT_FILE = "candidate_rankings.csv"


job_requirements = """
- Bachelor’s or Associate’s degree preferred, not required.
- Strong communication and interpersonal skills.
- Willingness to learn and grow in a team environment.
- Problem-solving mindset and proactive attitude.
- Ability to adapt to different customer needs.
"""

# ============ PDF TEXT EXTRACTION ============
def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

# ============ MODEL SETUP ============
client = Client()

def rank_candidate(resume_text, requirements_text):
    prompt = f"""
Compare the following resume to the job requirements.
Rate the candidate from 1 (poor match) to 10 (perfect match).
Then provide a brief explanation of why you gave that score.

Resume:
{resume_text}

Requirements:
{requirements_text}

Output in the following format:
Score: X/10
Reason: <short explanation>
"""
    try:
        response = client.chat(model=MODEL_NAME, messages=[
            {"role": "user", "content": prompt}
        ])
        return response['message']['content']
    except Exception as e:
        print(f"Error ranking candidate: {e}")
        return "Score: 0/10\nReason: Error during processing."

# ============ BULK PROCESSING ============
def process_resumes():
    results = []

    for filename in os.listdir(RESUME_FOLDER):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(RESUME_FOLDER, filename)
        print(f"Processing: {filename}")

        text = extract_text_from_pdf(file_path)
        if not text:
            continue

        response = rank_candidate(text, job_requirements)

        # Extract numeric score from model response
        match = re.search(r"Score:\s*(\d+)/10", response)
        score = int(match.group(1)) if match else 0

        results.append({
            "Candidate": filename,
            "Score": score,
            "Details": response
        })

    # Sort candidates by score (descending)
    df = pd.DataFrame(results)
    df = df.sort_values(by="Score", ascending=False)

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Results saved to: {OUTPUT_FILE}")
    print(df.head())

# ============ MAIN ============
if __name__ == "__main__":
    # process_resumes()
    text = extract_text_from_pdf("resumes/lenora.pdf")
    print(text)



