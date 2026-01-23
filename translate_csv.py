from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from accelerate import infer_auto_device_map

# Load Qwen model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()

# CUDA_VISIBLE_DEVICES=0
# Function to translate text
def translate_text(text):
    if pd.isna(text) or text.strip() == "":
        return text  # Skip empty or NaN cells
    prompt = f"Translate the following Chinese text to English: {text}"
    response, _ = model.chat(tokenizer, prompt, history=None)
    return response

# Load the Excel file
file_path = "Copy of USimage_match_sample_patient.xlsx"
df = pd.read_excel(file_path)

# Translate column headers
translated_columns = {col: translate_text(col) for col in df.columns}
df = df.rename(columns=translated_columns)

# Translate cell values
translated_df = df.applymap(translate_text)

# Save the translated DataFrame to a new Excel file
output_path = "Translated_USimage_match_sample_patient_Qwen.xlsx"
translated_df.to_excel(output_path, index=False)

print(f"Translation complete. Translated file saved to: {output_path}")
