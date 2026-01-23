import pandas as pd

# Load the CSV or Excel file
file_path = "Copy of USimage_match_sample_patient.xlsx"  # Replace with your file path
df = pd.read_excel(file_path)

# Relevant columns
relevant_columns = ["临床诊断", "检查提示", "检查所见", "左室射血分数"]

# Function to construct prompts
def construct_prompt(row):
    patient_data = []

    # Dynamically add attributes if they are not NaN or Unknown
    if pd.notna(row.get("临床诊断")) and row["临床诊断"] != "Unknown":
        patient_data.append(f"- Clinical Diagnosis: {row['临床诊断']}")
    if pd.notna(row.get("检查提示")) and row["检查提示"] != "Unknown":
        patient_data.append(f"- Examination Notes: {row['检查提示']}")
    if pd.notna(row.get("检查所见")) and row["检查所见"] != "Unknown":
        patient_data.append(f"- Examination Findings: {row['检查所见']}")
    if pd.notna(row.get("左室射血分数")) and row["左室射血分数"] != "Unknown":
        patient_data.append(f"- Left Ventricular Ejection Fraction: {row['左室射血分数']}%")
    if pd.notna(row.get("主动脉")) and row["主动脉"] != "Unknown":
        patient_data.append(f"- Aorta Diameter: {row['主动脉']} mm")
    if pd.notna(row.get("升主动脉")) and row["升主动脉"] != "Unknown":
        patient_data.append(f"- Ascending Aorta Diameter: {row['升主动脉']} mm")
    if pd.notna(row.get("左房")) and row["左房"] != "Unknown":
        patient_data.append(f"- Left Atrium Diameter: {row['左房']} mm")
    if pd.notna(row.get("左室舒张末")) and row["左室舒张末"] != "Unknown":
        patient_data.append(f"- Left Ventricular End-Diastole Diameter: {row['左室舒张末']} mm")
    if pd.notna(row.get("组织多普勒A")) and row["组织多普勒A"] != "Unknown":
        patient_data.append(f"- Tissue Doppler A: {row['组织多普勒A']} cm/s")
    if pd.notna(row.get("左室舒张功能E/E")) and row["左室舒张功能E/E"] != "Unknown":
        patient_data.append(f"- E/E Ratio: {row['左室舒张功能E/E']}")
    if pd.notna(row.get("三尖瓣E")) and row["三尖瓣E"] != "Unknown":
        patient_data.append(f"- Tricuspid Valve E: {row['三尖瓣E']} cm/s")
    if pd.notna(row.get("三尖瓣A")) and row["三尖瓣A"] != "Unknown":
        patient_data.append(f"- Tricuspid Valve A: {row['三尖瓣A']} cm/s")
    if pd.notna(row.get("二尖瓣PHT")) and row["二尖瓣PHT"] != "Unknown":
        patient_data.append(f"- Mitral Valve PHT: {row['二尖瓣PHT']} ms")
    if pd.notna(row.get("右侧内中膜厚度")) and row["右侧内中膜厚度"] != "Unknown":
        patient_data.append(f"- Right Inner Membrane Thickness: {row['右侧内中膜厚度']} mm")
    if pd.notna(row.get("右室壁厚度")) and row["右室壁厚度"] != "Unknown":
        patient_data.append(f"- Right Ventricular Wall Thickness: {row['右室壁厚度']} mm")
    if pd.notna(row.get("右室FAC")) and row["右室FAC"] != "Unknown":
        patient_data.append(f"- Right Ventricular Fractional Area Change: {row['右室FAC']}%")

    # Construct the prompt using the filtered patient data
    patient_data_str = "\n".join(patient_data)
    prompt = f"""
    You are a highly knowledgeable medical assistant. Based on the patient's echocardiographic and clinical data provided below, generate diverse and detailed Visual Question Answering (VQA) pairs.

    Patient Data:
    {patient_data_str}

    Instructions:
    Generate 10 detailed questions and answers related to this data, focusing on the clinical diagnosis, measurements, and their implications for the patient's cardiac health. Ensure that the questions are diverse and meaningful. In Json file to train VQA and instruction tuning model with it.
    """
    return prompt.strip()

    # Construct the prompt

# Generate prompts for all rows
for i, row in df.iterrows():
    prompts=[construct_prompt(row)]

    # Save prompts to a text file for local use
    output_file = f"prompt/Patient_VQA_Prompts_{i}.txt"
    with open(output_file, "w") as f:
        for i, prompt in enumerate(prompts):
            f.write(f"\n{prompt}\n\n")

    print(f"Prompts have been saved to {output_file}.")
