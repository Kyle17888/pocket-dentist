import string

translate_dent_json_zh2en = string.Template("""\
You are a professional dentist. Below you will be given a dictionary. 
Please translate all the values in the dictionary into English, ensuring that the translation is consistent with dental terminology. 
Finally, keep the output in json format, and do not output any additional content!

[Input]:
```json
$case
``` 

[Output]:
Must be output in json format.
""")

label_desc_zh = {
    "C1": {
        "name": "龋",
        "note": "不包括早期的白斑样病变"
    },
    "C2": {
        "name": "非龋性未修复的牙体缺损",
        "note": "非龋原因未修复导致的牙折裂或牙齿颈部出现的缺损，包括楔状缺损或凹陷，不包括生理或者病理性磨耗"
    }, 
    "C3": {
        "name": "牙磨耗或酸蚀",
        "note": "因生理或病理性原因导致的牙体表面的磨耗，以及酸蚀症，不包括因龋坏导致的缺损"
    },
    "C4": {
        "name": "牙龈红肿", 
        "note": "可能伴或者不伴出血，也可能有或者无牙槽骨吸收"
    },
    "C5": {
        "name": "牙龈退缩",
        "note": "由于生理或病理原因导致的牙龈退缩，露出牙根面，或者明显的黑三角（邻间隙牙龈退缩）"
    },
    "C6": {
        "name": "牙菌斑或牙结石",
        "note": "可见的牙菌斑或牙结石堆积，不包括个别食物残渣等情况。"
    },
    "C7": {
        "name": "牙体颜色异常",
        "note": "因染色、氟斑牙、死髓牙等原因导致的牙体颜色异常，或者牙釉质脱矿导致的白垩斑，不包括因龋坏导致的颜色变化"
    }, 
    "C8": {
        "name": "牙列缺损",
        "note": "缺失一颗或多颗牙齿，未进行修复"
    },
    "C9": {
        "name": "残根",
        "note": "牙冠缺失，仅存牙根部分"
    },
    "C10": {
        "name": "充填物",
        "note": "包括牙齿表面的各种充填材料，如树脂、银汞合金等，或者是临时充填、牙胶"
    },
    "C11": {
        "name": "固定修复体",
        "note": "包括牙冠、桥体、贴面、嵌体等修复体"
    },
    "C12": {
        "name": "活动义齿",
        "note": "包括局部义齿和全口义齿"
    },
    "C13": {
        "name": "牙列间隙", 
        "note": "无缺牙但存在牙间间隙，可能单纯牙间隙过大，或者为生理性间隙。不包括邻牙有接触时牙龈退缩导致的黑三角"
    }, 
    "C14": {
        "name": "牙列不齐或错颌畸形",
        "note": "包括个别或者整体牙扭转、拥挤、移位等表现，有矫治器或者正畸附件并不代表一定存在牙列不齐"
    },  
    "C15": {
        "name": "正畸传统矫治器",
        "note": "包括托槽、钢丝、橡皮圈等传统正畸矫治器材"
    },
    "C16": {
        "name": "正畸隐形矫治器",
        "note": "包括隐形牙套、附件、保持器等隐形正畸矫治器材"
    },
    "C17": {
        "name": "口腔溃疡",
        "note": "包括复发性口腔溃疡、创伤性溃疡等，不包括牙周问题导致的牙龈红肿"
    }, 
    "C18": {
        "name": "伤口",
        "note": "包括拔牙创口、外伤或手术导致的口腔组织伤口等，不包括牙龈炎导致的红肿出血"
    }
}

label_desc_en = {
    "C1": {
        "name": "Dental caries",
        "note": "Clearly visible dental caries; early white-spot lesions are excluded."
    },
    "C2": {
        "name": "Non-carious, unrestored tooth defect",
        "note": "Refers to tooth fractures or cervical defects not caused by caries and not yet restored (e.g. wedge-shaped defects or notching). Excludes physiological or pathological tooth wear."
    },
    "C3": {
        "name": "Tooth wear or erosion",
        "note": "Loss of tooth structure due to physiological or pathological wear, or erosion. Defects caused by caries or minor enamel cracks are excluded."
    },
    "C4": {
        "name": "Gingival inflammation",
        "note": "Gingival redness and swelling, may present with or without bleeding, and may or may not be accompanied by alveolar bone resorption."
    },
    "C5": {
        "name": "Gingival recession",
        "note": "Recession of the gingival margin due to physiological or pathological causes, resulting in root exposure or visible black triangles (interdental gingival recession)."
    },
    "C6": {
        "name": "Dental plaque or calculus",
        "note": "Visible accumulation of plaque or calculus. Excludes occasional food debris."
    },
    "C7": {
        "name": "Tooth discoloration",
        "note": "Abnormal tooth color caused by staining, fluorosis, or pulp necrosis, as well as chalky white spots due to enamel demineralization. Excludes dark discoloration due to caries."
    },
    "C8": {
        "name": "Partial edentulism",
        "note": "One or more missing teeth with no residual roots present and no prosthetic replacement."
    },
    "C9": {
        "name": "Residual root",
        "note": "Complete loss of the clinical crown, with only the root portion remaining."
    },
    "C10": {
        "name": "Dental filling (direct filling)",
        "note": "Includes various direct restorative materials on tooth surfaces, such as composite resin, amalgam, temporary fillings, or gutta-percha."
    },
    "C11": {
        "name": "Fixed prosthesis",
        "note": "Includes crowns, bridges, veneers, inlays, and other fixed dental prostheses."
    },
    "C12": {
        "name": "Removable denture",
        "note": "Includes partial and complete removable dentures."
    },
    "C13": {
        "name": "Interdental spacing",
        "note": "Presence of spaces between teeth without missing teeth, possibly due to diastema or physiological spacing. Excludes black triangles caused by gingival recession when adjacent teeth are in contact."
    },
    "C14": {
        "name": "Malocclusion or dental malalignment",
        "note": "Includes individual or generalized tooth rotation, crowding, or displacement. The presence of orthodontic appliances does not necessarily indicate malalignment."
    },
    "C15": {
        "name": "Conventional orthodontic appliance",
        "note": "Includes brackets, archwires, elastics, and other conventional orthodontic materials."
    },
    "C16": {
        "name": "Clear aligner orthodontic appliance",
        "note": "Includes clear aligners, attachments, retainers, and other components of invisible orthodontic systems."
    },
    "C17": {
        "name": "Oral ulcer",
        "note": "Includes recurrent aphthous ulcers and traumatic ulcers. Excludes gingival redness or swelling caused by periodontal inflammation."
    },
    "C18": {
        "name": "Oral wound",
        "note": "Includes extraction sockets, trauma-related wounds, or surgical wounds of the oral tissues. Excludes gingival redness or bleeding caused by gingivitis."
    }
}


classify_intraoral_condition = string.Template("""\
You are a professional dentist. Now you have some descriptive diagnostic texts about patients in JSON format. 
Please perform multi-class category extraction based on these texts. 
The categories and some extra information are as follows:

[Categories]:
$label_desc

Below is the input JSON:
[Input]:
```json
$case
```

Your output should be a JSON array, where each element is a dictionary containing the following keys:
- "id": The category ID (e.g., "C1", "C2", etc.)
- "name": The category name
- "evidence": The evidence text from the input that supports the classification into this category.

You can do some basic inference based on the input text to provide the evidence. 
But if the input does not contain information related to a specific category, do not include that category in the output.
Additionally, if low_confidence is True, do not include this annotation in the multi-class extraction.

[Output Template]:
```json
[
    <fill in the extracted categories as specified above>
]
```
""")

classify_intraoral_condition_for_image = string.Template("""\
You are a professional dentist. You are now given a clinical image of a patient.
Please perform multi-class category extraction based on this dental clinical image.

The categories and additional information are as follows:
[Categories]:
$label_desc

Your output should be a JSON array, where each element is a dictionary containing the following keys:
- "id": The category ID (e.g., "C1", "C2", etc.)
- "name": The category name
- "evidence": The evidence or visual cues observed in the image that support the classification into this category.

Important requirements:
- Only select categories that are visibly present in the image. Do not select or provide explanations for categories that cannot be seen.
- The "id" and "name" must strictly match and correspond to the given categories.
- You must only choose from the listed categories.
- It is acceptable to output an empty array if no categories apply.

[Output Template]:
```json
[
    <fill in the extracted categories as specified above>
]
```
""")

summary_intraoral_condition = string.Template("""\
You are a professional dentist. You are now given descriptive diagnostic texts in JSON format about a patient’s intraoral image.

Based on these texts, please generate a detailed description of the image content. 
Please note: 
- You are can be imaginative, assuming non-mentioned common abnormalities are normal for that region, you are encouraged to describe about these deduced normal content as well. 
- While you are encouraged to be imaginative, use your imagination with reasoning based on provided content, ensuring that all your output is not against the input. 
- You must respond in (un-structured) natural language, instead of structured formats like bullet points or numbered lists. 
- Adjust your tongue as if you are seeing the image instead of respond based on the text input. 

Below is the input JSON:
[Input]:
```json
$case
```

Your output should be a JSON object with the following keys:
- "description": A comprehensive description generated from the given texts. 

[Output Template]:
```json
{
    "description": "<fill in the detailed description including observed content, dental instruments, and abnormal findings>"
}
```
Please output json directly without extra output.
""")

captioning_intraoral_condition = string.Template("""\
You are a professional dentist. You are now given a clinical image of a patient. Please generate a detailed and vivid natural language description based on this image.

Your output should be a JSON object with one key, "description". The description must be written as a coherent paragraph, not as a list or dictionary. It should clearly and naturally describe the imaging direction (the angle or orientation from which the image was captured), the main subject of the image (the primary anatomical focus or structure), and all observed abnormalities (including pathological findings, dental defects, or visible dental instruments related to these abnormalities). You may also describe regions that appear normal if you are confident in their correctness, but your descriptions must remain accurate and factual, without any fabricated or speculative details. Use multiple sentences as needed to make the description fluent, expressive, and clinically meaningful.

[Output Template]:
```json
{
    "description": "<fill in the detailed description including observed content, dental instruments, and abnormal findings>"
}
```
Please output json directly without extra output.
""")

captioning_extraction_intraoral_condition = string.Template("""\
You are a professional dentist. You are now given descriptive diagnostic texts in JSON format about a patient’s intraoral image.

Based on the given texts, please list all observed abnormalities and present them in a list format.

Below is the input JSON:
[Input]:
```json
$case
```

Definition of abnormality: Any pathological findings, dental defects, or visible dental instruments associated with abnormalities. Do not describe normal findings.

Your output should be a JSON array, where each element is a dictionary containing the following keys:
- "abnormality": A summarized description of the abnormality based on the given texts
- "reason": The reasoning and supporting evidence for this abnormality description

[Output Template]:
```json
[
    <fill in the extracted abnormalities as specified above>
]
```
""")

captioning_score_intraoral_condition = string.Template("""\
You are a professional dentist. You are now given two sets of information in JSON format about a patient’s intraoral image: Reference – the ground truth abnormality descriptions. Prediction – the abnormality descriptions generated by an LLM.

Below is the Reference JSON input:
[Reference]:
```json
$reference
```

Below is the Prediction JSON input:
[Prediction]:
```json
$prediction
```

Definition of abnormality: Any pathological findings, dental defects, or visible dental instruments associated with abnormalities.

Based on these two inputs, please calculate the values for the confusion matrix: True Positive (TP), False Negative (FN), False Positive (FP), and True Negative (TN).

Your output should be a JSON object with the following keys:
- "TP": integer value
- "FN": integer value
- "FP": integer value
- "TN": integer value
- "reason": the reasoning and supporting evidence for these values

[Output Template]:
```json
{
    <fill in the confusion matrix values as specified above>
}
```
""")

vqa_intraoral_condition = string.Template("""\
You are a professional dentist. Now you are given some descriptive diagnostic texts about a patient's intraoral image JSON format.
Please generate visual question answering (VQA) questions based on these texts.

The question types include:
- **$multiple_choice** single-choice questions (each question has four options, A-D, with only one correct answer)
- **$true_false** true/false questions (each question has two options, A-B, with only one correct answer)

[Requirements]:
```
1. All questions must have exactly one correct answer, and the correct answer should be randomly distributed among the options across questions. 
2. Generate questions only for the content of the image. Do not include any treatment suggestions, or basic dental knowledge. 
3. Be professional about the wording, carefully review before answering to avoid incorrect dental terminology or descriptions.
4. Do not include any question regarding the uncertain content (marked with low_confidence is True in the input text).
5. The questions must be concise and precise, avoiding vague or ambiguous wording. No need to mention "based on the description" or "The description mentions"...
6. All questions should be generated strictly within the region described in the given text. Any findings not mentioned in the description should be considered normal within that region and may be used to construct questions and answers. For non-existent abnormality questions, you may choose from conditions such as dental caries, non-carious tooth defects, tooth wear or erosion, gingival redness and swelling, gingival recession, dental plaque or calculus, tooth discoloration, dentition defects, residual roots, restorations, fixed prostheses, removable dentures, interdental spacing, dental crowding or malocclusion, traditional orthodontic appliances, clear aligners, or oral ulcers. You may also generate questions based on your other dental knowledge, as long as the content remains consistent with the described region. Any abnormal findings not provided in the text should be treated as normal for that region. We require a diverse set of questions. Options may include 'All of the above', 'None of the above' or 'Unknown'.
7. Single-choice questions and true/false questions must test different knowledge points (they cannot be the same or similar). If there is limited abnormal information, you may design questions based on point 6.
8. The output must be in valid JSON array, without any extra content.
```

Note:
When describing tooth positions, the FDI notation is used by default. Sometimes the # symbol is omitted in the description. For example, "11" refers to "Upper Right Central Incisor," and "18" refers to "Upper Right Third Molar.". For ranges, e.g., #12-#23 means "Upper Right Lateral Incisor to Upper Left Canine," which includes #12, #11, #21, #22, and #23. 
**FDI Tooth Numbering System**:
| Upper Right | Upper Left |
| Lower Right | Lower Left |

```Permanent tooth
| #18, #17, #16, #15, #14, #13, #12, #11 | #21, #22, #23, #24, #25, #26, #27, #28 |  
| #48, #47, #46, #45, #44, #43, #42, #41 | #31, #32, #33, #34, #35, #36, #37, #38 |
```

```Deciduous tooth
| #55, #54, #53, #52, #51 | #61, #62, #63, #64, #65 |
| #85, #84, #83, #82, #81 | #71, #72, #73, #74, #75 |
```

Below is the input JSON:
[Input]:
```json
$case
```

Your output should be a JSON array, where each element is a dictionary containing the following keys:
- "question_type": The question type ID (for single-choice: multiple_choice; for true/false: judge)
- "question": The question text
- "choice": The options, formatted as a dictionary where the key is the option label (A–D for single-choice, A–B for true/false) and the value is the corresponding option text
- "answer": The correct answer (A-D for single-choice; A-B for true/false)
- "reason": The rationale and supporting evidence for why the correct answer is chosen

[Output Template for Each question]:
{
    "question_type": <fill in the question type as specified above>,
    "question": <fill in the question text as specified above>,
    "choice": <fill in the options as specified above>,
    "answer": <fill in the correct answer as specified above>,
    "reason": <fill in the rationale and supporting evidence as specified above>
}

[Output Template]:
```json
[
    <fill in the generated questions as specified above>
]
```
""")

vqa_answer_intraoral_condition = string.Template("""\
You are a professional dentist. You are now presented with a clinical image of a patient and a multiple-choice question.
Please select only one correct answer based on the visual evidence from the image.

Below is the question:
[Question]:
```json
{
    "question": $question,
    "choice": $choice
}
```

Your output should be a JSON object with the following keys:
- "answer": Your selected option, represented as one of $answer_options.
- "reason": The reasoning and supporting visual evidence for your chosen answer.

Do not include any additional explanations, text, or formatting outside the JSON.

[Output Template]:
```json
{
    "answer": <fill in the selected answer as specified above>,
    "reason": <fill in the reasoning and supporting evidence as specified above>
}
```
""")

verifier_intraoral_condition = string.Template("""\

You are a professional dentist with expertise in oral diagnostics and imaging. You will be provided with two inputs in JSON format:
- A descriptive diagnostic text detailing clinical findings from a patient’s intraoral images.
- An AI-generated Visual Question Answering (VQA) set, consisting of questions and answers about the same images.
Your task is to evaluate the consistency of the AI-generated VQA content against the diagnostic text and revise it as necessary to ensure full alignment.

If you determine that the current VQA does not match the texts, you must revise the VQA so that it satisfies the requirements. 
Note that: 
1. The question must focus on the region described in the text or using the region inferred from the description. For example, an image showing only the upper jaw should not have questions about the lower jaw (or the answer should be "Unknown" if the question is about the lower jaw, or the question should be about the visibility of the lower jaw). 
2. Any abnormalities/disease not mentioned in the text (but within the described region) are assumed absent, constructing questions based on such deduced non-existent abnormalities is acceptable.
3. If any item in the diagnostic text is marked with low_confidence: true, it must not be used to generate or support any question or answer. Any VQA content relying on such uncertain observations must be revised or removed.
4. Be conservative in your judgement and revisions. If you are unsure, it is better to mark the VQA as valid. If the VQA content is largely correct with only minor issues, make minimal necessary changes to ensure accuracy and alignment with the text.
5. When describing tooth positions, the FDI notation is used by default. Sometimes the # symbol is omitted in the description. For example, "11" refers to "Upper Right Central Incisor," and "18" refers to "Upper Right Third Molar.". Interpret tooth ranges as inclusive sequences: ranges spanning opposite quadrants (e.g., 12–22 or 22–12) include all teeth crossing the midline (12,11,21,22); ranges within the same quadrant (e.g., 21–25) include sequential teeth in that quadrant (21,22,23,24,25).

**FDI Tooth Numbering System**:
| Upper Right | Upper Left |
| Lower Right | Lower Left |

```Permanent tooth
| #18, #17, #16, #15, #14, #13, #12, #11 | #21, #22, #23, #24, #25, #26, #27, #28 |  
| #48, #47, #46, #45, #44, #43, #42, #41 | #31, #32, #33, #34, #35, #36, #37, #38 |
```

```Deciduous tooth
| #55, #54, #53, #52, #51 | #61, #62, #63, #64, #65 |
| #85, #84, #83, #82, #81 | #71, #72, #73, #74, #75 |
```

Below is the patient's descriptive diagnostic text:
[Diagnostic Text Input]:
```json
$case
```

Below is the AI-generated VQA content:
[AI Input]:
```json
$ai_input
```

Your output should be a JSON object with the following keys:
- "invalid": Whether there is an error with the current VQA content. (true/false)
- "error_type": One of the following error categories: 
    * multiple answers present
    * using uncertain information
    * incorrect answer
    * common knowledge error
    * original annotation insufficient detail
    * out-of-region question
    * incorrect tooth position
    * hallucination (error within the described region)
    * other
    * null (if "invalid" is false)
- "evidence": The evidence text from the input that supports your judgment.
- "new_question": A dictionary containing the corrected VQA data. This field is required only when "invalid" is true. 

Note on construction of the revised question if new_question is needed (besides the above criteria): 
The question should be a JSON array, where each element is a dictionary containing the following keys:
- "question_type": The question type ID (for single-choice: multiple_choice; for true/false: judge)
- "question": The question text
- "choice": The options, formatted as a dictionary where the key is the option label (A–D for single-choice, A–B for true/false) and the value is the corresponding option text
- "answer": The correct answer (A-D for single-choice; A-B for true/false)
- "reason": The rationale and supporting evidence for why the correct answer is chosen
1. Generate questions only for the content of the image. Do not include any treatment suggestions, or basic dental knowledge. 
2. Be professional about the wording, carefully review before answering to avoid incorrect dental terminology or descriptions.
3. The questions must be concise and precise, avoiding vague or ambiguous wording. No need to mention "based on the description" or "The description mentions"...
4. All questions should be generated strictly within the region described in the given text. Any findings not mentioned in the description should be considered normal within that region and may be used to construct questions and answers. For non-existent abnormality questions, you may choose from conditions such as dental caries, non-carious tooth defects, tooth wear or erosion, gingival redness and swelling, gingival recession, dental plaque or calculus, tooth discoloration, dentition defects, residual roots, restorations, fixed prostheses, removable dentures, interdental spacing, dental crowding or malocclusion, traditional orthodontic appliances, clear aligners, or oral ulcers. You may also generate questions based on your other dental knowledge, as long as the content remains consistent with the described region. Any abnormal findings not provided in the text should be treated as normal for that region. We require a diverse set of questions. Options may include 'All of the above', 'None of the above' or 'Unknown'.

The output must be in JSON format.
[Output Template]:
```json
{
    "invalid": true/false,
    "error_type": ...,
    "evidence": ...,
    "new_question": ...
}
```
""")
