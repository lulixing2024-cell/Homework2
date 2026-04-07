# Prompt Engineering Log — Math Problem Extractor

## Task
Extract the **final answer** and **full solution explanation** from a photo of a handwritten math problem, returning structured JSON for downstream evaluation.

---

## Initial Version

```
Please analyze this math problem image and extract:
1. The final answer only (no steps)
2. The solution/explanation (the full working process)

Respond in this exact JSON format:
{
  "answer": "...",
  "explanation": "..."
}
Only return the JSON, no other text.
```

---

## Revision 1

```
You are a math expert. Analyze the math problem in the image and extract the following:

1. **Answer**: The final answer only, with no working steps. Format in LaTeX.
2. **Explanation**: The complete solution process exactly as shown in the image — do not modify, reorder, or summarize any steps. Format in LaTeX.

Important: If either the answer or the explanation cannot be clearly identified from the image, return "cannot show" for that field.

Respond in this exact JSON format:
{
  "answer": "...",
  "explanation": "..."
}
Return only the JSON. No preamble, no additional text.
```

**What changed and why:**
A role assignment ("You are a math expert") was added to anchor the model's behavior, and LaTeX formatting was explicitly required after observing that the initial version returned plain text expressions like `4pi` instead of `$4\pi$`, making the output hard to render. A fallback value (`"cannot show"`) was also introduced after the model returned empty strings or hallucinated content for low-quality handwritten images.

**What improved, stayed the same, or got worse:**
Output formatting improved significantly — LaTeX expressions rendered correctly in the Streamlit UI. The JSON structure remained stable. However, for problems with multiple solutions (e.g., quadratic equations), the model still returned only one root, suggesting the instruction was not specific enough about exhaustive answers.

---

## Revision 2

```
You are a math expert. Analyze the math problem in the image and extract the following:

1. **Answer**: The final answer only, with no working steps. Format in LaTeX. If there are multiple answers, list all of them.
2. **Explanation**: The complete solution process exactly as shown in the image — do not modify, reorder, or summarize any steps. Format in LaTeX.

Important: If either the answer or the explanation cannot be clearly identified from the image, return "cannot show" for that field.

Respond in this exact JSON format:
{
  "answer": "...",
  "explanation": "..."
}
Return only the JSON. No preamble, no additional text.
```

**What changed and why:**
The phrase "If there are multiple answers, list all of them" was added to the Answer field after testing on the quadratic equation image (x² - 8x + 15 = 0), where Revision 1 returned only `x = 5` and omitted `x = 3`. The fix was minimal and targeted — no other instructions were altered to avoid regression.

**What improved, stayed the same, or got worse:**
The model now correctly returns both roots for multi-solution problems. JSON structure and LaTeX formatting remained consistent with Revision 1. One limitation that persists is language consistency — when the image contains Hindi or Chinese text (as in Example 3), the model occasionally returns the explanation in the source language rather than English, which may affect readability in a mixed-language evaluation dataset.
