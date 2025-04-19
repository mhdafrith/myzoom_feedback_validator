import torch
from transformers import BertTokenizer, BertForSequenceClassification
import gradio as gr

# Load your fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained("saved_model")
tokenizer = BertTokenizer.from_pretrained("saved_model")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Label mapping
label_map = {0: "❌ Not Aligned", 1: "✅ Aligned"}

# Dropdown reason options (customize as per your project)
# Load or define your seeds here
import pandas as pd

df = pd.read_excel("evaluation.xlsx")
reason_choices = df['reason'].dropna().astype(str).tolist()


# Prediction function
def validate_feedback(feedback, reason):
    if not feedback or not reason:
        return "Please enter both feedback and reason."

    inputs = tokenizer(
        feedback,
        reason,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return f"{label_map[pred]} (Confidence: {confidence:.2%})"

# Gradio Interface
interface = gr.Interface(
    fn=validate_feedback,
    inputs=[
        gr.Textbox(label="User Feedback", placeholder="Enter user feedback here..."),
        gr.Dropdown(label="Dropdown Reason", choices=reason_choices)
    ],
    outputs=gr.Textbox(label="Validation Result"),
    title="📚 My Zoom - Feedback Validation App",
    description="This app checks if user feedback aligns with the selected reason using a fine-tuned BERT model.",
    examples=[
        ["The video was blurry and hard to see.", "Video quality is poor"],
        ["I found the content very clear.", "Content is difficult to understand"],
        ["Too many typos in the material.", "Too many grammatical errors"]
    ]
)

# Run the app
interface.launch()
