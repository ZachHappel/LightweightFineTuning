from transformers import AutoTokenizer
import torch
from peft import AutoPeftModelForSequenceClassification

'''

After running `lightweight_finetuning.py`, you can use this script
to test your own prompts. There are some already provided but it's
fun to test out your own!

If you've changed the default LoRA savename in the previous file, 
make sure that change is reflected below in the variable `lora_name`

Have fun!

'''

# ------------------------------------------------------------------

# Name used when training/saving LoRA (adjust if necessary)
lora_name = "gpt2-lora-bias_in_bios" 

# Put your prompts here 
prompts = ["She worked in an elementary class room. Elizabeth Stiner was her name.",
           "He worked in an elementary class room. Matthew Stiner was his name.",
           "Mr. Xi graduated from Harvard University, summa cum laude.",
           "A preschool teacher, a poet, and even a librarian, Mr. Adams loved all things literature and school!",
           "The developer of this repository finds this work to be fascinating and he looks forward to doing more of it.",
           "Deckhands were in short supply and a new one was desperately needed before they would depart from port. When Sabrina submitted her application, it was immediately accepted."]


# ------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

lora_model = AutoPeftModelForSequenceClassification.from_pretrained(lora_name, num_labels=2, ignore_mismatched_sizes=True).to(device)
lora_model.config.pad_token_id = tokenizer.pad_token_id

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = lora_model(**inputs)

    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    
    id2label = {0: "MALE", 1: "FEMALE"}
    predicted_label = id2label[predicted_class_id]
    
    return predicted_label

def perform_predictions(prompts):
    for i in range(len(prompts)):
        prediction = predict(prompts[i])
        print(f"Prediction for input '{prompts[i]}': {prediction}")


if __name__ == '__main__':
    perform_predictions(prompts)
        