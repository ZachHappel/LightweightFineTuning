# About

Model: GPT-2 

Dataset: [Bias in Bios](https://huggingface.co/datasets/LabHC/bias_in_bios)

### Goal 

Use Parameter-Efficient Fine-Tuning (PEFT) / Low-Rank Adaptation to fine-tune a pre-existing model with the goal being that it performs better at a particular task after the fact.

### Approach
Used `gender` as the target label.

Initially, I intended on using `profession` but after inspecting the data itself, it turned out that the fifteen potential variations for the label was in reality something like twenty-seven. I was able to easily verify that data for `gender` was consistent with what was described in their overview, so I went with that.  

### Outcome

Comparing the model's predictions from before (unmodified base weights) and after (post-adaptation,) it significantly improves in the task of discerning the gender of a sentence's subject. 

---

### Comments (For Cognizant)

While I was initially working in the workspace, I found it easier to work on my machine. Unfortunately, my GPU did not come in handy. For a reason I have yet to nail down, GPU utilization when performing any sort of training/evaluation is negligible and it may as well have been on my CPU. With that said, if this project were to run on someone else's machine, their results may vary. There is a preliminary check for a GPU with CUDA and if it detects one, it will load the tensors to the GPU. 

I thoroughly enjoyed this project. It gave me the opportunity to actually delve deep into what I had only used before at a surface level. Prior to this, I had a wide variety of experience training LoRAs for use in Stable Diffusion. Prior to getting my RTX 3090, I found an interactive Jupyter notebook which became my go-to method, as it was the quickest. You could see the code, and I intended to eventually run something like it locally - instead of using one of the local GUI methods I have already installed. 

I turns out, as I now recognize, it was using a HuggingFace Trainer! Having knowledge of some of the particulars regarding the training of LoRAs definitely helped, but there was definitely a learning curve to getting it operating successfully and in a streamlined fashion using code only. 

I would like to thank you for allowing me to partake in this program. I would also like to thank everyone responsible for putting this all together! 

Sincerely

Zacharie Happel
