---
library_name: transformers
license: mit
pipeline_tag: image-text-to-text
---
📢 [[GitHub Repo](https://github.com/microsoft/OmniParser/tree/master)] [[OmniParser V2 Blog Post](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/)] [Huggingface demo](https://huggingface.co/spaces/microsoft/OmniParser-v2)

# Model Summary
OmniParser is a general screen parsing tool, which interprets/converts UI screenshot to structured format, to improve existing LLM based UI agent. 
Training Datasets include: 1) an interactable icon detection dataset, which was curated from popular web pages and automatically annotated to highlight clickable and actionable regions, and 2) an icon description dataset, designed to associate each UI element with its corresponding function. 

This model hub includes a finetuned version of YOLOv8 and a finetuned Florence-2 base model on the above dataset respectively. For more details of the models used and finetuning, please refer to the [paper](https://arxiv.org/abs/2408.00203).

# What's new in V2?
- Larger and cleaner set of icon caption + grounding dataset
- 60% improvement in latency compared to V1. Avg latency: 0.6s/frame on A100, 0.8s on single 4090. 
- Strong performance: 39.6 average accuracy on [ScreenSpot Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding)
- Your agent only need one tool: OmniTool. Control a Windows 11 VM with OmniParser + your vision model of choice. OmniTool supports out of the box the following large language models - OpenAI (4o/o1/o3-mini), DeepSeek (R1), Qwen (2.5VL) or Anthropic Computer Use. Check out our github repo for details. 


# Responsible AI Considerations
## Intended Use
- OmniParser is designed to be able to convert unstructured screenshot image into structured list of elements including interactable regions location and captions of icons on its potential functionality. 
- OmniParser is intended to be used in settings where users are already trained on responsible analytic approaches and critical reasoning is expected. OmniParser is capable of providing extracted information from the screenshot, however human judgement is needed for the output of OmniParser. 
- OmniParser is intended to be used on various screenshots, which includes both PC and Phone, and also on various applications.  
## limitations
- OmniParser is designed to faithfully convert screenshot image into structured elements of interactable regions and semantics of the screen, while it does not detect harmful content in its input (like users have freedom to decide the input of any LLMs), users are expected to provide input to the OmniParser that is not harmful. 
- While OmniParser only converts screenshot image into texts, it can be used to construct an GUI agent based on LLMs that is actionable. When developing and operating the agent using OmniParser, the developers need to be responsible and follow common safety standard. 

# License
Please note that icon_detect model is under AGPL license, and icon_caption is under MIT license. Please refer to the LICENSE file in the folder of each model. 

