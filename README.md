# gita-gpt

I've outlined the process at https://lnkd.in/gZFSaB-T. Thanks in big part to [Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series, specifically on the first lecture on nanoGPT, I trained GPT-2.0 on Gita using Lambda.ai's entry level GH200 instance (96 GB GPU, $1.49/hour). To run the inference, I put to use 4-year old entry level NVIDIA GeForce MX150 GPU (2GB VRAM) when I noticed CPU inferences are slow. 

GPU + CUDA performance improvements were close to 13x in my case (typically 5-10x). 

<img width="1918" height="1015" alt="GeForce-MX150-Inference" src="https://github.com/user-attachments/assets/7d99e659-1747-45be-82de-97f549715bc6" />


And finally, when I see my transformer quote Krishna, I know this project was a success:
<img width="1917" height="567" alt="Sthitapragya-RAG" src="https://github.com/user-attachments/assets/85e68187-264b-47be-aa94-d6740602e8cf" />


### License

MIT
