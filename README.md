This project is a Retrieval-Augmented Generation (RAG) chatbot built with Flask. It uses the Mistral-7B-Instruct model and a custom knowledge base to intelligently answer customer inquiries.

🛠️ Setup Instructions:

1. Clone the repository
2. Install the dependencies
pip install -r requirements.txt

3.🔐 Hugging Face Token Setup:
This project uses the Mistral-7B-Instruct model, which requires a Hugging Face access token.

To Get a Token:
Sign up or log in: https://huggingface.co

Go to your tokens page: https://huggingface.co/settings/tokens

Click "New Token", choose Read access

Copy the token
4.Add It to a .env File
Create a .env file in the root directory:
HF_TOKEN=your_token_here
⚠️ This file is ignored by Git and must not be shared.

4.🧠 Model and Knowledge Base Info
The first run will download the Mistral model (~GBs in size), which may take a few minutes.

The chatbot uses knowledge_base.txt as its source for answers. Make sure to fill this file with clear, relevant information.

📝 Notes:

- The first time running the code will be slower due to model downloads.

- You will need to generate a Hugging Face token (as described above).

- The bot pulls answers from knowledge_base.txt – customize this file with your business FAQs or info.

