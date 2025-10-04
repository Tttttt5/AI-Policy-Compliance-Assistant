import os
from dotenv import load_dotenv
load_dotenv()
print("API KEY is:", os.getenv("OPENAI_API_KEY"))