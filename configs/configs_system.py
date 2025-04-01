import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
@dataclass

class LoadConfig:
    TIMEOUT = 50
    # LLM_CONFIG
    GPT_MODEL = 'gpt-4o-mini-2024-07-18'
    TEMPERATURE_RAG = 0.2
    TEMPERATURE_CHAT = 0.5
    MAX_TOKEN = 1024
    EMBEDDING_BAAI = 'BAAI/bge-small-en-v1.5'
    VECTOR_EMBED_BAAI = 384
    EMBEDDING_OPENAI = 'text-embedding-3-small'
    VECTOR_EMBED_OPENAI = 1536
    TOP_K_PRODUCT = 3
    TOP_K_QUESTION = 3
    TOP_P = 0.9
    TOP_CONVERSATION = 8
    SYSTEM_MESSAGE = {"error_system": "Hiện tại em đang chưa hiểu rõ yêu cầu anh/chị đưa ra, để đảm bảo trải nghiệm tốt nhất trong quá trình mua sắm em mong anh/chị vui lòng đặt lại câu hỏi hoặc liên hệ đến tổng đài: 18009377 (miễn phí cước gọi) để được tư vấn thêm & hỗ trợ ạ!",
                      "end_message": "Cảm ơn anh/chị đã quan tâm đến sản phẩm và dịch vụ tư vấn y tế bên em." ,
                      "question_other": "Hiện tại em đang chưa hiểu rõ yêu cầu anh/chị đưa ra, để đảm bảo trải nghiệm tốt nhất trong quá trình tư vấn em mong anh/chị vui lòng đặt lại câu hỏi hoặc liên hệ đến tổng đài: 18009377 (miễn phí cước gọi) để được tư vấn thêm & hỗ trợ ạ! Em xin chân thành cảm ơn!"} # lỗi

SYSTEM_CONFIG = LoadConfig()

    