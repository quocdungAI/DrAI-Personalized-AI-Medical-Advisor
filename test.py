import os
import re
import pandas as pd
from tqdm.auto import tqdm
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import gradio as gr

# 1. Cấu hình OpenAI API
client = OpenAI(api_key=api_key)

# 2. Hàm chia nhỏ văn bản thành các đoạn nhỏ (với số từ làm ngưỡng)
def chunk_text(text, chunk_size=500, overlap=50):
    """Chia văn bản dài thành các đoạn nhỏ với độ trùng lặp"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

# 3. Hàm tiền xử lý văn bản
def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    return ""

# 4. Tải và xử lý dữ liệu từ các file .txt
def load_data_from_txt_files(directory_path):
    """Tải dữ liệu từ các file .txt trong thư mục"""
    data = []
    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    
    for file_name in tqdm(txt_files, desc="Đang đọc file"):
        file_path = os.path.join(directory_path, file_name)
        category = os.path.splitext(file_name)[0]
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Phương pháp 1: Tìm cặp Q&A theo định dạng
                qa_pairs = re.findall(
                    r'(?:Câu hỏi|Q|Hỏi):\s*(.*?)\s*(?:Trả lời|A|Đáp):\s*(.*?)(?=(?:Câu hỏi|Q|Hỏi):|$)',
                    content, re.DOTALL
                )
                # Nếu không tìm thấy, chia theo đoạn
                if not qa_pairs:
                    paragraphs = re.split(r'\n\s*\n', content)
                    if len(paragraphs) >= 2:
                        qa_pairs = [(paragraphs[i], paragraphs[i+1]) for i in range(0, len(paragraphs)-1, 2)]
                
                if not qa_pairs:
                    chunks = chunk_text(content, chunk_size=500, overlap=50)
                    for chunk in chunks:
                        data.append({
                            'category': category,
                            'question': f"Cho tôi biết về {category}?",
                            'answer': chunk
                        })
                else:
                    for q, a in qa_pairs:
                        data.append({
                            'category': category,
                            'question': q.strip(),
                            'answer': a.strip()
                        })
        except Exception as e:
            print(f"Lỗi khi đọc file {file_name}: {e}")
    
    print(f"Đã tải {len(data)} mục từ {len(txt_files)} file")
    return pd.DataFrame(data)

# 5. Thiết lập Chroma Vector DB
def setup_chroma_db():
    """Khởi tạo Chroma database"""
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )
    persist_directory = "./chroma_db"
    chroma_client = chromadb.Client(chromadb.Settings(persist_directory=persist_directory))
    try:
        chroma_client.delete_collection("vi_medical_corpus")
    except Exception as e:
        pass
    collection = chroma_client.create_collection(
        name="vi_medical_corpus",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

# 6. Chèn dữ liệu vào Chroma DB với kiểm tra độ dài tài liệu
def insert_data_to_chroma(collection, df):
    """Chèn dữ liệu từ DataFrame vào Chroma DB, chia nhỏ nếu quá dài"""
    documents = []
    metadatas = []
    ids = []
    max_words = 500  # Ngưỡng số từ tối đa cho mỗi tài liệu

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Đang lưu vào vector DB"):
        # Tạo nội dung tài liệu
        base_doc = f"Câu hỏi: {row['question']}\nTrả lời: {row['answer']}"
        if 'category' in row and pd.notna(row['category']):
            base_doc = f"Chuyên mục: {row['category']}\n" + base_doc
        
        # Tiền xử lý metadata
        metadata = {
            "question": preprocess_text(row['question']),
            "answer": preprocess_text(row['answer'])
        }
        if 'category' in row and pd.notna(row['category']):
            metadata["category"] = preprocess_text(row['category'])
        
        # Nếu tài liệu quá dài, chia nhỏ
        if len(base_doc.split()) > max_words:
            chunks = chunk_text(base_doc, chunk_size=max_words, overlap=50)
            for j, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append(metadata)
                ids.append(f"doc_{i}_{j}")
        else:
            documents.append(base_doc)
            metadatas.append(metadata)
            ids.append(f"doc_{i}")
    
    # Chèn dữ liệu theo từng batch
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        end_idx = min(i + batch_size, len(documents))
        collection.add(
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx]
        )
    
    print(f"Đã chèn {len(documents)} tài liệu vào Chroma DB")

# 7. Truy vấn vector search (RAG)
def vector_search(collection, query, n_results=5):
    """Truy vấn từ vector DB sử dụng vector search (RAG)"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

# 8. Sinh câu trả lời từ OpenAI dựa trên context
def generate_response(query, context):
    prompt = f"""Bạn là một trợ lý y tế ảo có khả năng trả lời các câu hỏi y tế bằng tiếng Việt.
Dựa trên thông tin được cung cấp bên dưới, hãy trả lời câu hỏi của người dùng một cách chính xác.
Nếu thông tin không có trong dữ liệu được cung cấp, hãy nói rằng bạn không có đủ thông tin và đề xuất họ tham khảo ý kiến bác sĩ.
Luôn đưa ra lời khuyên thận trọng và nhắc nhở người dùng tham khảo ý kiến chuyên gia y tế.

THÔNG TIN Y TẾ:
{context}

CÂU HỎI: {query}

TRẢ LỜI:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "Bạn là trợ lý tư vấn y tế, trả lời bằng tiếng Việt. Luôn đưa ra thông tin chính xác và đề xuất người dùng tham khảo ý kiến bác sĩ."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=600
    )
    return response.choices[0].message.content

# 9. Giao diện Gradio sử dụng vector search
def create_interface(collection):
    def chat_function(message, history):
        search_results = vector_search(collection, message, n_results=3)
        if not search_results['documents'][0]:
            return "Tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong cơ sở dữ liệu y tế. Vui lòng hỏi câu hỏi khác hoặc tham khảo ý kiến bác sĩ."
        context = "\n\n".join(search_results['documents'][0])
        print(context)
        response = generate_response(message, context)
        return response
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Chatbot Tư vấn Y tế
            
            Chào mừng bạn đến với chatbot tư vấn y tế. Chatbot sử dụng bộ dữ liệu y tế tiếng Việt để trả lời các câu hỏi liên quan đến sức khỏe.
            
            **Lưu ý**: Đây chỉ là công cụ hỗ trợ thông tin, không thay thế cho lời khuyên y tế chuyên nghiệp. Vui lòng tham khảo ý kiến bác sĩ cho các vấn đề sức khỏe cụ thể.
            """
        )
        chatbot = gr.ChatInterface(
            chat_function,
            examples=[
                "Tôi bị đau đầu và sốt nhẹ, đó có thể là triệu chứng của bệnh gì?",
                "Cách phòng ngừa bệnh tiểu đường",
                "Triệu chứng của bệnh sốt xuất huyết là gì?",
                "Làm thế nào để phòng tránh COVID-19?",
                "Các bài tập thể dục tốt cho người cao tuổi",
                "Bệnh Alzheimer có những triệu chứng gì?",
                "Cách chữa trị bệnh gout",
                "Bệnh viêm gan B lây truyền như thế nào?"
            ],
            chatbot=gr.Chatbot(height=500),
            title="Hỏi đáp Y tế"
        )
        gr.Markdown(
            """
            ### Thông tin thêm
            
            Chatbot này sử dụng bộ dữ liệu Vi-medical-corpus, bao gồm nhiều chuyên mục y tế khác nhau như:
            - Bệnh Alzheimer
            - Bệnh tiểu đường
            - Bệnh tim mạch
            - Các bệnh truyền nhiễm
            - Cách phòng ngừa bệnh tật
            - Và nhiều lĩnh vực y tế khác
            
            Chatbot sử dụng công nghệ vector search (RAG) để tìm kiếm thông tin chính xác và tạo câu trả lời dựa trên bộ dữ liệu y tế.
            """
        )
    return demo

# 10. Hàm chính chạy ứng dụng
def main():
    data_directory = "/Users/dangnam/Downloads/ChatBot_AIH/Corpus"  # Điều chỉnh đường dẫn nếu cần
    if os.path.exists("./chroma_db") and os.path.isdir("./chroma_db"):
        print("Tìm thấy vector DB đã lưu. Đang tải...")
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small"
        )
        chroma_client = chromadb.Client(chromadb.Settings(persist_directory="./chroma_db"))
        collection = chroma_client.get_collection(
            name="vi_medical_corpus", 
            embedding_function=openai_ef
        )
        print("Đã tải vector DB thành công!")
    else:
        print("Không tìm thấy vector DB. Đang tạo mới...")
        df = load_data_from_txt_files(data_directory)
        collection = setup_chroma_db()
        insert_data_to_chroma(collection, df)
        print("Đã tạo vector DB thành công!")
    
    demo = create_interface(collection)
    demo.launch(share=True)
    print("Ứng dụng đang chạy!")

if __name__ == "__main__":
    main()
