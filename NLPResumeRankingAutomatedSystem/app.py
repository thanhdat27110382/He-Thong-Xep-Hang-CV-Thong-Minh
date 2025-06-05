import os
import numpy as np
from flask import Flask, request, render_template
import PyPDF2
import pickle
from nltk.corpus import stopwords
import nltk
import re
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time
from werkzeug.utils import secure_filename
import joblib
import torch
import pdfplumber
import fitz  
import camelot

app = Flask(__name__)

# Set up logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load Spacy model for NER
nlp = spacy.load("en_core_web_sm")

# Thiết lập device
device = torch.device("cpu")

# Định nghĩa mô hình Transformer (để khởi tạo lại cấu trúc)
class TransformerModel(torch.nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.input_projection = torch.nn.Linear(input_dim, d_model)
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.2, batch_first=True),
            num_layers=num_layers
        )
        self.fc = torch.nn.Linear(d_model, 1)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # Thêm chiều seq_len = 1
        x = self.input_projection(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # Lấy output cuối cùng
        x = self.fc(x)
        return x

# Định nghĩa lớp Ensemble
class EnsembleRXTModel(torch.nn.Module):
    def __init__(self, rf_model, xgb_model, transformer_model, rf_weight=0.6, xgb_weight=0.2, transformer_weight=0.2):
        super(EnsembleRXTModel, self).__init__()
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.transformer_model = transformer_model
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        self.transformer_weight = transformer_weight
    
    def forward(self, x):
        x_numpy = x.cpu().numpy()
        rf_output = torch.tensor(self.rf_model.predict(x_numpy), dtype=torch.float32).to(x.device)
        xgb_output = torch.tensor(self.xgb_model.predict(x_numpy), dtype=torch.float32).to(x.device)
        transformer_output = self.transformer_model(x).squeeze()
        ensemble_output = (self.rf_weight * rf_output +
                           self.xgb_weight * xgb_output +
                           self.transformer_weight * transformer_output)
        return ensemble_output

# Tải mô hình ensemble rf + xgb + transformer
try:
    # 1. Tải Random Forest
    rf_model = joblib.load("D:/BaiDoAnChuyenNganh3/NLPResumeRankingAutomatedSystem/model/rf_model.pkl")
    logging.info("Mô hình Random Forest đã được tải thành công.")
    
    # 2. Tải XGBoost
    xgb_model = joblib.load("D:/BaiDoAnChuyenNganh3/NLPResumeRankingAutomatedSystem/model/xgb_model.pkl")
    logging.info("Mô hình XGBoost đã được tải thành công.")
    
    # 3. Tải Transformer
    input_dim = 1540  
    d_model = 128
    nhead = 4
    num_layers = 2
    transformer_model = TransformerModel(input_dim, d_model, nhead, num_layers).to(device)
    transformer_model.load_state_dict(torch.load("D:/BaiDoAnChuyenNganh3/NLPResumeRankingAutomatedSystem/model/transformer_model_best.pth", weights_only=True))
    transformer_model.eval()
    logging.info("Mô hình Transformer đã được tải thành công.")
    
    # Tạo mô hình ensemble
    rf_xgb_transformer_model = EnsembleRXTModel(rf_model, xgb_model, transformer_model, rf_weight=0.6, xgb_weight=0.2, transformer_weight=0.2).to(device)

except Exception as e:
    logging.error(f"Không tải được mô hình ensemble: {str(e)}")
    raise Exception(f"Không tải được mô hình ensemble: {str(e)}")

# Tải mô hình Sentence Transformer
try:
    model_st = SentenceTransformer('all-mpnet-base-v2')
    logging.info("Tải mô hình SentenceTransformer (all-mpnet-base-v2) thành công.")
except Exception as e:
    logging.error(f"Không tải được mô hình SentenceTransformer: {str(e)}")
    raise Exception(f"Không tải được mô hình SentenceTransformer: {str(e)}")

# Đảm bảo thư mục tải lên tồn tại
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB limit

def extract_text_from_pdf(pdf_path):
    """
    Phiên bản nâng cấp kết hợp 3 thư viện:
    1. pdfplumber (ưu tiên) - giữ layout tốt nhất
    2. PyMuPDF (fallback) - xử lý nhanh và hỗ trợ OCR
    3. Camelot - trích xuất bảng đặc biệt trong CV
    """
    result = {
        "text": "",
        "tables": [],
        "metadata": {}
    }

    try:
        # --- Lớp 1: pdfplumber làm chính ---
        full_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Tối ưu hóa extract_text() cho CV
                page_text = page.extract_text(
                    x_tolerance=1,
                    y_tolerance=1,
                    keep_blank_chars=False,
                    use_text_flow=True
                )
                if page_text:
                    full_text.append(page_text)
                else:
                    # Fallback cho trang không trích xuất được text
                    page_text = page.extract_text_simple()
                    full_text.append(page_text or f"[No text on page {page.page_number}]")
        
        result["text"] = "\n".join(full_text)

        # --- Lớp 2: PyMuPDF bổ sung metadata và fallback ---
        doc = fitz.open(pdf_path)
        result["metadata"] = {
            "author": doc.metadata.get("author"),
            "title": doc.metadata.get("title"),
            "pages": len(doc)
        }

        # Fallback nếu pdfplumber không trích xuất đủ text
        if len(result["text"].strip()) < 50:  # Nếu text quá ít
            result["text"] = "\n".join([page.get_text() for page in doc])

        # --- Lớp 3: Camelot cho bảng (phần Skills, Experience) ---
        try:
            tables = camelot.read_pdf(
                pdf_path,
                flavor="lattice",
                pages="all",
                strip_text="\n"
            )
            result["tables"] = [table.df.to_markdown() for table in tables if table.parsing_report["accuracy"] > 80]
        except Exception as e:
            logging.warning(f"Không trích xuất được bảng: {str(e)}")
            result["tables"] = []

    except Exception as e:
        logging.error(f"Lỗi khi xử lý PDF: {str(e)}")
        # Fallback cuối cùng bằng PyMuPDF
        try:
            doc = fitz.open(pdf_path)
            result["text"] = "\n".join([page.get_text() for page in doc])
        except:
            result["text"] = "Không thể trích xuất nội dung PDF"

    # Kết hợp text và bảng
    combined_text = result["text"]
    if result["tables"]:
        combined_text += "\n\n=== BẢNG TRÍCH XUẤT ===\n" + "\n\n".join(result["tables"])

    return combined_text

# Kết hợp làm sạch văn bản và trích xuất tính năng NER
stop_words = set(stopwords.words('english'))
def preprocess_and_extract_features(text):
    """
    Xử lý văn bản và trích xuất thông tin kỹ năng, kinh nghiệm, học vấn cho ngành CNTT
    Tự động điều chỉnh phạm vi quét theo độ dài văn bản
    """
    # Tiền xử lý cơ bản
    text = text.lower()
    text = re.sub(r'\S+@\S+', ' ', text)  # Remove emails
    text = re.sub(r'http\S+|www\S+', ' ', text)  # Remove URLs
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', ' ', text)  # Remove phone numbers
    text = ''.join([char for char in text if char.isalnum() or char.isspace() or char in [',', '.', '-']])
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Chuẩn hóa thuật ngữ CNTT
    replacements = {
        r'\b(yrs?|years?)\b': 'years',
        r'\b(exp|experience)\b': 'experience',
        r'\b(edu|education)\b': 'education',
        r'\b(prog|programming)\b': 'programming',
        r'\b(dev|developer)\b': 'developer',
        r'\b(eng|engineer)\b': 'engineer',
        r'\b(c#|csharp)\b': 'c#',
        r'\b(js|javascript)\b': 'javascript',
        r'\b(py|python)\b': 'python'
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
    
    # Từ điển keyword cho ngành CNTT
    it_skill_keywords = [
    # Các phần header thông thường
    'skills', 'technical skills', 'technical competencies', 'key skills',
    'core competencies', 'technical expertise', 'skill set', 'skill summary',
    'programming skills', 'technical abilities', 'professional skills',
    
    # Các cách diễn đạt về kỹ năng
    'proficient in', 'experienced in', 'knowledge of', 'expertise in',
    'skilled in', 'familiar with', 'competent in', 'adept at', 
    'strong background in', 'hands-on experience with', 'working knowledge of',
    
    # Các loại kỹ năng cụ thể
    'programming languages', 'frameworks', 'tools', 'libraries',
    'platforms', 'databases', 'cloud technologies', 'devops tools',
    'frontend technologies', 'backend technologies', 'fullstack technologies',
    'mobile development', 'ai/ml technologies', 'data science tools',
    'cybersecurity skills', 'networking knowledge', 'operating systems',
    'version control systems', 'ci/cd tools', 'containerization',
    'testing frameworks', 'debugging tools', 'ide experience',
    
    # Các kỹ năng mềm liên quan
    'problem solving', 'algorithm design', 'system architecture',
    'code optimization', 'performance tuning', 'security practices',
    'agile methodologies', 'code review', 'technical documentation'
]

    it_experience_keywords = [
    # Các phần header thông thường
    'experience', 'work experience', 'professional experience',
    'employment history', 'career history', 'work background',
    'professional background', 'relevant experience',
    
    # Các cách diễn đạt về kinh nghiệm
    'worked as', 'served as', 'held position as', 'acted as',
    'years of experience', 'hands-on experience', 'it experience',
    'technical experience', 'industry experience', 'practical experience',
    
    # Các hoạt động công việc
    'projects', 'implemented', 'developed', 'deployed', 'architected',
    'engineered', 'designed', 'built', 'created', 'maintained',
    'optimized', 'debugged', 'tested', 'documented', 'led',
    'managed', 'coordinated', 'collaborated on', 'contributed to',
    
    # Các mức độ kinh nghiệm
    'junior', 'senior', 'lead', 'principal', 'architect', 'manager',
    'director', 'team lead', 'technical lead', 'subject matter expert',
    
    # Các loại kinh nghiệm cụ thể
    'full-time', 'part-time', 'contract', 'freelance', 'internship',
    'volunteer'
]

    it_education_keywords = [
    # Các phần header thông thường
    'education', 'academic background', 'qualifications',
    'degrees', 'certifications', 'academic credentials',
    'educational background', 'training', 'courses',
    
    # Các loại tổ chức giáo dục
    'university', 'college', 'institute', 'school', 'academy',
    'polytechnic', 'technical school', 'bootcamp', 'online course',
    
    # Các bằng cấp
    'bachelor', "bachelor's", 'bsc', 'bs', 'b.eng', 'b.tech',
    'master', "master's", 'msc', 'ms', 'm.eng', 'm.tech',
    'phd', 'ph.d', 'doctorate', 'postdoc', 'postdoctoral',
    'diploma', 'certificate', 'associate degree',
    
    # Các chuyên ngành CNTT
    'computer science', 'information technology', 'software engineering',
    'data science', 'computer engineering', 'it', 'cybersecurity',
    'network engineering', 'artificial intelligence', 'machine learning',
    'computer systems', 'database systems', 'web development',
    'mobile development', 'cloud computing', 'devops',
    
    # Các khóa học và đào tạo
    'coursework', 'relevant courses', 'training programs',
    'professional development', 'certification programs',
    'online learning', 'moocs', 'workshops', 'seminars'
]
    
    # Tự động điều chỉnh phạm vi quét
    text_length = len(text)
    base_range = min(1000, max(300, int(text_length * 0.3)))  # Từ 300-1000 ký tự
    
    scan_ranges = {
        'skills': base_range * 1.5,  # Ưu tiên quét rộng hơn cho kỹ năng
        'experience': base_range,
        'education': base_range
    }
    
    # Hàm quét thông minh
    def smart_scan(content, keyword, scan_range):
        pos = content.find(keyword)
        if pos == -1:
            return ""
    
        start = max(0, pos - 50)  # Quét ngược 50 ký tự
        end = min(len(content), start + int(scan_range))  # Ensure scan_range is integer
    
        # Ưu tiên lấy đến hết section nếu có dấu phân cách
        next_section = re.search(r'\n\s*\n', content[int(pos):int(end)])  # Convert to integers
        if next_section:
            end = pos + next_section.start()
    
        return content[int(start):int(end)].strip()  # Ensure integer indices
    
    # Trích xuất thông tin
    skills = set()
    experience = set()
    education = set()
    
    # 1. Trích xuất kỹ năng CNTT
    for keyword in it_skill_keywords:
        section = smart_scan(text, keyword, scan_ranges['skills'])
        if section:
            # Trích xuất từ danh sách
            skill_items = re.split(r'[,•\-–;()\n/]', section)
            for item in skill_items:
                item = item.strip()
                if 2 <= len(item) <= 50:  # Độ dài hợp lý
                    skills.add(item.title())
            
            # Trích xuất bằng NER
            doc = nlp(section)
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT'] and any(char.isalpha() for char in ent.text):
                    skills.add(ent.text.title())
    
    # 2. Trích xuất kinh nghiệm
    for keyword in it_experience_keywords:
        section = smart_scan(text, keyword, scan_ranges['experience'])
        if section:
            # Số năm kinh nghiệm
            years_exp = re.findall(
                r'(\d+)\s*(?:\+)?\s*(?:years?|yrs?)(?:\s*(?:of|in)\s*(?:experience|exp))?', 
                section
            )
            for year in years_exp:
                experience.add(f"{year} years experience")
            
            # Vị trí công việc
            job_titles = re.findall(
                r'(?:senior|junior|lead|principal)?\s*'
                r'(?:software|it|data|devops|cloud|systems)?\s*'
                r'(?:engineer|developer|architect|analyst|specialist)', 
                section
            )
            for title in job_titles:
                experience.add(title.title())
    
    # 3. Trích xuất học vấn
    for keyword in it_education_keywords:
        section = smart_scan(text, keyword, scan_ranges['education'])
        if section:
            # Bằng cấp
            degrees = re.findall(
                r'(bsc|b\.?sc|bachelor|bs|b\.?s|'
                r'msc|m\.?sc|master|ms|m\.?s|'
                r'phd|ph\.?d|doctorate)\b', 
                section, flags=re.IGNORECASE
            )
            for degree in degrees:
                education.add(degree.title())
            
            # Tên trường
            doc = nlp(section)
            for ent in doc.ents:
                if ent.label_ == 'ORG' and any(w in ent.text.lower() for w in ['university', 'college', 'institute']):
                    education.add(ent.text.title())
    
    # Làm sạch kết quả
    def clean_items(items):
        cleaned = set()
        for item in items:
            item = re.sub(r'[^\w\s\-\.#+]', '', item)  # Loại bỏ ký tự đặc biệt
            item = item.strip()
            if item and not item.isdigit():
                cleaned.add(item)
        return sorted(cleaned)
    
    return {
        'cleaned_text': ' '.join([token.lemma_ for token in nlp(text) if token.text not in stop_words]),
        'features': {
            'skills': clean_items(skills),
            'experience': clean_items(experience),
            'education': clean_items(education)
        }
    }

# Chuẩn hóa các thuật ngữ phổ biến
def normalize_terms(text):
    # Chuyển đổi văn bản thành chữ thường để khớp không phân biệt chữ hoa chữ thường
    text_lower = text.lower()
    
    # Từ điển thay thế mở rộng
    replacements = {
        # Ngôn ngữ lập trình
        'javascript': 'js',
        'java script': 'js',
        'python': 'py',
        'c plus plus': 'c++',
        'cplus plus': 'c++',
        'c plusplus': 'c++',
        'c#': 'csharp',
        'c sharp': 'csharp',
        'java': 'java',
        'ruby': 'ruby',
        'php': 'php',
        'go': 'go',
        'golang': 'go',
        'rust': 'rust',
        'kotlin': 'kotlin',
        'swift': 'swift',
        'typescript': 'ts',
        'scala': 'scala',
        'perl': 'perl',
        'r language': 'r',
        'r programming': 'r',
        'dart': 'dart',
        'lua': 'lua',
        'elixir': 'elixir',
        'erlang': 'erlang',
        'haskell': 'haskell',
        'clojure': 'clojure',
        'f#': 'fsharp',
        'f sharp': 'fsharp',
        'groovy': 'groovy',
        'matlab': 'matlab',
        'shell scripting': 'shell',
        'bash': 'bash',
        'powershell': 'powershell',
        
        # Khung và Thư viện
        'scikit-learn': 'sklearn',
        'scikit learn': 'sklearn',
        'tensorflow': 'tensorflow',
        'tensor flow': 'tensorflow',
        'pytorch': 'pytorch',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'django': 'django',
        'flask': 'flask',
        'fastapi': 'fastapi',
        'spring': 'spring',
        'spring boot': 'springboot',
        'hibernate': 'hibernate',
        'asp.net': 'aspnet',
        'asp net': 'aspnet',
        'vb.net': 'vbnet',
        'vb net': 'vbnet',
        'j2ee': 'j2ee',
        'node.js': 'nodejs',
        'node js': 'nodejs',
        'react': 'react',
        'reactjs': 'react',
        'react.js': 'react',
        'angular': 'angular',
        'angularjs': 'angular',
        'vue.js': 'vuejs',
        'vuejs': 'vuejs',
        'vue js': 'vuejs',
        'express': 'express',
        'express.js': 'express',
        'laravel': 'laravel',
        'symfony': 'symfony',
        'rails': 'rails',
        'ruby on rails': 'rails',
        'next.js': 'nextjs',
        'nextjs': 'nextjs',
        'nuxt.js': 'nuxtjs',
        'nuxtjs': 'nuxtjs',
        'svelte': 'svelte',
        'sveltekit': 'sveltekit',
        'meteor': 'meteor',
        'backbone.js': 'backbonejs',
        'ember.js': 'emberjs',
        'jquery': 'jquery',
        'bootstrap': 'bootstrap',
        'tailwind css': 'tailwind',
        'material ui': 'mui',
        'ant design': 'antd',
        
        # Công cụ và Môi trường phát triển
        'visual studio': 'vs',
        'visual studio code': 'vscode',
        'vs code': 'vscode',
        'intellij': 'intellij',
        'intellij idea': 'intellij',
        'eclipse': 'eclipse',
        'pycharm': 'pycharm',
        'webstorm': 'webstorm',
        'rubymine': 'rubymine',
        'android studio': 'androidstudio',
        'xcode': 'xcode',
        'jupyter': 'jupyter',
        'jupyter notebook': 'jupyter',
        'jupyter lab': 'jupyterlab',
        'git': 'git',
        'github': 'github',
        'gitlab': 'gitlab',
        'bitbucket': 'bitbucket',
        'svn': 'svn',
        'mercurial': 'mercurial',
        'npm': 'npm',
        'yarn': 'yarn',
        'pnpm': 'pnpm',
        'maven': 'maven',
        'gradle': 'gradle',
        'ant': 'ant',
        'cmake': 'cmake',
        'make': 'make',
        
        # Databases
        'mysql': 'mysql',
        'sql server': 'sqlserver',
        'microsoft sql server': 'sqlserver',
        'ms sql server': 'sqlserver',
        'postgresql': 'postgres',
        'postgres': 'postgres',
        'oracle': 'oracle',
        'mongodb': 'mongodb',
        'redis': 'redis',
        'cassandra': 'cassandra',
        'dynamodb': 'dynamodb',
        'sqlite': 'sqlite',
        'mariadb': 'mariadb',
        'couchdb': 'couchdb',
        'neo4j': 'neo4j',
        'influxdb': 'influxdb',
        'rethinkdb': 'rethinkdb',
        'nosql': 'nosql',
        'elasticsearch': 'elasticsearch',
        'elastic search': 'elasticsearch',
        'opensearch': 'opensearch',
        
        # Artificial Intelligence and Machine Learning
        'machine learning': 'ml',
        'deep learning': 'dl',
        'natural language processing': 'nlp',
        'computer vision': 'cv',
        'support vector machine': 'svm',
        'k nearest neighbors': 'knn',
        'k-nearest neighbors': 'knn',
        'principal component analysis': 'pca',
        'random forest': 'rf',
        'decision tree': 'dt',
        'naive bayes': 'nb',
        'naïve bayes': 'nb',
        'gradient boosting': 'gb',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'catboost': 'catboost',
        'yolo': 'yolo',
        'convolutional neural network': 'cnn',
        'recurrent neural network': 'rnn',
        'long short term memory': 'lstm',
        'gated recurrent unit': 'gru',
        'transformer': 'transformer',
        'bert': 'bert',
        'gpt': 'gpt',
        't5': 't5',
        'roberta': 'roberta',
        'llama': 'llama',
        'mistral': 'mistral',
        'generative ai': 'genai',
        'reinforcement learning': 'rl',
        'federated learning': 'fl',
        'autoencoders': 'autoencoders',
        'gan': 'gan',
        'generative adversarial network': 'gan',
        
        # DevOps và CI/CD
        'devops': 'devops',
        'continuous integration': 'ci',
        'continuous deployment': 'cd',
        'continuous delivery': 'cd',
        'docker': 'docker',
        'kubernetes': 'k8s',
        'k8s': 'k8s',
        'helm': 'helm',
        'jenkins': 'jenkins',
        'ansible': 'ansible',
        'terraform': 'terraform',
        'puppet': 'puppet',
        'chef': 'chef',
        'gitlab ci': 'gitlabci',
        'github actions': 'ghactions',
        'circleci': 'circleci',
        'travis ci': 'travisci',
        'bamboo': 'bamboo',
        'prometheus': 'prometheus',
        'grafana': 'grafana',
        'elk stack': 'elk',
        'splunk': 'splunk',
        'new relic': 'newrelic',
        'datadog': 'datadog',
        
        # Điện toán đám mây
        'cloud computing': 'cloud',
        'amazon web services': 'aws',
        'aws': 'aws',
        'microsoft azure': 'azure',
        'azure': 'azure',
        'google cloud platform': 'gcp',
        'gcp': 'google cloud',
        'google cloud': 'google cloud',
        'ibm cloud': 'ibmcloud',
        'oracle cloud': 'oraclecloud',
        'alibaba cloud': 'alibabacloud',
        'heroku': 'heroku',
        'digitalocean': 'digitalocean',
        'linode': 'linode',
        'vultr': 'vultr',
        'serverless': 'serverless',
        'aws lambda': 'awslambda',
        'azure functions': 'azurefunctions',
        'google cloud functions': 'gcpfunctions',
        'cloudformation': 'cloudformation',
        'azure resource manager': 'arm',
        
        # An ninh mạng
        'cybersecurity': 'cybersecurity',
        'information security': 'infosec',
        'infosec': 'infosec',
        'owasp': 'owasp',
        'oauth': 'oauth',
        'openid': 'openid',
        'saml': 'saml',
        'ssl': 'ssl',
        'tls': 'tls',
        'penetration testing': 'pentest',
        'pentest': 'pentest',
        'ethical hacking': 'ethicalhacking',
        'vulnerability assessment': 'vulnassess',
        'siem': 'siem',
        'firewall': 'firewall',
        'ids': 'ids',
        'ips': 'ips',
        'vpn': 'vpn',
        'zero trust': 'zerotrust',
        
        # Phương pháp phát triển
        'agile': 'agile',
        'scrum': 'scrum',
        'kanban': 'kanban',
        'lean': 'lean',
        'test driven development': 'tdd',
        'tdd': 'tdd',
        'behavior driven development': 'bdd',
        'bdd': 'bdd',
        'pair programming': 'pairprogramming',
        'extreme programming': 'xp',
        'xp': 'xp',
        'devsecops': 'devsecops',
        'site reliability engineering': 'sre',
        'sre': 'sre',
        
        # UI/UX
        'user interface': 'ui',
        'ui': 'ui',
        'user experience': 'ux',
        'ux': 'ux',
        'front end': 'frontend',
        'frontend': 'frontend',
        'back end': 'backend',
        'backend': 'backend',
        'full stack': 'fullstack',
        'fullstack': 'fullstack',
        'responsive design': 'responsivedesign',
        'progressive web app': 'pwa',
        'pwa': 'pwa',
        
        # API và Tích hợp
        'application programming interface': 'api',
        'api': 'api',
        'rest': 'rest',
        'restful': 'rest',
        'graphql': 'graphql',
        'soap': 'soap',
        'grpc': 'grpc',
        'microservices': 'microservices',
        'service oriented architecture': 'soa',
        'soa': 'soa',
        'event driven architecture': 'eda',
        'eda': 'eda',
        
        # Data Analysis and Visualization
        'tableau': 'tableau',
        'power bi': 'powerbi',
        'powerbi': 'powerbi',
        'qlikview': 'qlikview',
        'qliksense': 'qliksense',
        'looker': 'looker',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'ggplot': 'ggplot',
        'd3.js': 'd3js',
        'd3js': 'd3js',
        
        # Big Data
        'hadoop': 'hadoop',
        'hadoop ecosystem': 'hadoop',
        'spark': 'spark',
        'apache spark': 'spark',
        'kafka': 'kafka',
        'apache kafka': 'kafka',
        'flink': 'flink',
        'hive': 'hive',
        'pig': 'pig',
        'hbase': 'hbase',
        'impala': 'impala',
        'sqoop': 'sqoop',
        'oozie': 'oozie',
        'zookeeper': 'zookeeper',
        'airflow': 'airflow',
        'nifi': 'nifi',
        
        # Education
        'bachelor of science': 'bs',
        'b.s.': 'bs',
        'bsc': 'bs',
        'bs': 'bs',
        'bachelor of arts': 'ba',
        'b.a.': 'ba',
        'ba': 'ba',
        'master of science': 'ms',
        'm.s.': 'ms',
        'msc': 'ms',
        'ms': 'ms',
        'master of arts': 'ma',
        'm.a.': 'ma',
        'ma': 'ma',
        'bachelor of technology': 'btech',
        'b.tech': 'btech',
        'btech': 'btech',
        'ph.d.': 'phd',
        'phd': 'phd',
        'doctor of philosophy': 'phd',
        'm.b.a.': 'mba',
        'mba': 'mba',
        'master of business administration': 'mba',
        'bachelor of engineering': 'beng',
        'b.eng': 'beng',
        'beng': 'beng',
        'bachelor of computer science': 'bcs',
        'bcs': 'bcs',
        
        # Chứng chỉ CNTT
        'aws certified': 'awscertified',
        'aws solutions architect': 'awscertified',
        'aws developer': 'awscertified',
        'aws sysops': 'awscertified',
        'microsoft certified': 'mscertified',
        'azure certified': 'azurecertified',
        'google cloud certified': 'gcpcertified',
        'cisco certified network associate': 'ccna',
        'ccna': 'ccna',
        'cisco certified network professional': 'ccnp',
        'ccnp': 'ccnp',
        'certified information systems security professional': 'cissp',
        'cissp': 'cissp',
        'certified ethical hacker': 'ceh',
        'ceh': 'ceh',
        'comptia security+': 'securityplus',
        'security+': 'securityplus',
        'comptia network+': 'networkplus',
        'network+': 'networkplus',
        'comptia a+': 'aplus',
        'a+': 'aplus',
        'scrum master': 'scrummaster',
        'certified scrum master': 'scrummaster',
        'pmp': 'pmp',
        'project management professional': 'pmp',
        'itil': 'itil',
        'togaf': 'togaf',
        'six sigma': 'sixsigma',
        
        # Experience
        'one year': '1 year',
        'two years': '2 years',
        'three years': '3 years',
        'four years': '4 years',
        'five years': '5 years',
        'six years': '6 years',
        'seven years': '7 years',
        'eight years': '8 years',
        'nine years': '9 years',
        'ten years': '10 years',
        
        # Thuật ngữ CNTT khác
        'data science': 'ds',
        'software engineer': 'swe',
        'software engineering': 'swe',
        'data engineer': 'dataengineer',
        'data engineering': 'dataengineer',
        'cloud engineer': 'cloudengineer',
        'ml engineer': 'mlengineer',
        'ai engineer': 'aiengineer',
        'object oriented': 'oo',
        'object-oriented': 'oo',
        'functional programming': 'fp',
        'blockchain': 'blockchain',
        'web3': 'web3',
        'internet of things': 'iot',
        'iot': 'iot',
        'virtual reality': 'vr',
        'vr': 'vr',
        'augmented reality': 'ar',
        'ar': 'ar',
        'mixed reality': 'mr',
        'mr': 'mr',
        'mobile development': 'mobiledev',
        'android development': 'androiddev',
        'ios development': 'iosdev',
        'cross platform': 'crossplatform',
        'flutter': 'flutter',
        'react native': 'reactnative',
        'xamarin': 'xamarin',
        'game development': 'gamedev',
        'unity': 'unity',
        'unreal engine': 'unreal',
        'embedded systems': 'embedded',
        'firmware': 'firmware',
        'robotics': 'robotics',
        'quantum computing': 'quantum',
        'edge computing': 'edgecomputing',
        'data warehouse': 'datawarehouse',
        'etl': 'etl',
        'data lake': 'datalake',
        'data pipeline': 'datapipeline',
        'business intelligence': 'bi',
        'bi': 'bi'
    }
    
    # Thay thế các thuật ngữ bằng cách sử dụng văn bản viết thường, nhưng vẫn giữ nguyên văn bản gốc đối với các phần không khớp
    for key, value in replacements.items():
        # Sử dụng regex để thay thế toàn bộ từ, không phân biệt chữ hoa chữ thường
        import re
        pattern = r'\b' + re.escape(key) + r'\b'
        text = re.sub(pattern, value, text_lower, flags=re.IGNORECASE)
    
    return text

# Kết hợp các tính năng NER vào một định dạng có cấu trúc
def combine_ner_features(cv_feats, jd_feats):
    # Đếm số lượng khớp giữa các kỹ năng CV và JD
    skill_matches = len(set(cv_feats['skills']) & set(jd_feats['skills']))
    # Đếm các kết quả trùng khớp trong giáo dục (ví dụ: cùng trường đại học hoặc loại bằng cấp)
    education_matches = len(set(cv_feats['education']) & set(jd_feats['education']))
    # Trích xuất nhiều năm kinh nghiệm và so sánh
    cv_exp_years = 0
    jd_exp_years = 0
    for exp in cv_feats['experience']:
        match = re.search(r'(\d+)\s*years?', exp, re.IGNORECASE)
        if match:
            cv_exp_years = max(cv_exp_years, int(match.group(1)))
    for exp in jd_feats['experience']:
        match = re.search(r'(\d+)\s*years?', exp, re.IGNORECASE)
        if match:
            jd_exp_years = max(jd_exp_years, int(match.group(1)))
    exp_match = 1 if cv_exp_years >= jd_exp_years and jd_exp_years > 0 else 0
    return [skill_matches, education_matches, exp_match]

# Dự đoán điểm cho nhiều CV trong một đợt
def predict_scores(cv_texts, jd_text, rf_xgb_transformer_model, model_st):
    try:
        start_time = time.time()
        
        # Tiền xử lý JD
        jd_result = preprocess_and_extract_features(jd_text)
        jd_cleaned = normalize_terms(jd_result['cleaned_text'])
        jd_feats = jd_result['features']
        
        # Xử lý CV 
        cv_results = [preprocess_and_extract_features(cv_text) for cv_text in cv_texts]
        cv_cleaned_texts = [normalize_terms(result['cleaned_text']) for result in cv_results]
        cv_feats_list = [result['features'] for result in cv_results]
        
        # Mã hóa embeddings
        cv_embeddings = model_st.encode(cv_cleaned_texts, show_progress_bar=True, batch_size=32)  # Shape: (n_cvs, 768)
        jd_embedding = model_st.encode([jd_cleaned])[0]  # Shape: (768,)
        
        # Tính độ tương đồng cosin
        similarities = cosine_similarity(cv_embeddings, jd_embedding.reshape(1, -1)).flatten()  # Shape: (n_cvs,)
        
        # Trích xuất các tính năng NER cho mỗi cặp CV-JD
        ner_features = np.array([combine_ner_features(cv_feats, jd_feats) for cv_feats in cv_feats_list])  # Shape: (n_cvs, 3)
        
        # Kết hợp các tính năng cho tất cả CV
        jd_embeddings_tiled = np.tile(jd_embedding, (len(cv_texts), 1))  # Shape: (n_cvs, 768)
        similarities_array = similarities.reshape(-1, 1)  # Shape: (n_cvs, 1)
        features = np.hstack([cv_embeddings, jd_embeddings_tiled, ner_features, similarities_array])  # Shape: (n_cvs, 1540)
        
        # Dự đoán điểm số bằng mô hình ensemble
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        with torch.no_grad():
            scores = rf_xgb_transformer_model(features_tensor).cpu().numpy()  # Shape: (n_cvs,)
        
        end_time = time.time()
        logging.info(f"Điểm số dự đoán cho {len(cv_texts)} CVs trong {end_time - start_time:.2f} giây.")
        return scores
    except Exception as e:
        logging.error(f"Lỗi trong quá trình dự đoán: {str(e)}")
        raise Exception(f"Lỗi dự đoán: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    # Validate JD text
    jd_text = request.form.get('jd_text', '').strip()
    if not jd_text:
        logging.warning("Job Description text is empty.")
        return render_template('index.html', error="Job Description text is required.")

    # Validate CV files
    cv_files = request.files.getlist('cv_files')
    if not cv_files or all(not cv_file for cv_file in cv_files):
        logging.warning("No CV files uploaded.")
        return render_template('index.html', error="At least one CV file is required.")

    results = []
    cv_texts = []
    cv_names = []
    cv_contents = []

    # Process each CV file
    for cv_file in cv_files:
        if not cv_file or not cv_file.filename:
            continue
        
        if not cv_file.filename.endswith('.pdf'):
            logging.warning(f"Invalid file type for {cv_file.filename}: Must be a PDF.")
            continue
        
        filename = secure_filename(cv_file.filename)
        cv_file.seek(0, os.SEEK_END)
        file_size = cv_file.tell()
        cv_file.seek(0)
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            logging.warning(f"File {filename} exceeds size limit of 200MB.")
            continue
        
        cv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            cv_file.save(cv_path)
            cv_text = extract_text_from_pdf(cv_path)
            cv_texts.append(cv_text)
            cv_names.append(filename)
            cv_contents.append(cv_text if cv_text else "No content extracted")
        except Exception as e:
            logging.error(f"Error processing CV {filename}: {str(e)}")
            results.append({'cv_name': filename, 'error': str(e)})
        finally:
            if os.path.exists(cv_path):
                os.remove(cv_path)

    if cv_texts:
        try:
            scores = predict_scores(cv_texts, jd_text, rf_xgb_transformer_model, model_st)
            for cv_name, score, cv_content in zip(cv_names, scores, cv_contents):
                results.append({
                    'cv_name': cv_name,
                    'score': round(score, 2),
                    'cv_content': cv_content.replace('"', "'")  # Replace quotes to avoid JSON issues
                })
        except Exception as e:
            return render_template('index.html', error=str(e))

    results = sorted([r for r in results if 'score' in r], key=lambda x: x['score'], reverse=True) + \
              [r for r in results if 'error' in r]
    
    return render_template('index.html', 
                         results=results, 
                         jd_text=jd_text.replace('"', "'"),  # Replace quotes to avoid JSON issues
                         safe_json=lambda x: x)

if __name__ == '__main__':
    app.run(debug=True)