🌱 AgroGuard: AI-Powered Plant Disease & Soil Quality Detection

AgroGuard is an AI-powered agricultural assistant designed to help farmers and researchers detect plant diseases and evaluate soil quality with high accuracy. The project leverages deep learning (MobileNetV2) for plant disease detection and machine learning models for soil analysis, offering a comprehensive tool for smarter farming.

🚀 Features

✅ Plant Disease Detection

Uses MobileNetV2 (Transfer Learning) for accurate plant disease classification

Predicts disease severity levels

Provides treatment suggestions (planned module)

✅ Soil Quality Prediction

Analyzes soil parameters (N, P, K, pH, etc.)

Predicts soil fertility/quality class

Helps in choosing suitable crops

✅ User-Friendly Interface

Built with Streamlit (app.py)

Upload leaf images for instant predictions

Enter soil data manually to get quality analysis

✅ Future Enhancements

🌍 Region-wise disease analysis

🔥 Wildfire/Flood/Drought detection via satellite data (SatGuard integration)

📱 Mobile app deployment (TensorFlow Lite)

📂 Project Structure
AgroGuard/
│── app.py                   # Streamlit web app
│── requirements.txt         # Dependencies
│── models/                  # Saved models
│── notebook/                # Jupyter notebooks (EDA, preprocessing, training)
│── soil_dataset/            # Soil data (tabular format)
│── label_summary.txt        # Class labels summary
│── .gitignore               # Ignored files (datasets, env, cache)
│── README.md                # Project documentation

⚙️ Installation & Setup

1️⃣ Clone the repository

git clone https://github.com/Prakhar2104/AgroGuard.git
cd AgroGuard


2️⃣ Create a virtual environment

python -m venv agro_env
.\agro_env\Scripts\activate     # On Windows
source agro_env/bin/activate    # On Linux/Mac


3️⃣ Install dependencies

pip install -r requirements.txt


4️⃣ Run the app

streamlit run app.py

🧠 Models Used

MobileNetV2 → Plant disease classification

Random Forest / Logistic Regression (planned) → Soil quality prediction

TensorFlow & PyTorch (exploration) → Deep learning experiments

📊 Datasets

Plant Disease Dataset → Preprocessed leaf images

Soil Dataset → Tabular soil data (N, P, K, pH, etc.)

⚠️ Datasets are not included in the repo due to large size. You can add your own or request links separately.

📈 Results

Achieved >95% accuracy in plant disease detection (MobileNetV2).

Soil quality prediction tested on custom dataset with promising results.

🤝 Contribution

Contributions are welcome!

Fork the repo

Create a new branch (feature-xyz)

Commit changes

Open a Pull Request 🚀

📧 Contact

👤 Prakhar Tiwari

GitHub: Prakhar2104
