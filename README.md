# MYNA — AI Outfit Stylist for Myntra
### *"See It. Style It. Shop It."*

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-red) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-yellow) ![AWS](https://img.shields.io/badge/AWS-RDS%20%2B%20S3-orange) ![Docker](https://img.shields.io/badge/Docker-Deployment-blue)

---

## Project Overview

MYNA is an AI-powered personal stylist built for Myntra. Users upload their photo, get virtual try-ons powered by IDM-VTON and Leffa, receive smart recommendations from a catalog of 352,706 Myntra products, and interact through a conversational AI agent — all in a single Streamlit interface.

---

## Features (9 Approaches)

### 1. Photo Upload and Interaction
- Upload photo via Streamlit UI
- YOLOv8n detects clothing components automatically
- Chat-based onboarding with 5 style selection buttons
- Image preprocessed (RGB convert, resize, optimize)

### 2. AI Outfit Generation
- **Upper body** → IDM-VTON (yisol/IDM-VTON on HuggingFace)
- **Lower body** → Leffa (franciszzj/Leffa on HuggingFace)
- Conversation iteration: "make it blue", "change to formal"
- Generated images saved to AWS S3
- Auto-retry on GPU quota with 3 attempts (10s, 20s, 30s waits)

### 3. Outfit Component Detection
- SegFormer B2 (mattmdjaga/segformer_b2_clothes) pixel-level segmentation
- Detects exact garment type: Topwear, Bottomwear, Footwear, Accessories
- After try-on shows: `👕 Topwear — Printed Straight Kurta`
- Falls back to YOLOv8 zone detection if SegFormer unavailable

### 4. Recommendation Engine
- 352,706 Myntra products (CSV-based, no API needed)
- Content-based filtering with gender, price, occasion, usage filters
- 5 products per category with image, price, brand, rating
- Direct "Buy on Myntra" links
- Supports misspelled queries: "footware", "snekers" → Footwear

### 5. Conversational AI with RAG
- LangChain + ChromaDB with 15,000 product embeddings
- sentence-transformers all-MiniLM-L6-v2
- 6 intents: WHY / ALT / MATCH / BRAND / BUDGET / DEFAULT
- "Why did you recommend this?" ✅
- "Show alternatives under ₹1000" ✅
- "What matches with this kurta?" ✅

### 6. Cart Management
- Add to Cart button on each product card
- Cart visible in left sidebar with total price
- Remove items with ❌ button
- Cross-sell suggestions after adding item
- Checkout on Myntra button

### 7. Data Storage
- **AWS RDS PostgreSQL** — 4 tables: users, outfits, product_catalog, cart_recommendations
- **AWS S3** — generated outfit images stored with public URLs
- **ChromaDB** — RAG vector embeddings (62 MB, 15K products)
- Local fallback if cloud unavailable

### 8. Analytics Dashboard
- Total users, outfits, products, cart metrics
- Cart conversion rate with progress bar
- Top categories and styles bar charts
- CTR tracking per recommendation

### 9. Docker Deployment
- Dockerfile + docker-compose.yml included
- Runs locally or on any cloud platform
- All dependencies in requirements.txt

---

## Models Used

| Model | Purpose | Runs On | Cost |
|-------|---------|---------|------|
| YOLOv8n | Clothing detection | Local | Free |
| SegFormer B2 | Pixel-level segmentation | Local | Free |
| IDM-VTON | Upper body virtual try-on | HuggingFace | Free* |
| Leffa | Lower body virtual try-on | HuggingFace | Free* |
| sentence-transformers | RAG product embeddings | Local | Free |
| Pandas + NLP | Product recommendations | Local | Free |

*Free tier = 120 GPU seconds per day. App handles quota with auto-retry.

---

## Database Schema

### Users Table
| Column | Type | Description |
|--------|------|-------------|
| user_id | UUID PK | Unique identifier |
| name | VARCHAR | Full name |
| email | VARCHAR | Email address |
| created_at | TIMESTAMP | Account creation time |
| last_login | TIMESTAMP | Last login time |

### Outfits Table
| Column | Type | Description |
|--------|------|-------------|
| outfit_id | UUID PK | Unique identifier |
| user_id | UUID FK | References Users |
| photo_url | VARCHAR | AWS S3 URL |
| style_type | VARCHAR | Ethnic / Casual / Party |
| top_label | VARCHAR | Detected top type |
| bottom_label | VARCHAR | Detected bottom type |
| created_at | TIMESTAMP | Generation timestamp |

### Product Catalog Table
| Column | Type | Description |
|--------|------|-------------|
| product_id | VARCHAR PK | Unique product identifier |
| name | VARCHAR | Product name |
| category | VARCHAR | Top / Bottom / Footwear |
| price | DECIMAL | Selling price |
| image_url | VARCHAR | Product image URL |
| product_url | VARCHAR | Myntra product link |

### Cart Recommendations Table
| Column | Type | Description |
|--------|------|-------------|
| entry_id | UUID PK | Unique entry identifier |
| user_id | UUID FK | References Users |
| outfit_id | UUID FK | References Outfits |
| product_id | VARCHAR FK | References Product Catalog |
| component_type | VARCHAR | Top / Bottom / Footwear |
| added_to_cart | BOOLEAN | True if added to cart |
| purchased | BOOLEAN | True if purchased |

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Create .env file in project root

HUGGINGFACE_TOKEN=hf_your_token_here
DB_HOST=your-rds-endpoint.amazonaws.com
DB_PORT=5432
DB_NAME=myna_db
DB_USER=postgres
DB_PASSWORD=your_password
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=ap-south-1
S3_BUCKET_NAME=your_bucket_name

### Step 3 — Get HuggingFace Token
1. Go to huggingface.co
2. Settings → Access Tokens → New Token
3. Paste in .env as HUGGINGFACE_TOKEN

### Step 4 — Run the app

streamlit run src/app/streamlit_app.py --server.fileWatcherType none

### Run with Docker

---

## Project Structure

```
MYNA-Myntra-AI-Stylist/
├── src/
│   ├── app/streamlit_app.py
│   ├── conversation/rag_agent.py
│   ├── database/db_manager.py
│   ├── detection/detector.py
│   ├── outfit_generation/outfit_generator.py
│   ├── recommender/recommender.py
│   ├── storage/s3_manager.py
│   └── vision/
│       ├── component_detector.py
│       └── segmentation.py
├── data/
│   ├── chroma_db/
│   ├── processed/clean_myntra_products.csv
│   └── raw/user_uploads/
├── notebooks/
├── reports/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
---

## Complete Flow

1. User uploads photo
2. YOLOv8n detects clothing zones
3. User asks — blue kurta for Pongal under 2000
4. Recommender filters 352,706 products
5. 5 products shown with Try On and Cart buttons
6. Try On → IDM-VTON for top / Leffa for bottom
7. Result image shown in chat
8. SegFormer detects components in result image
9. Why recommend → RAG agent explains
10. Add to Cart → saved to AWS RDS
11. Analytics tab → charts and metrics
---

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | Streamlit |
| Virtual Try-On | IDM-VTON + Leffa |
| Detection | YOLOv8n |
| Segmentation | SegFormer B2 |
| Embeddings | sentence-transformers |
| Vector Store | ChromaDB |
| RAG | LangChain |
| Database | PostgreSQL on AWS RDS |
| Storage | AWS S3 |
| Container | Docker |
| Language | Python 3.12 |

---

## Limitations and Future Scope

| Limitation | Future Scope |
|------------|--------------|
| HuggingFace free tier 120 GPU seconds per day | Dedicated GPU instance |
| Best results with plain background photos | Background removal preprocessing |
| Cart does not persist across sessions | User login and persistent sessions |
| RAG indexes 15K of 352K products | Full catalog GPU embedding |
| YOLOv8 trained on COCO not fashion | Fine-tune on Myntra categories |

---

*MYNA — AI Outfit Stylist — April 2025*
