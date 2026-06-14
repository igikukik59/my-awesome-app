from transformers import pipeline
import time

# این تابع یک بار هنگام deploy اجرا می‌شه (برای لود کردن مدل)
def init():
    global classifier
    print("Loading model...")
    classifier = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    print("Model loaded successfully!")

# این تابع برای هر درخواست اجرا می‌شه
def predict(text: str) -> dict:
    start_time = time.time()
    
    if not text or len(text.strip()) == 0:
        return {"error": "Please provide a valid text"}
    
    # برش متن به حداکثر 512 کاراکتر
    text = text[:512]
    
    # اجرای مدل
    result = classifier(text)[0]
    
    processing_time = time.time() - start_time
    
    return {
        "text": text,
        "sentiment": result["label"],
        "confidence": round(result["score"], 4),
        "processing_time_seconds": round(processing_time, 3),
        "model": "twitter-roberta-base-sentiment"
    }
