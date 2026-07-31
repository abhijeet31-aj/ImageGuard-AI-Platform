from fastapi import FastAPI

app = FastAPI(
    title="ImageGuard Python Engine",
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "success": True,
        "message": "ImageGuard Python Engine Running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }