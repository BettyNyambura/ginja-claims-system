from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Ginja Claims Engine",
    description="AI-powered healthcare claims adjudication system for emerging markets.",
    version="1.0.0",
)

app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)