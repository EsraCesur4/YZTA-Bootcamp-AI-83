from fastapi import FastAPI
from app.models import Base, engine
from app.auth import router as auth_router

Base.metadata.create_all(bind=engine)

app = FastAPI()
app.include_router(auth_router, prefix="/auth")

@app.get("/")
def root():
    return {"status": "backend is working"}