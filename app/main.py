from fastapi import FastAPI, Depends
from app.auth import router as auth_router, get_current_user
from app.schemas import UserOut

app = FastAPI()

app.include_router(auth_router, prefix="/auth", tags=["auth"])

@app.get("/")
def root():
    return {"message": "Welcome to your API"}

@app.get("/me", response_model=UserOut)
async def read_current_user(current_user=Depends(get_current_user)):
    return current_user
