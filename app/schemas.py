from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    tc_no: str
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    email: EmailStr

    class Config:
        orm_mode = True
