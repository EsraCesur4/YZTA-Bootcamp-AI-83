from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    tc_no: str
    email: str
    password: str
    is_admin: bool = False  # default normal user

class UserOut(BaseModel):
    id: int
    email: EmailStr

    class Config:
        orm_mode = True
