from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    tc_no: str
    email: EmailStr
    password: str
    is_admin: bool = False

class UserOut(BaseModel):
    id: str
    tc_no: str
    email: EmailStr
    is_admin: bool

    class Config:
        orm_mode = True