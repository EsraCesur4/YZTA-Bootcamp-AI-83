from sqlalchemy import Column, Integer, String, create_engine, Boolean
from sqlalchemy.orm import declarative_base
import uuid

DATABASE_URL = 'postgresql://postgres:1234@localhost/test'
engine = create_engine(DATABASE_URL, echo=True)
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    tc_no = Column(String(11), unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)

# Create tables
Base.metadata.create_all(bind=engine)