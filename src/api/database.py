from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create an SQLite database
db_url = "sqlite:///predictions.db"
engine = create_engine(db_url)

# Create a SQLAlchemy session
Session = sessionmaker(bind=engine)
session = Session()

# Create a declarative base
Base = declarative_base()

# Define a table to store predictions
class Prediction(Base):
    __tablename__ = "predictions2"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String)
    label = Column(String)
    predicted = Column(String)


# Create the table if it doesn't exist
Base.metadata.create_all(engine)