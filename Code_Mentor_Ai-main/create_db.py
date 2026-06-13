from app.run import app
from app import db
# Import all models so SQLAlchemy knows about them
from app.models import User_details, Snippet, Challenge, UserChallengeProgress

with app.app_context():
    db.create_all()
    print("Database tables created!")