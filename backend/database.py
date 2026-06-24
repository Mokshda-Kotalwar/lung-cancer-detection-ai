from motor.motor_asyncio import AsyncIOMotorClient
from backend.core.config import settings

class Database:
    client: AsyncIOMotorClient = None
    db = None

db_manager = Database()

async def connect_to_mongo():
    db_manager.client = AsyncIOMotorClient(settings.MONGODB_URL)
    db_manager.db = db_manager.client[settings.DATABASE_NAME]

async def close_mongo_connection():
    if db_manager.client:
        db_manager.client.close()

def get_database():
    return db_manager.db
