from ..core.storage.gcs import GCSStorageProvider
from ..config import Config
from ..core.storage.base import StorageProvider
from fastapi import HTTPException
import random

# List of adjectives and nouns
adjectives = [
    "funny", "clever", "brave", "calm", "eager", "gentle", "happy", "jolly",
    "kind", "lively", "nice", "proud", "silly", "tender", "wise", "blue",
    "red", "green", "yellow", "purple", "bright", "shiny", "swift", "smart",
    "quick", "peaceful", "friendly", "cheerful", "merry", "joyful",
    "goofy", "bouncy", "giggly", "wobbly", "quirky", "zany", "loopy",
    "dizzy", "wiggly", "bubbly", "sparkly", "fluffy", "snuggly", "cuddly"
]

nouns = [
    "fox", "dog", "cat", "wolf", "bear", "lion", "tiger", "eagle", "hawk",
    "owl", "fish", "deer", "rabbit", "mouse", "horse", "cow", "sheep",
    "goat", "pig", "duck", "dragon", "unicorn", "phoenix", "dolphin", "panda",
    "koala", "penguin", "otter", "hamster", "butterfly", "narwhal", "platypus",
    "sloth", "raccoon", "hedgehog", "alpaca", "quokka", "capybara", "axolotl",
    "wombat", "lemur", "meerkat", "chinchilla", "manatee", "walrus"
]

async def get_storage_client() -> GCSStorageProvider:
    client = GCSStorageProvider()
    await client.initialize(config={
        "bucket_name": Config.GCS_BUCKET_NAME,
        "project_id": Config.GCS_PROJECT_ID
    })
    return client

async def get_storage_provider() -> StorageProvider:
    return await get_storage_client()


def get_random_element(array):
    return random.choice(array)

def generate_random_name():
    adjective = get_random_element(adjectives)
    noun = get_random_element(nouns)
    number = random.randint(1, 1000)  # Random number between 1 and 1000
    return f"{adjective}-{noun}-{number}"

def handle_error(e: Exception):
    """Handle error"""
    if Config.DEBUG:
        import traceback
        traceback.print_exc()
    if isinstance(e, ValueError):
        raise HTTPException(status_code=400, detail=str(e))
    raise HTTPException(status_code=500, detail=str(e))
