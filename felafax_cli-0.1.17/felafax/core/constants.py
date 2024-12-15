from typing import List, Dict, Any

# TPU-related constants
VALID_TPU_FAMILIES = ["v3", "v4", "v5p", "v5e"]
VALID_TPU_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024]

# Cloud locations configuration
LOCATIONS = [
    {
        "name": "North America",
        "regions": [
            "northamerica-northeast1",
            "northamerica-northeast2",
            "us-central1",
            "us-east1",
            "us-east4",
            "us-east5",
            "us-south1",
            "us-west1",
            "us-west2",
            "us-west3",
            "us-west4",
        ],
        "default_region": "us-west1",
        "default_zone": "us-west1-b",
    },
    {
        "name": "South America",
        "regions": [
            "southamerica-east1",
            "southamerica-west1",
        ],
        "default_region": "southamerica-east1",
        "default_zone": "southamerica-east1-b",
    },
    {
        "name": "Europe",
        "regions": [
            "europe-central2",
            "europe-north1",
            "europe-southwest1",
            "europe-west1",
            "europe-west2",
            "europe-west3",
            "europe-west4",
            "europe-west6",
            "europe-west8",
            "europe-west9",
        ],
        "default_region": "europe-west4",
        "default_zone": "europe-west4-a",
    },
    {
        "name": "Asia",
        "regions": [
            "asia-east1",
            "asia-east2",
            "asia-northeast1",
            "asia-northeast2",
            "asia-northeast3",
            "asia-south1",
            "asia-south2",
            "asia-southeast1",
            "asia-southeast2",
        ],
        "default_region": "asia-southeast1",
        "default_zone": "asia-southeast1-b",
    },
    {
        "name": "Middle East",
        "regions": [
            "me-west1",
        ],
        "default_region": "me-west1",
        "default_zone": "me-west1-b",
    },
    {
        "name": "Australia",
        "regions": [
            "australia-southeast1",
            "australia-southeast2",
        ],
        "default_region": "australia-southeast1",
        "default_zone": "australia-southeast1-c",
    },
]

# Supported models for fine-tuning
# TODO: update in finetune handler mapping also if you change this
SUPPORTED_MODELS = [
    "llama3_2-1b",
    "llama3_1-8b",
    "llama3_1-70b",
    "llama3_1-405b",
    # Add other supported models here
]

# Helper functions
def get_all_regions() -> List[str]:
    """Get a flat list of all valid regions"""
    return [region for location in LOCATIONS for region in location["regions"]]

def get_default_zone_for_region(region: str) -> str:
    """Get the default zone for a given region"""
    for location in LOCATIONS:
        if region in location["regions"]:
            return f"{region}-b"  # Most regions use -b as default
    return f"{region}-a"  # Fallback to -a 