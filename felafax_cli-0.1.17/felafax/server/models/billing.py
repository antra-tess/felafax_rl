from pydantic import BaseModel
from datetime import datetime
from typing import Dict, List

class GPUUsage(BaseModel):
    """GPU usage record schema"""
    seconds: int
    cost_per_hour: float
    total_cost: float

class JobCost(BaseModel):
    """Job cost record schema"""
    job_id: str
    gpu_type: str
    duration_seconds: int
    cost: float

class BillingPeriod(BaseModel):
    """Billing period schema"""
    start_date: datetime
    end_date: datetime
    total_cost: float
    usage_breakdown: Dict[str, GPUUsage]
    jobs: List[JobCost]

class BillingStoragePaths:
    """Billing storage path generator"""
    @staticmethod
    def base_path(user_id: str) -> str:
        return f"users/{user_id}/metadata"
    
    @staticmethod
    def current_period_path(user_id: str) -> str:
        return f"users/{user_id}/metadata/billing.json"
    
    @staticmethod
    def history_path(user_id: str) -> str:
        return f"users/{user_id}/metadata/billing_history.json"