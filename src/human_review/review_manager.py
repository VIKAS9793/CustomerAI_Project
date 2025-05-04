import os
import uuid
import logging
import datetime
import threading
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum
import asyncio
from pydantic import BaseModel, Field
from src.utils.date_provider import DateProvider

# For integrations
import slack_sdk
import requests
from databases import Database

# Local imports
from src.utils.logger import get_logger

logger = get_logger("human_review")

class ReviewPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ReviewStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"
    EXPIRED = "expired"

class ReviewCategory(str, Enum):
    FINANCIAL_ADVICE = "financial_advice"
    CUSTOMER_RESPONSE = "customer_response"
    DOCUMENT_ANALYSIS = "document_analysis"
    COMPLIANCE_CHECK = "compliance_check"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    GENERAL = "general"

class ReviewItem(BaseModel):
    """Review item following Anthropic's human feedback model structure"""
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    response: str
    category: ReviewCategory
    priority: ReviewPriority
    confidence_score: float
    model_id: str
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    due_by: Optional[datetime.datetime] = None
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer_id: Optional[str] = None
    review_timestamp: Optional[datetime.datetime] = None
    feedback: Optional[str] = None
    edited_response: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    risk_score: Optional[float] = None
    explanations: Optional[Dict[str, Any]] = None
    routing_info: Optional[Dict[str, Any]] = None
    
    class Config:
        arbitrary_types_allowed = True

class ReviewerRole(str, Enum):
    GENERAL = "general"
    FINANCIAL = "financial"
    COMPLIANCE = "compliance"
    SUBJECT_MATTER_EXPERT = "sme"
    ADMINISTRATOR = "admin"

class Reviewer(BaseModel):
    """Reviewer model based on LinkedIn's review system"""
    reviewer_id: str
    name: str
    email: str
    roles: List[ReviewerRole]
    specializations: List[str] = []
    availability_status: bool = True
    max_daily_reviews: int = 20
    current_daily_reviews: int = 0
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    
class ReviewQueue:
    """Review queue implementation inspired by Asana's task management system"""
    def __init__(self, name: str, categories: List[ReviewCategory] = None, 
                 max_size: int = 1000, reviewers: List[Reviewer] = None):
        self.name = name
        self.categories = categories or list(ReviewCategory)
        self.max_size = max_size
        self.items: Dict[str, ReviewItem] = {}
        self.reviewers = reviewers or []
        self.lock = threading.Lock()
        
    def add_item(self, item: ReviewItem) -> bool:
        """Add item to queue"""
        with self.lock:
            if len(self.items) >= self.max_size:
                return False
            self.items[item.item_id] = item
            # If Slack integration enabled, send notification for high priority items
            if item.priority in [ReviewPriority.HIGH, ReviewPriority.CRITICAL]:
                self._send_notifications(item)
            return True
    
    def get_item(self, item_id: str) -> Optional[ReviewItem]:
        """Get item by ID"""
        return self.items.get(item_id)
    
    def get_items_by_status(self, status: ReviewStatus) -> List[ReviewItem]:
        """Get items by status"""
        return [item for item in self.items.values() if item.status == status]
    
    def get_items_by_priority(self, priority: ReviewPriority) -> List[ReviewItem]:
        """Get items by priority"""
        return [item for item in self.items.values() if item.priority == priority]
    
    def get_next_item(self, reviewer: Reviewer) -> Optional[ReviewItem]:
        """Get next item for review based on reviewer specialization and item priority"""
        with self.lock:
            # Follow Salesforce Einstein's review assignment algorithm
            # 1. Match reviewer specialization with item category
            # 2. Prioritize by item priority
            # 3. Consider reviewer's current workload
            
            if reviewer.current_daily_reviews >= reviewer.max_daily_reviews:
                return None
                
            # Filter items that match reviewer's specialization
            matching_items = []
            for item in self.items.values():
                if item.status != ReviewStatus.PENDING:
                    continue
                    
                # Match reviewer role with item category
                if ReviewerRole.ADMINISTRATOR in reviewer.roles:
                    matching_items.append(item)
                elif ReviewerRole.FINANCIAL in reviewer.roles and item.category in [
                    ReviewCategory.FINANCIAL_ADVICE, ReviewCategory.COMPLIANCE_CHECK]:
                    matching_items.append(item)
                elif ReviewerRole.COMPLIANCE in reviewer.roles and item.category == ReviewCategory.COMPLIANCE_CHECK:
                    matching_items.append(item)
                elif ReviewerRole.SUBJECT_MATTER_EXPERT in reviewer.roles and any(
                    spec in item.metadata.get("specialization_tags", []) for spec in reviewer.specializations):
                    matching_items.append(item)
                elif ReviewerRole.GENERAL in reviewer.roles and item.category in [
                    ReviewCategory.CUSTOMER_RESPONSE, ReviewCategory.SENTIMENT_ANALYSIS, ReviewCategory.GENERAL]:
                    matching_items.append(item)
            
            # Sort by priority (critical -> high -> medium -> low)
            priority_order = {
                ReviewPriority.CRITICAL: 0,
                ReviewPriority.HIGH: 1,
                ReviewPriority.MEDIUM: 2,
                ReviewPriority.LOW: 3
            }
            
            matching_items.sort(key=lambda x: (
                priority_order[x.priority],
                x.timestamp  # Older items first
            ))
            
            if matching_items:
                next_item = matching_items[0]
                next_item.status = ReviewStatus.IN_PROGRESS
                next_item.reviewer_id = reviewer.reviewer_id
                reviewer.current_daily_reviews += 1
                return next_item
                
            return None
    
    def update_item(self, item_id: str, status: ReviewStatus, 
                   feedback: Optional[str] = None, 
                   edited_response: Optional[str] = None) -> bool:
        """Update review item with decision"""
        with self.lock:
            if item_id not in self.items:
                return False
                
            item = self.items[item_id]
            item.status = status
            item.review_timestamp = datetime.DateProvider.get_instance().now()
            
            if feedback:
                item.feedback = feedback
            
            if edited_response:
                item.edited_response = edited_response
                
            # Record review decision for model improvement
            self._record_review_decision(item)
            
            return True
    
    def _send_notifications(self, item: ReviewItem) -> None:
        """Send notifications for high priority reviews - Slack integration"""
        try:
            # Based on Confluent's Slack integration for critical reviews
            if os.getenv("SLACK_API_TOKEN") and os.getenv("SLACK_CHANNEL_ID"):
                client = slack_sdk.WebClient(token=os.getenv("SLACK_API_TOKEN"))
                
                # Format message based on category and priority
                emoji = "🔴" if item.priority == ReviewPriority.CRITICAL else "🟠"
                category_display = item.category.value.replace("_", " ").title()
                
                blocks = [
                    {
                        "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": f"{emoji} {category_display} Review Required"
                        }
                    },
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*ID:* {item.item_id}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Priority:* {item.priority.value}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Confidence:* {item.confidence_score:.2f}"
                            }
                        ]
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Query:*\n{item.query[:100]}..."
                        }
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Response:*\n{item.response[:100]}..."
                        }
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "Review Now"
                                },
                                "url": f"{os.getenv('REVIEW_DASHBOARD_URL', 'http://localhost:8501')}/review/{item.item_id}",
                                "style": "primary"
                            }
                        ]
                    }
                ]
                
                result = client.chat_postMessage(
                    channel=os.getenv("SLACK_CHANNEL_ID"),
                    text=f"New {item.priority.value} priority review required: {item.item_id}",
                    blocks=blocks
                )
                
                logger.info(f"Slack notification sent for item {item.item_id}")
                
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {str(e)}")
    
    def _record_review_decision(self, item: ReviewItem) -> None:
        """Record review decision for model improvement - based on Anthropic RLHF approach"""
        try:
            # Store review decision for future model fine-tuning
            review_data = {
                "item_id": item.item_id,
                "query": item.query,
                "model_response": item.response,
                "human_feedback": item.feedback,
                "human_edited_response": item.edited_response,
                "status": item.status.value,
                "model_id": item.model_id,
                "confidence_score": item.confidence_score,
                "category": item.category.value,
                "reviewer_id": item.reviewer_id,
                "timestamp": item.review_timestamp.isoformat() if item.review_timestamp else None
            }
            
            # Store in database for model improvement
            # This follows Anthropic's RLHF collection approach
            feedback_dir = os.path.join(os.getenv("DATA_DIR", "data"), "human_feedback")
            os.makedirs(feedback_dir, exist_ok=True)
            
            feedback_file = os.path.join(feedback_dir, f"{datetime.DateProvider.get_instance().now().strftime('%Y%m%d')}.jsonl")
            with open(feedback_file, "a") as f:
                f.write(json.dumps(review_data) + "\n")
                
            logger.info(f"Recorded review decision for item {item.item_id}")
            
        except Exception as e:
            logger.error(f"Failed to record review decision: {str(e)}")


class ReviewManager:
    """
    Review Manager implementation based on Scale AI and Appen review systems
    with workflow concepts from LinkedIn and Meta's review platforms
    """
    def __init__(self, database_url: Optional[str] = None):
        self.queues: Dict[str, ReviewQueue] = {}
        self.reviewers: Dict[str, Reviewer] = {}
        self.database_url = database_url or os.getenv("DATABASE_URI")
        self.database = Database(self.database_url) if self.database_url else None
        self.is_running = True
        self.expiry_thread = threading.Thread(target=self._check_expired_items, daemon=True)
        self.expiry_thread.start()
        
        # Create default queue
        self.create_queue("default")
        
        # Load reviewers from database
        self._load_reviewers()
        
    def create_queue(self, name: str, categories: List[ReviewCategory] = None, 
                    max_size: int = 1000) -> ReviewQueue:
        """Create a new review queue"""
        if name in self.queues:
            return self.queues[name]
            
        queue = ReviewQueue(name=name, categories=categories, max_size=max_size)
        self.queues[name] = queue
        return queue
    
    def get_queue(self, name: str) -> Optional[ReviewQueue]:
        """Get queue by name"""
        return self.queues.get(name)
    
    def add_reviewer(self, reviewer: Reviewer) -> bool:
        """Add a reviewer"""
        if reviewer.reviewer_id in self.reviewers:
            return False
            
        self.reviewers[reviewer.reviewer_id] = reviewer
        return True
    
    def get_reviewer(self, reviewer_id: str) -> Optional[Reviewer]:
        """Get reviewer by ID"""
        return self.reviewers.get(reviewer_id)
    
    def queue_item(self, item: ReviewItem, queue_name: str = "default") -> bool:
        """Queue an item for review"""
        queue = self.get_queue(queue_name)
        if not queue:
            return False
            
        # Calculate due time based on priority, following Asana's SLA approach
        if not item.due_by:
            now = datetime.DateProvider.get_instance().now()
            if item.priority == ReviewPriority.CRITICAL:
                item.due_by = now + datetime.timedelta(hours=1)
            elif item.priority == ReviewPriority.HIGH:
                item.due_by = now + datetime.timedelta(hours=4)
            elif item.priority == ReviewPriority.MEDIUM:
                item.due_by = now + datetime.timedelta(hours=24)
            else:  # LOW
                item.due_by = now + datetime.timedelta(hours=72)
        
        # Calculate risk score based on confidence and category
        # Following Google's risk assessment approach
        if item.risk_score is None:
            base_risk = 1.0 - item.confidence_score
            
            # Category risk multipliers
            category_risk = {
                ReviewCategory.FINANCIAL_ADVICE: 2.0,
                ReviewCategory.COMPLIANCE_CHECK: 1.8,
                ReviewCategory.CUSTOMER_RESPONSE: 1.2,
                ReviewCategory.DOCUMENT_ANALYSIS: 1.5,
                ReviewCategory.SENTIMENT_ANALYSIS: 0.8,
                ReviewCategory.GENERAL: 1.0
            }
            
            item.risk_score = base_risk * category_risk.get(item.category, 1.0)
            
            # If risk score is very high, escalate priority
            if item.risk_score > 0.8 and item.priority != ReviewPriority.CRITICAL:
                if item.priority == ReviewPriority.HIGH:
                    item.priority = ReviewPriority.CRITICAL
                elif item.priority == ReviewPriority.MEDIUM:
                    item.priority = ReviewPriority.HIGH
                elif item.priority == ReviewPriority.LOW:
                    item.priority = ReviewPriority.MEDIUM
                
                # Recalculate due_by based on new priority
                now = datetime.DateProvider.get_instance().now()
                if item.priority == ReviewPriority.CRITICAL:
                    item.due_by = now + datetime.timedelta(hours=1)
                elif item.priority == ReviewPriority.HIGH:
                    item.due_by = now + datetime.timedelta(hours=4)
        
        return queue.add_item(item)
    
    async def get_review_stats(self) -> Dict[str, Any]:
        """Get review statistics - based on Appen's analytics dashboard"""
        stats = {
            "total_pending": 0,
            "total_in_progress": 0,
            "total_completed": 0,
            "total_expired": 0,
            "by_priority": {
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0
            },
            "by_category": {},
            "average_time_to_review": 0,
            "reviewer_loads": [],
            "queue_sizes": []
        }
        
        total_review_time = datetime.timedelta()
        review_count = 0
        
        for queue_name, queue in self.queues.items():
            queue_stats = {
                "name": queue_name,
                "total": len(queue.items),
                "pending": 0,
                "in_progress": 0,
                "completed": 0,
                "expired": 0
            }
            
            for item in queue.items.values():
                # Update status counts
                if item.status == ReviewStatus.PENDING:
                    stats["total_pending"] += 1
                    queue_stats["pending"] += 1
                elif item.status == ReviewStatus.IN_PROGRESS:
                    stats["total_in_progress"] += 1
                    queue_stats["in_progress"] += 1
                elif item.status in [ReviewStatus.APPROVED, ReviewStatus.REJECTED, ReviewStatus.MODIFIED]:
                    stats["total_completed"] += 1
                    queue_stats["completed"] += 1
                    
                    # Calculate review time
                    if item.review_timestamp and item.timestamp:
                        review_time = item.review_timestamp - item.timestamp
                        total_review_time += review_time
                        review_count += 1
                elif item.status == ReviewStatus.EXPIRED:
                    stats["total_expired"] += 1
                    queue_stats["expired"] += 1
                
                # Update priority counts
                if item.priority == ReviewPriority.CRITICAL:
                    stats["by_priority"]["critical"] += 1
                elif item.priority == ReviewPriority.HIGH:
                    stats["by_priority"]["high"] += 1
                elif item.priority == ReviewPriority.MEDIUM:
                    stats["by_priority"]["medium"] += 1
                else:  # LOW
                    stats["by_priority"]["low"] += 1
                
                # Update category counts
                category = item.category.value
                if category not in stats["by_category"]:
                    stats["by_category"][category] = 0
                stats["by_category"][category] += 1
            
            stats["queue_sizes"].append(queue_stats)
        
        # Calculate average review time
        if review_count > 0:
            avg_seconds = total_review_time.total_seconds() / review_count
            stats["average_time_to_review"] = round(avg_seconds / 60, 2)  # in minutes
        
        # Calculate reviewer loads
        for reviewer_id, reviewer in self.reviewers.items():
            stats["reviewer_loads"].append({
                "reviewer_id": reviewer_id,
                "name": reviewer.name,
                "current_load": reviewer.current_daily_reviews,
                "max_load": reviewer.max_daily_reviews,
                "load_percentage": round((reviewer.current_daily_reviews / reviewer.max_daily_reviews) * 100, 2) if reviewer.max_daily_reviews > 0 else 0
            })
        
        return stats
    
    def _check_expired_items(self) -> None:
        """Check for expired items and update their status"""
        while self.is_running:
            now = datetime.DateProvider.get_instance().now()
            
            for queue in self.queues.values():
                with queue.lock:
                    for item in queue.items.values():
                        if item.status in [ReviewStatus.PENDING, ReviewStatus.IN_PROGRESS] and item.due_by and now > item.due_by:
                            # Item has expired
                            item.status = ReviewStatus.EXPIRED
                            
                            # For high priority items, send notification about expiry
                            if item.priority in [ReviewPriority.CRITICAL, ReviewPriority.HIGH]:
                                logger.warning(f"High priority review item {item.item_id} has expired")
                                
                                # If integration enabled, send escalation
                                if os.getenv("SLACK_API_TOKEN") and os.getenv("SLACK_ESCALATION_CHANNEL"):
                                    try:
                                        client = slack_sdk.WebClient(token=os.getenv("SLACK_API_TOKEN"))
                                        client.chat_postMessage(
                                            channel=os.getenv("SLACK_ESCALATION_CHANNEL"),
                                            text=f"⚠️ *ESCALATION*: Review item {item.item_id} ({item.category.value}) has expired without review. Priority: {item.priority.value}",
                                        )
                                    except Exception as e:
                                        logger.error(f"Failed to send escalation notification: {str(e)}")
            
            # Check every minute
            time.sleep(60)
    
    def _load_reviewers(self) -> None:
        """Load reviewers from database"""
        # Mock implementation - in production this would load from database
        default_reviewers = [
            Reviewer(
                reviewer_id="admin1",
                name="Admin User",
                email="admin@example.com",
                roles=[ReviewerRole.ADMINISTRATOR],
                specializations=["general"],
                max_daily_reviews=100
            ),
            Reviewer(
                reviewer_id="financial1",
                name="Financial Expert",
                email="financial@example.com",
                roles=[ReviewerRole.FINANCIAL],
                specializations=["investments", "loans", "credit"],
                max_daily_reviews=40
            ),
            Reviewer(
                reviewer_id="compliance1",
                name="Compliance Officer",
                email="compliance@example.com",
                roles=[ReviewerRole.COMPLIANCE],
                specializations=["regulations", "privacy"],
                max_daily_reviews=30
            ),
            Reviewer(
                reviewer_id="general1",
                name="General Reviewer",
                email="reviewer@example.com",
                roles=[ReviewerRole.GENERAL],
                max_daily_reviews=50
            )
        ]
        
        for reviewer in default_reviewers:
            self.add_reviewer(reviewer)
    
    def shutdown(self) -> None:
        """Shutdown the review manager"""
        self.is_running = False
        if self.expiry_thread.is_alive():
            self.expiry_thread.join(timeout=2.0)
        
        if self.database:
            asyncio.run(self.database.disconnect())


# Singleton instance
_review_manager = None

def get_review_manager() -> ReviewManager:
    """Get the global review manager instance"""
    global _review_manager
    if _review_manager is None:
        _review_manager = ReviewManager()
    return _review_manager 