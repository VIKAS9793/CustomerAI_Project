# CustomerAI Insights Platform - Usage Examples

This document provides step-by-step examples for using the CustomerAI Insights Platform's various features and functionalities.

## Table of Contents

1. [Setting Up the Environment](#setting-up-the-environment)
2. [Sentiment Analysis](#sentiment-analysis)
3. [Response Generation](#response-generation)
4. [Privacy Protection](#privacy-protection)
5. [Bias Detection](#bias-detection)
6. [Human Review Workflow](#human-review-workflow)
7. [Dashboard Analytics](#dashboard-analytics)
8. [Integration Examples](#integration-examples)

## Setting Up the Environment

### Basic Setup

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/username/customerai-insights.git
cd customerai-insights
```

2. Create and activate a virtual environment:

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:

```bash
# For Windows
copy env.example .env
notepad .env

# For macOS/Linux
cp env.example .env
nano .env
```

5. Initialize the database:

```bash
python -m scripts.init_db
```

6. Start the API server:

```bash
uvicorn api.main:app --reload
```

7. In a separate terminal, start the dashboard:

```bash
streamlit run app.py
```

## Sentiment Analysis

### Analyzing a Single Customer Interaction

Using the API directly:

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/api/v1/analyze/sentiment"

# Request headers
headers = {
    "Authorization": "Bearer YOUR_JWT_TOKEN",
    "Content-Type": "application/json"
}

# Request data
data = {
    "text": "I've been trying to reach customer service for days with no response!",
    "use_ai": True
}

# Send request
response = requests.post(url, headers=headers, json=data)

# Print response
print(json.dumps(response.json(), indent=2))
```

Expected output:

```json
{
  "error": false,
  "status_code": 200,
  "data": {
    "sentiment": "negative",
    "positive": 0.03,
    "negative": 0.92,
    "neutral": 0.05,
    "analysis": {
      "satisfaction_score": 2,
      "key_negatives": ["response time", "customer service availability"],
      "urgency": "high"
    }
  },
  "timestamp": "2023-08-15T15:23:45.123456"
}
```

### Batch Analysis of Multiple Customer Interactions

```python
import requests
import json
import pandas as pd

# API endpoint
url = "http://localhost:8000/api/v1/analyze/batch"

# Request headers
headers = {
    "Authorization": "Bearer YOUR_JWT_TOKEN",
    "Content-Type": "application/json"
}

# Sample data in a DataFrame
conversations = pd.DataFrame({
    'id': ['conv1', 'conv2', 'conv3'],
    'text': [
        "Thank you for resolving my issue so quickly.",
        "I'm still waiting for my refund after three weeks.",
        "Could you explain how to check my account balance online?"
    ]
})

# Convert to the format expected by the API
request_data = {
    "conversations": [
        {"id": row.id, "text": row.text} 
        for _, row in conversations.iterrows()
    ]
}

# Send request
response = requests.post(url, headers=headers, json=request_data)

# Print response
print(json.dumps(response.json(), indent=2))
```

## Response Generation

### Generating a Response to a Customer Query

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/api/v1/generate/response"

# Request headers
headers = {
    "Authorization": "Bearer YOUR_JWT_TOKEN",
    "Content-Type": "application/json"
}

# Request data
data = {
    "query": "How can I check if my loan application has been approved?",
    "customer_id": "cust123",
    "context": {
        "previous_interactions": 2,
        "product": "personal_loan"
    }
}

# Send request
response = requests.post(url, headers=headers, json=data)

# Print response
print(json.dumps(response.json(), indent=2))
```

### Handling Financial Advice Queries

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/api/v1/generate/response"

# Request headers
headers = {
    "Authorization": "Bearer YOUR_JWT_TOKEN",
    "Content-Type": "application/json"
}

# Request data
data = {
    "query": "What stocks should I invest in for maximum returns?",
    "customer_id": "cust456",
    "context": {
        "risk_profile": "moderate",
        "investment_horizon": "long_term"
    }
}

# Send request
response = requests.post(url, headers=headers, json=data)

# Check if human review is required
result = response.json()
if result['data'].get('requires_human_review', False):
    print("This response requires human review before being sent to the customer.")
else:
    print("Response:", result['data']['response'])
```

## Privacy Protection

### Anonymizing Customer Data

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/api/v1/privacy/anonymize"

# Request headers
headers = {
    "Authorization": "Bearer YOUR_JWT_TOKEN",
    "Content-Type": "application/json"
}

# Request data
data = {
    "text": "Hello, my name is Jane Smith and my account number is 1234567890. My phone number is 555-123-4567 and my email is jane.smith@example.com.",
    "keep_mapping": True
}

# Send request
response = requests.post(url, headers=headers, json=data)

# Print response
result = response.json()
print("Anonymized text:", result['data']['anonymized_text'])
print("\nMapping:", json.dumps(result['data']['mapping'], indent=2))
```

### Processing a Dataset with PII

```python
import requests
import json
import pandas as pd

# Function to anonymize a dataset with customer information
def anonymize_dataset(df, text_column):
    # API endpoint
    url = "http://localhost:8000/api/v1/privacy/anonymize"
    
    # Request headers
    headers = {
        "Authorization": "Bearer YOUR_JWT_TOKEN",
        "Content-Type": "application/json"
    }
    
    # Process each row
    anonymized_texts = []
    for text in df[text_column]:
        data = {"text": text, "keep_mapping": False}
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        anonymized_texts.append(result['data']['anonymized_text'])
    
    # Create new dataframe with anonymized text
    df_anonymized = df.copy()
    df_anonymized[text_column + '_anonymized'] = anonymized_texts
    
    return df_anonymized

# Example usage
customer_data = pd.DataFrame({
    'customer_id': ['cust1', 'cust2', 'cust3'],
    'feedback': [
        "My name is John Doe and I'm unhappy with your service.",
        "Please update my phone number to 555-987-6543.",
        "Email me the statement at mary.johnson@example.com."
    ]
})

anonymized_data = anonymize_dataset(customer_data, 'feedback')
print(anonymized_data)
```

## Bias Detection

### Analyzing Dataset for Demographic Bias

```python
import requests
import json
import pandas as pd
import numpy as np

# Create a sample dataset with demographic information
np.random.seed(42)  # For reproducible results

# Generate synthetic data
age_groups = ['18-30', '31-45', '46-60', '60+']
genders = ['male', 'female', 'non-binary']
income_levels = ['low', 'medium', 'high']

n_samples = 200
data = []

for _ in range(n_samples):
    age_group = np.random.choice(age_groups)
    gender = np.random.choice(genders)
    income = np.random.choice(income_levels)
    
    # Introduce a bias based on age_group for this example
    if age_group == '60+':
        satisfaction = max(1, np.random.normal(2.5, 0.5))  # Lower satisfaction for older age group
        resolved = np.random.choice([True, False], p=[0.6, 0.4])  # Lower resolution rate
    else:
        satisfaction = min(5, np.random.normal(4.2, 0.7))  # Higher satisfaction for others
        resolved = np.random.choice([True, False], p=[0.9, 0.1])  # Higher resolution rate
    
    data.append({
        'age_group': age_group,
        'gender': gender,
        'income_level': income,
        'satisfaction_score': round(satisfaction),
        'resolved': resolved
    })

# Create dataframe
df = pd.DataFrame(data)

# Send to bias detection API
url = "http://localhost:8000/api/v1/fairness/analyze"

headers = {
    "Authorization": "Bearer YOUR_JWT_TOKEN",
    "Content-Type": "application/json"
}

request_data = {
    "data": df.to_dict(orient='records'),
    "attributes": ["age_group", "gender", "income_level"],
    "outcome_columns": ["satisfaction_score", "resolved"]
}

# Send request
response = requests.post(url, headers=headers, json=request_data)

# Print fairness report
fairness_report = response.json()['data']
print("Fairness score:", fairness_report['fairness_score'])
print("\nDetailed findings:")
for finding in fairness_report['detailed_findings']:
    print(f"\n- {finding['attribute']} → {finding['outcome']}")
    print(f"  Disparity score: {finding['disparity_score']}")
    print(f"  Concern level: {finding['concern_level']}")

print("\nRecommendations:")
for rec in fairness_report['recommendations']:
    print(f"- {rec}")
```

## Human Review Workflow

### Submitting an Item for Review

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/api/v1/review/queue"

# Request headers
headers = {
    "Authorization": "Bearer YOUR_JWT_TOKEN",
    "Content-Type": "application/json"
}

# Request data
data = {
    "query": "What's the best investment strategy for retirement?",
    "response": "Based on general principles, a diversified portfolio with a mix of stocks, bonds, and other assets that gradually becomes more conservative as you approach retirement age is often recommended. However, the optimal strategy depends on your age, risk tolerance, and financial goals. I'd be happy to discuss this further during a consultation where we can analyze your specific situation.",
    "category": "investment_advice",
    "priority": 2,  # High priority (scale 1-3)
    "metadata": {
        "customer_id": "cust123",
        "interaction_id": "int456"
    }
}

# Send request
response = requests.post(url, headers=headers, json=data)

# Store the item_id for future reference
result = response.json()
item_id = result['data']['item_id']
print(f"Item queued for review with ID: {item_id}")
print(f"Estimated review time: {result['data']['estimated_review_time']}")
```

### Reviewing and Approving Items

```python
import requests
import json

# API endpoint for retrieving next item
get_url = "http://localhost:8000/api/v1/review/next"
decision_url = "http://localhost:8000/api/v1/review/decision"

# Request headers
headers = {
    "Authorization": "Bearer YOUR_JWT_TOKEN",
    "Content-Type": "application/json"
}

# Get next item for review
response = requests.get(get_url, headers=headers)
review_item = response.json()['data']

print(f"Reviewing item: {review_item['item_id']}")
print(f"Query: {review_item['query']}")
print(f"Proposed response: {review_item['response']}")
print(f"Category: {review_item['category']}")

# Submit review decision
decision_data = {
    "item_id": review_item['item_id'],
    "approved": True,  # Approve the response
    "feedback": "Response is accurate and includes appropriate disclaimers",
    "edits": {
        "modified_response": review_item['response'] + " Please note that this information is general advice only and not specific to your individual financial situation."
    }
}

# Send decision
decision_response = requests.post(decision_url, headers=headers, json=decision_data)
result = decision_response.json()

if result['error'] == False:
    print("Review completed successfully!")
else:
    print(f"Error: {result['message']}")
```

## Dashboard Analytics

### Accessing Dashboard Data Programmatically

```python
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# API endpoint
url = "http://localhost:8000/api/v1/analytics/summary"

# Request headers
headers = {
    "Authorization": "Bearer YOUR_JWT_TOKEN",
    "Content-Type": "application/json"
}

# Set date range for last 30 days
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

# Request parameters
params = {
    "start_date": start_date,
    "end_date": end_date,
    "granularity": "daily"
}

# Send request
response = requests.get(url, headers=headers, params=params)
result = response.json()['data']

# Extract timeline data
timeline_df = pd.DataFrame(result['timeline_data'])
timeline_df['date'] = pd.to_datetime(timeline_df['date'])

# Create a simple visualization
plt.figure(figsize=(12, 6))

# Plot conversations count
ax1 = plt.subplot(2, 1, 1)
ax1.plot(timeline_df['date'], timeline_df['conversations'], 'b-', marker='o')
ax1.set_ylabel('Conversations')
ax1.set_title('Customer Conversations Over Time')
ax1.grid(True)

# Plot average sentiment
ax2 = plt.subplot(2, 1, 2)
ax2.plot(timeline_df['date'], timeline_df['avg_sentiment'], 'g-', marker='o')
ax2.set_ylabel('Avg Sentiment')
ax2.set_xlabel('Date')
ax2.grid(True)

plt.tight_layout()
plt.savefig('analytics_summary.png')
plt.close()

print("Analytics summary:")
print(f"Total conversations: {result['total_conversations']}")
print(f"Average satisfaction: {result['average_satisfaction']}")
print(f"Resolution rate: {result['resolution_rate'] * 100:.1f}%")
print("\nSentiment distribution:")
for sentiment, value in result['sentiment_distribution'].items():
    print(f"- {sentiment}: {value * 100:.1f}%")

print("\nChart saved as 'analytics_summary.png'")
```

## Integration Examples

### Integrating with Customer Support Systems

This example shows how to integrate the CustomerAI API with a ticket management system:

```python
import requests
import json
import time

class CustomerAIIntegration:
    """Integration class for CustomerAI with support ticket systems"""
    
    def __init__(self, api_base_url, api_token):
        self.api_base_url = api_base_url
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    def process_ticket(self, ticket_id, customer_message, customer_id=None):
        """Process a support ticket through CustomerAI"""
        
        # Step 1: Analyze sentiment
        sentiment_data = self._analyze_sentiment(customer_message)
        
        # Step 2: Generate response
        response_data = self._generate_response(
            customer_message, 
            customer_id=customer_id,
            sentiment=sentiment_data
        )
        
        # Step 3: If high-risk, queue for human review
        if response_data.get('requires_human_review', False):
            review_data = self._queue_for_review(
                customer_message,
                response_data['response'],
                category=response_data['category'],
                metadata={"ticket_id": ticket_id, "customer_id": customer_id}
            )
            
            return {
                "ticket_id": ticket_id,
                "status": "pending_review",
                "sentiment": sentiment_data,
                "suggested_response": response_data['response'],
                "review_id": review_data['item_id']
            }
        
        # Step 4: Return processed result
        return {
            "ticket_id": ticket_id,
            "status": "processed",
            "sentiment": sentiment_data,
            "response": response_data['response'],
            "confidence": response_data.get('confidence', 0)
        }
    
    def _analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        url = f"{self.api_base_url}/analyze/sentiment"
        data = {"text": text, "use_ai": True}
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()['data']
    
    def _generate_response(self, query, customer_id=None, sentiment=None):
        """Generate response to customer query"""
        url = f"{self.api_base_url}/generate/response"
        data = {
            "query": query,
            "customer_id": customer_id,
            "context": {}
        }
        
        # Add sentiment context if available
        if sentiment:
            data["context"]["sentiment"] = sentiment['sentiment']
            data["context"]["satisfaction_score"] = sentiment.get('analysis', {}).get('satisfaction_score')
        
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()['data']
    
    def _queue_for_review(self, query, response, category=None, metadata=None):
        """Queue an item for human review"""
        url = f"{self.api_base_url}/review/queue"
        data = {
            "query": query,
            "response": response,
            "category": category or "general",
            "priority": 2,  # Default to high priority
            "metadata": metadata or {}
        }
        
        response = requests.post(url, headers=self.headers, json=data)
        return response.json()['data']
    
    def check_review_status(self, review_id):
        """Check status of a review"""
        url = f"{self.api_base_url}/review/status/{review_id}"
        response = requests.get(url, headers=self.headers)
        return response.json()['data']

# Example usage
if __name__ == "__main__":
    # Initialize integration
    integration = CustomerAIIntegration(
        api_base_url="http://localhost:8000/api/v1",
        api_token="YOUR_JWT_TOKEN"
    )
    
    # Process a support ticket
    result = integration.process_ticket(
        ticket_id="T12345",
        customer_message="I've been charged twice for my monthly premium and need this resolved immediately.",
        customer_id="C98765"
    )
    
    print(json.dumps(result, indent=2))
    
    # If sent for review, check status (in real system, this would be a separate process)
    if result['status'] == 'pending_review':
        print("Waiting for human review...")
        
        # Poll for review completion (simplified example)
        for _ in range(5):  # Try 5 times
            time.sleep(2)  # Wait 2 seconds between checks
            status = integration.check_review_status(result['review_id'])
            
            if status['status'] == 'reviewed':
                print("Review completed!")
                print(f"Approved: {status['approved']}")
                print(f"Response: {status['final_response']}")
                break
        else:
            print("Review still pending...")
```

### Batch Processing Historical Conversations

```python
import pandas as pd
import requests
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Load historical conversations (example with CSV file)
conversations = pd.read_csv('historical_conversations.csv')

# API configuration
API_BASE_URL = "http://localhost:8000/api/v1"
API_TOKEN = "YOUR_JWT_TOKEN"
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# Function to process a single conversation
def process_conversation(row):
    # Analyze sentiment
    sentiment_url = f"{API_BASE_URL}/analyze/sentiment"
    sentiment_data = {
        "text": row['customer_message'],
        "use_ai": True
    }
    
    try:
        sentiment_response = requests.post(
            sentiment_url, 
            headers=HEADERS, 
            json=sentiment_data,
            timeout=10
        )
        sentiment_result = sentiment_response.json()['data']
        
        # Return combined results
        return {
            'conversation_id': row['conversation_id'],
            'customer_id': row['customer_id'],
            'date': row['date'],
            'message': row['customer_message'],
            'sentiment': sentiment_result['sentiment'],
            'positive_score': sentiment_result['positive'],
            'negative_score': sentiment_result['negative'],
            'satisfaction_score': sentiment_result.get('analysis', {}).get('satisfaction_score')
        }
    except Exception as e:
        return {
            'conversation_id': row['conversation_id'],
            'error': str(e)
        }

# Process in parallel with progress bar
results = []

print(f"Processing {len(conversations)} historical conversations...")
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(process_conversation, row) for _, row in conversations.iterrows()]
    
    for future in tqdm(futures, total=len(futures)):
        results.append(future.result())

# Convert results to DataFrame
results_df = pd.DataFrame([r for r in results if 'error' not in r])
errors_df = pd.DataFrame([r for r in results if 'error' in r])

# Save results
results_df.to_csv('processed_conversations.csv', index=False)

print(f"Processing complete. {len(results_df)} conversations processed successfully.")
if not errors_df.empty:
    print(f"{len(errors_df)} conversations had errors.")
    errors_df.to_csv('processing_errors.csv', index=False)
```

These examples demonstrate common usage patterns for the CustomerAI Insights Platform. For more specific use cases or advanced integrations, please refer to the full API documentation. 