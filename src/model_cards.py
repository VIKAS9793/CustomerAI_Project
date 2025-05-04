import os
import json
import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from src.utils.date_provider import DateProvider

class ModelPerformanceMetric(str, Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    ROC_AUC = "roc_auc"
    RMSE = "rmse"  # Regression
    MAE = "mae"  # Regression
    PERPLEXITY = "perplexity"  # Language models
    ROUGE = "rouge"  # Summarization
    BLEU = "bleu"  # Translation
    EXACT_MATCH = "exact_match"  # Question answering
    DEMOGRAPHIC_PARITY = "demographic_parity"  # Fairness
    EQUAL_OPPORTUNITY = "equal_opportunity"  # Fairness
    LATENCY = "latency"  # Performance
    THROUGHPUT = "throughput"  # Performance

class ModelCardGraphics(BaseModel):
    """Graphics for model cards following Google's model cards specification"""
    description: str
    image_path: str
    image_alt_text: Optional[str] = None

class ModelCardQuantitativeAnalysis(BaseModel):
    """Quantitative analysis section of a model card"""
    performance_metrics: Dict[str, Dict[str, float]]
    performance_by_group: Optional[Dict[str, Dict[str, float]]] = None
    decision_thresholds: Optional[Dict[str, float]] = None
    graphics: Optional[List[ModelCardGraphics]] = None
    
    @validator('performance_metrics')
    def validate_metrics(cls, metrics):
        """Validate that performance metrics are appropriate"""
        for group, group_metrics in metrics.items():
            for metric_name, value in group_metrics.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Metric {metric_name} must be a number, got {type(value)}")
                if value < 0 and metric_name not in [ModelPerformanceMetric.RMSE, ModelPerformanceMetric.MAE]:
                    raise ValueError(f"Metric {metric_name} should not be negative")
        return metrics

class ModelCardConsiderations(BaseModel):
    """Considerations section of a model card"""
    users: List[str]
    use_cases: List[str]
    limitations: List[str]
    tradeoffs: List[str]
    ethical_considerations: List[str]
    fairness_considerations: Optional[List[str]] = None
    
class ModelCardDataset(BaseModel):
    """Dataset information for model cards"""
    name: str
    link: Optional[str] = None
    sensitive_data: bool = False
    graphics: Optional[List[ModelCardGraphics]] = None
    description: str
    data_splits: Optional[Dict[str, int]] = None  # train/test/validation sizes
    data_format: Optional[str] = None
    known_biases: Optional[List[str]] = None
    preprocessing: Optional[List[str]] = None

class ModelCardModelDetails(BaseModel):
    """Model details section of a model card"""
    name: str
    version: str
    type: str  # e.g., classification, regression, language model
    architecture: str
    input_format: str
    output_format: str
    license: Optional[str] = None
    references: Optional[List[str]] = None
    citation: Optional[str] = None
    developers: List[str]
    sensitive_elements: Optional[Dict[str, str]] = None
    compliance_specifications: Optional[List[str]] = None
    # e.g., GDPR, CCPA, HIPAA, etc.

class ModelCard(BaseModel):
    """
    Comprehensive model card following Google's model cards specification
    with additional elements from Hugging Face and OpenAI's model card formats
    """
    name: str
    version: str
    last_updated: datetime.datetime = Field(default_factory=datetime.datetime.now)
    document_id: str = Field(default_factory=lambda: f"model-card-{datetime.DateProvider.get_instance().now().strftime('%Y%m%d-%H%M%S')}")
    model_details: ModelCardModelDetails
    model_description: str
    quantitative_analysis: ModelCardQuantitativeAnalysis
    considerations: ModelCardConsiderations
    datasets: List[ModelCardDataset]
    
    # Additional fields based on industry standards
    finetuning_details: Optional[Dict[str, Any]] = None
    environmental_impact: Optional[Dict[str, Any]] = None
    model_provenance: Optional[Dict[str, Any]] = None
    regulatory_compliance: Optional[Dict[str, Any]] = None
    known_vulnerabilities: Optional[List[str]] = None
    testing_procedures: Optional[List[str]] = None
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime.datetime: lambda v: v.isoformat()
        }
    
    def to_json(self) -> str:
        """Convert model card to JSON string"""
        return json.dumps(self.dict(), indent=2, default=str)
    
    def to_html(self) -> str:
        """Generate HTML representation of the model card"""
        # Basic HTML template that could be extended with styles and JS
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Model Card: {self.name} v{self.version}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1000px; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .section {{ margin-bottom: 30px; border: 1px solid #eee; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin-right: 20px; margin-bottom: 10px; }}
                .metric-value {{ font-size: 1.2em; font-weight: bold; }}
                .metric-name {{ font-size: 0.9em; color: #7f8c8d; }}
                .consideration {{ margin-bottom: 10px; }}
                .consideration-item {{ margin-left: 20px; margin-bottom: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .footer {{ margin-top: 30px; font-size: 0.8em; color: #7f8c8d; text-align: center; }}
            </style>
        </head>
        <body>
            <h1>Model Card: {self.name}</h1>
            <p>Version: {self.version} | Last Updated: {self.last_updated.strftime('%Y-%m-%d')}</p>
            
            <div class="section">
                <h2>Model Description</h2>
                <p>{self.model_description}</p>
            </div>
            
            <div class="section">
                <h2>Model Details</h2>
                <table>
                    <tr><th>Name</th><td>{self.model_details.name}</td></tr>
                    <tr><th>Version</th><td>{self.model_details.version}</td></tr>
                    <tr><th>Type</th><td>{self.model_details.type}</td></tr>
                    <tr><th>Architecture</th><td>{self.model_details.architecture}</td></tr>
                    <tr><th>Input Format</th><td>{self.model_details.input_format}</td></tr>
                    <tr><th>Output Format</th><td>{self.model_details.output_format}</td></tr>
                    <tr><th>License</th><td>{self.model_details.license or 'Not specified'}</td></tr>
                    <tr><th>Developers</th><td>{', '.join(self.model_details.developers)}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Quantitative Analysis</h2>
                <h3>Performance Metrics</h3>
        """
        
        # Add performance metrics
        for group, metrics in self.quantitative_analysis.performance_metrics.items():
            html += f"<h4>{group}</h4>"
            html += "<div>"
            for metric_name, value in metrics.items():
                html += f"""
                <div class="metric">
                    <div class="metric-value">{value:.4f}</div>
                    <div class="metric-name">{metric_name}</div>
                </div>
                """
            html += "</div>"
        
        # Add performance by group if available
        if self.quantitative_analysis.performance_by_group:
            html += "<h3>Performance by Group</h3>"
            html += "<table><tr><th>Group</th>"
            
            # Get all metric names
            all_metrics = set()
            for group_metrics in self.quantitative_analysis.performance_by_group.values():
                all_metrics.update(group_metrics.keys())
            
            # Create header row
            for metric in sorted(all_metrics):
                html += f"<th>{metric}</th>"
            html += "</tr>"
            
            # Add rows for each group
            for group, group_metrics in self.quantitative_analysis.performance_by_group.items():
                html += f"<tr><td>{group}</td>"
                for metric in sorted(all_metrics):
                    value = group_metrics.get(metric, '')
                    html += f"<td>{value:.4f if isinstance(value, (int, float)) else value}</td>"
                html += "</tr>"
            
            html += "</table>"
        
        # Continue with considerations
        html += """
            </div>
            
            <div class="section">
                <h2>Considerations</h2>
        """
        
        # Add each consideration type
        consideration_types = [
            ("Intended Users", self.considerations.users),
            ("Use Cases", self.considerations.use_cases),
            ("Limitations", self.considerations.limitations),
            ("Tradeoffs", self.considerations.tradeoffs),
            ("Ethical Considerations", self.considerations.ethical_considerations)
        ]
        
        if self.considerations.fairness_considerations:
            consideration_types.append(("Fairness Considerations", self.considerations.fairness_considerations))
        
        for title, items in consideration_types:
            html += f"""
                <div class="consideration">
                    <h3>{title}</h3>
                    <ul>
            """
            for item in items:
                html += f"<li class='consideration-item'>{item}</li>"
            html += "</ul></div>"
        
        # Add datasets
        html += """
            </div>
            
            <div class="section">
                <h2>Datasets</h2>
        """
        
        for dataset in self.datasets:
            dataset_link = f"<a href='{dataset.link}'>{dataset.name}</a>" if dataset.link else dataset.name
            html += f"""
                <div class="consideration">
                    <h3>{dataset_link}</h3>
                    <p>{dataset.description}</p>
            """
            if dataset.sensitive_data:
                html += "<p><strong>Note:</strong> Contains sensitive data</p>"
            
            if dataset.known_biases:
                html += "<h4>Known Biases</h4><ul>"
                for bias in dataset.known_biases:
                    html += f"<li>{bias}</li>"
                html += "</ul>"
            
            html += "</div>"
        
        # Footer
        html += f"""
            </div>
            
            <div class="footer">
                <p>Generated by CustomerAI Model Cards | Document ID: {self.document_id}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def save(self, directory: str, format_type: str = "json") -> str:
        """
        Save model card to file
        
        Args:
            directory: Directory to save model card to
            format_type: Format to save as (json, html, or both)
            
        Returns:
            Path to saved file
        """
        os.makedirs(directory, exist_ok=True)
        
        base_filename = f"{self.model_details.name.replace(' ', '_')}_v{self.version}"
        
        if format_type == "json" or format_type == "both":
            json_path = os.path.join(directory, f"{base_filename}.json")
            with open(json_path, "w") as f:
                f.write(self.to_json())
        
        if format_type == "html" or format_type == "both":
            html_path = os.path.join(directory, f"{base_filename}.html")
            with open(html_path, "w") as f:
                f.write(self.to_html())
        
        return os.path.join(directory, base_filename + "." + format_type)


def create_example_model_card() -> ModelCard:
    """Create an example model card for a sentiment analysis model"""
    return ModelCard(
        name="Financial Sentiment Analyzer",
        version="1.2.0",
        model_description="""
        This model analyzes sentiment in financial texts such as customer feedback, 
        financial news, and earnings calls. It's specifically trained to identify positive,
        negative, and neutral sentiment in financial contexts, with particular attention
        to domain-specific terminology and expressions.
        """,
        model_details=ModelCardModelDetails(
            name="FinSentiment-Large",
            version="1.2.0",
            type="classification",
            architecture="DistilBERT fine-tuned with financial corpus",
            input_format="Text string up to 512 tokens",
            output_format="Sentiment classification (positive, negative, neutral) with confidence scores",
            license="MIT",
            references=[
                "https://huggingface.co/docs/transformers/model_doc/distilbert",
                "https://arxiv.org/abs/1910.01108"
            ],
            developers=["CustomerAI Financial ML Team", "FinTech Research Group"],
            compliance_specifications=["GDPR", "CCPA", "SOC 2"],
        ),
        quantitative_analysis=ModelCardQuantitativeAnalysis(
            performance_metrics={
                "Overall": {
                    "accuracy": 0.891,
                    "precision": 0.876,
                    "recall": 0.863,
                    "f1": 0.869,
                },
                "Positive Sentiment": {
                    "precision": 0.903,
                    "recall": 0.897,
                    "f1": 0.900,
                },
                "Negative Sentiment": {
                    "precision": 0.856,
                    "recall": 0.841,
                    "f1": 0.848,
                },
                "Neutral Sentiment": {
                    "precision": 0.842,
                    "recall": 0.819,
                    "f1": 0.830,
                }
            },
            performance_by_group={
                "Financial News": {
                    "accuracy": 0.912,
                    "f1": 0.896
                },
                "Customer Feedback": {
                    "accuracy": 0.874,
                    "f1": 0.857
                },
                "Earnings Calls": {
                    "accuracy": 0.889,
                    "f1": 0.871
                }
            },
            decision_thresholds={
                "positive": 0.6,
                "negative": 0.6,
                "neutral": 0.5
            }
        ),
        considerations=ModelCardConsiderations(
            users=[
                "Financial analysts and researchers",
                "Customer service representatives",
                "Compliance and risk management professionals",
                "Financial news aggregation services",
                "Investment advisory firms"
            ],
            use_cases=[
                "Analyzing customer feedback for financial services",
                "Monitoring sentiment in financial news",
                "Evaluating customer satisfaction in banking services",
                "Analyzing sentiment trends in earnings calls",
                "Real-time monitoring of social media for financial sentiment"
            ],
            limitations=[
                "May not accurately interpret sarcasm or subtle language nuances",
                "Limited to English text only",
                "May not perform well on non-financial domains",
                "Maximum text length of 512 tokens",
                "Baseline performance may degrade for very technical financial jargon"
            ],
            tradeoffs=[
                "Optimized for precision in financial domain at expense of general sentiment understanding",
                "Model size balanced for performance vs. computational efficiency",
                "Optimized for Western markets and may not perform well on emerging markets terminology"
            ],
            ethical_considerations=[
                "Model outputs should not be the sole basis for investment decisions",
                "Potential for reinforcing existing biases in financial language",
                "Should not be used to analyze personal communications without consent",
                "Automated sentiment analysis should be supplemented with human judgment for high-stakes decisions"
            ],
            fairness_considerations=[
                "Model may perform differently across demographic groups when analyzing customer feedback",
                "Model was evaluated for performance across multiple geographic regions with different financial terminologies",
                "Balanced training data was used to minimize gender and cultural biases"
            ]
        ),
        datasets=[
            ModelCardDataset(
                name="FinancialPhraseBank",
                link="https://huggingface.co/datasets/financial_phrasebank",
                sensitive_data=False,
                description="A collection of financial news sentences manually annotated for sentiment",
                data_splits={
                    "train": 3000,
                    "validation": 500,
                    "test": 500
                },
                data_format="Text with sentiment labels",
                known_biases=["Over-representation of Western financial markets", "Limited coverage of emerging markets"]
            ),
            ModelCardDataset(
                name="CustomerAI Financial Feedback",
                sensitive_data=True,
                description="Proprietary dataset of anonymized customer feedback from financial services",
                data_splits={
                    "train": 25000,
                    "validation": 5000,
                    "test": 5000
                },
                known_biases=["Geographical concentration in North America and Europe"]
            )
        ],
        regulatory_compliance={
            "gdpr_assessment": "Completed with Privacy Impact Assessment",
            "regulatory_approvals": ["ISO 27001", "SOC 2 Type II"],
            "data_privacy": "Model does not store or retain input data"
        },
        environmental_impact={
            "training_computational_cost": "Approximately 45 kWh",
            "inference_computational_cost": "0.02 kWh per 1000 inferences"
        }
    )


def get_model_card_registry():
    """Get or create the model card registry singleton"""
    if not hasattr(get_model_card_registry, "registry"):
        registry_dir = os.path.join(os.getenv("DATA_DIR", "data"), "model_cards")
        os.makedirs(registry_dir, exist_ok=True)
        get_model_card_registry.registry = ModelCardRegistry(registry_dir)
    return get_model_card_registry.registry


class ModelCardRegistry:
    """Registry for model cards following industrial MLOps practices"""
    
    def __init__(self, registry_dir: str):
        self.registry_dir = registry_dir
        self.model_cards: Dict[str, ModelCard] = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load model cards from registry directory"""
        try:
            for filename in os.listdir(self.registry_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(self.registry_dir, filename)
                    with open(file_path, "r") as f:
                        model_card_data = json.load(f)
                        # Convert the data back to a ModelCard
                        model_card = ModelCard(**model_card_data)
                        # Use model_name@version as key
                        key = f"{model_card.model_details.name}@{model_card.version}"
                        self.model_cards[key] = model_card
        except Exception as e:
            # If registry is empty or there's an error, it's okay
            print(f"Warning loading model card registry: {str(e)}")
    
    def register_model_card(self, model_card: ModelCard) -> bool:
        """Register a model card in the registry"""
        key = f"{model_card.model_details.name}@{model_card.version}"
        
        # Save to registry directory
        model_card.save(self.registry_dir, format_type="both")
        
        # Add to in-memory registry
        self.model_cards[key] = model_card
        return True
    
    def get_model_card(self, model_name: str, version: Optional[str] = None) -> Optional[ModelCard]:
        """
        Get a model card from the registry
        
        Args:
            model_name: Name of the model
            version: Version of the model, if None, latest version is returned
            
        Returns:
            ModelCard if found, None otherwise
        """
        if version is None:
            # Find latest version
            latest_version = None
            latest_card = None
            
            for key, card in self.model_cards.items():
                if card.model_details.name == model_name:
                    if latest_version is None or card.version > latest_version:
                        latest_version = card.version
                        latest_card = card
            
            return latest_card
        
        key = f"{model_name}@{version}"
        return self.model_cards.get(key)
    
    def list_model_cards(self) -> List[Dict[str, str]]:
        """List all model cards in the registry"""
        return [
            {
                "name": card.model_details.name,
                "version": card.version,
                "description": card.model_description[:100] + "..." if len(card.model_description) > 100 else card.model_description,
                "last_updated": card.last_updated.isoformat()
            }
            for card in self.model_cards.values()
        ]
    
    def get_model_cards_by_type(self, model_type: str) -> List[ModelCard]:
        """Get all model cards of a specific type"""
        return [
            card for card in self.model_cards.values()
            if card.model_details.type == model_type
        ]
    
    def get_model_cards_by_compliance(self, compliance_spec: str) -> List[ModelCard]:
        """Get all model cards that meet a compliance specification"""
        return [
            card for card in self.model_cards.values()
            if card.model_details.compliance_specifications and
            compliance_spec in card.model_details.compliance_specifications
        ]


# Create example card if this file is run directly
if __name__ == "__main__":
    # Create example model card
    example_card = create_example_model_card()
    
    # Save to registry
    registry = get_model_card_registry()
    registry.register_model_card(example_card)
    
    # Print registry contents
    print("Model Card Registry Contents:")
    for item in registry.list_model_cards():
        print(f"- {item['name']} (v{item['version']})")
    
    # Save example as standalone files
    out_dir = "model_cards"
    os.makedirs(out_dir, exist_ok=True)
    example_card.save(out_dir, format_type="both")
    print(f"Example model card saved to {out_dir}/") 