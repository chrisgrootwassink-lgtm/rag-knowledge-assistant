"""
Generates synthetic demo documents in the data/ folder.
Run once before launching the chatbot or graph builder:

    python create_demo_data.py
    python "vectorstore creator.py"
    streamlit run chatbot.py

The demo covers three fictional interviews about technology adoption at a
software company, giving enough content to ask questions like:
  - "What cloud platforms are used and why?"
  - "How does the team handle CI/CD and code quality?"
  - "What AI tools have been adopted and what challenges came up?"
"""
from pathlib import Path
from docx import Document


# Each interview is a list of (speaker, text) tuples.
# "I" = interviewer, "R" = respondent.
INTERVIEWS = [
    {
        "filename": "demo_cloud_infrastructure.docx",
        "turns": [
            ("I", "Can you walk me through the cloud infrastructure you currently run?"),
            ("R", "Sure. We migrated to Microsoft Azure about two years ago. Before that everything ran on-premises, which meant long provisioning cycles and a lot of manual work for the operations team. Azure gave us Infrastructure-as-Code through Terraform, which was a huge step forward."),
            ("I", "What drove the decision to choose Azure over AWS or GCP?"),
            ("R", "Primarily existing enterprise agreements and the integration with Active Directory. Our organisation already used Microsoft 365 heavily, so identity management was much simpler. That said, we evaluated all three and Azure won mostly on the enterprise side, not pure technical merit."),
            ("I", "How is the infrastructure organised?"),
            ("R", "We run everything on Azure Kubernetes Service — AKS. All our microservices are containerised with Docker. Helm charts manage the deployments, and ArgoCD handles the GitOps workflow so that every merge to main triggers a deployment automatically."),
            ("I", "What does your monitoring stack look like?"),
            ("R", "Prometheus for metrics collection, Grafana for dashboards. We set up alerting through PagerDuty. For logs we use the ELK stack — Elasticsearch, Logstash, Kibana. Application traces go through OpenTelemetry and land in Jaeger."),
            ("I", "What were the biggest challenges during the migration?"),
            ("R", "Stateful services were painful. Databases don't containerise cleanly, so we ended up using Azure Database for PostgreSQL as a managed service instead of running Postgres in Kubernetes. That decision saved us a lot of operational headache. The other big challenge was networking — VNet peering, private endpoints, firewall rules — that took months to get right."),
            ("I", "How do you handle disaster recovery?"),
            ("R", "We run in two Azure regions — West Europe as primary and North Europe as secondary. Databases replicate asynchronously across regions. For our recovery time objective we target four hours, and recovery point objective is fifteen minutes. We test failover quarterly with a chaos engineering exercise using Azure Chaos Studio."),
            ("I", "What about cost management?"),
            ("R", "Cost was a shock initially. We moved fast and didn't tag resources properly, so we couldn't attribute spend to teams. Now every resource has mandatory tags for team, environment, and project. We use Azure Cost Management with budget alerts. Spot instances cover about forty percent of our non-critical workloads, which cuts compute costs significantly."),
            ("I", "Are there things you would do differently if starting over?"),
            ("R", "Start with a proper landing zone architecture from day one. We bolted on governance — policies, naming conventions, resource locks — after the fact, which meant a painful retrofit. I would also invest earlier in platform engineering. We now have a dedicated platform team, but for the first year developers had to manage too much infrastructure themselves."),
        ],
    },
    {
        "filename": "demo_software_development.docx",
        "turns": [
            ("I", "How is the development process structured?"),
            ("R", "We run two-week sprints using Scrum. Each team has a product owner, a scrum master, and between five and eight engineers. We do sprint planning on Monday, a short daily standup, and a retrospective plus review at the end of each sprint. Backlog refinement happens mid-sprint so planning meetings don't run long."),
            ("I", "How do you manage code quality?"),
            ("R", "Several layers. First, every pull request requires at least two approvals — one from the team and one from a senior engineer outside the team for architectural changes. We use SonarQube for static analysis; it runs in the CI pipeline and blocks merges if coverage drops below eighty percent or if there are critical vulnerabilities."),
            ("I", "What does the CI/CD pipeline look like?"),
            ("R", "GitHub Actions for CI. On every pull request we run unit tests, integration tests, the SonarQube scan, and a container image build. If everything passes the image goes to our artifact registry. ArgoCD then picks up the new image tag and deploys to staging automatically. Production promotion is manual — a one-click approval in our deployment dashboard."),
            ("I", "How do you handle technical debt?"),
            ("R", "We reserve twenty percent of each sprint capacity for tech debt. Teams maintain a debt register in Jira with severity ratings. Anything rated critical blocks new feature work until resolved. It's not perfect — there's always pressure from product — but having the formal allocation means it actually gets done rather than being kicked to a mythical 'hardening sprint.'"),
            ("I", "What testing strategy do you follow?"),
            ("R", "We follow the test pyramid. Lots of unit tests — fast, cheap, run on every commit. Integration tests that cover service boundaries, run on PRs. End-to-end tests in a staging environment, run nightly. We're investing in contract testing with Pact to reduce the integration test surface, because they've become slow as the microservice count grew."),
            ("I", "How do you manage feature releases?"),
            ("R", "Feature flags through LaunchDarkly. Almost every new feature ships behind a flag, which decouples deployment from release. We can deploy to production and enable for internal users first, then gradually roll out to five, twenty, fifty, and then a hundred percent of users. It also makes rollbacks instant — just flip the flag."),
            ("I", "How does the team handle incidents?"),
            ("R", "We follow a blameless post-mortem culture. Every incident above severity two gets a post-mortem within forty-eight hours. We use the five-whys method to find root causes. Action items go into Jira with owners and deadlines. The post-mortems are published internally so everyone can learn from them."),
            ("I", "What tools do developers use day to day?"),
            ("R", "GitHub for source control and code review. Jira for project tracking. Confluence for documentation — though adoption is mixed, people still prefer putting things in READMEs. Slack for communication, with integrations for CI build notifications and PagerDuty alerts. Most engineers use VS Code or JetBrains IDEs depending on the language."),
            ("I", "How do you onboard new engineers?"),
            ("R", "We have a structured thirty-sixty-ninety day plan. First thirty days are about getting familiar with the codebase and pairing with a buddy engineer. By day sixty the new engineer should be independently shipping small features. By day ninety they take on a meaningful piece of work end to end. We also have an internal developer portal built on Backstage that documents all services and their ownership."),
        ],
    },
    {
        "filename": "demo_ai_adoption.docx",
        "turns": [
            ("I", "How has the organisation approached AI and machine learning?"),
            ("R", "It's been gradual. Two years ago we had a small data science team running experiments in Jupyter notebooks, mostly disconnected from production. Now we have a proper ML platform team and several models running in production. The shift happened when leadership saw concrete business value from a recommendation model that increased user retention by twelve percent."),
            ("I", "What does your ML infrastructure look like?"),
            ("R", "We use Databricks as the primary compute platform for training. Data sits in a Snowflake data warehouse. Feature engineering pipelines run on Databricks and write features back to a feature store — we built a lightweight one internally, though we're evaluating Feast. Models are tracked and versioned in MLflow."),
            ("I", "How do you deploy models to production?"),
            ("R", "Trained models get registered in the MLflow registry. Our deployment pipeline packages them as FastAPI services in Docker containers and deploys to Kubernetes via the same GitOps workflow as application services. For high-throughput inference we use NVIDIA Triton Inference Server. Latency-sensitive models are served through that; batch prediction jobs run on Databricks scheduled notebooks."),
            ("I", "Have you integrated large language models?"),
            ("R", "Yes, in the past year that has accelerated significantly. We have a customer support assistant built on GPT-4o that handles first-line queries and reduces ticket volume by about thirty percent. Internally we built a code review assistant that runs on Claude and surfaces potential issues before pull requests go to human review."),
            ("I", "How are you approaching retrieval-augmented generation?"),
            ("R", "RAG has become our default pattern for anything that needs grounding in internal knowledge. We have a knowledge base assistant that ingests internal documentation, Confluence pages, and runbooks into a vector database — we use Chroma for that — and lets engineers ask natural language questions. The answer quality is much higher than pure LLM output because the model is constrained to actual documentation."),
            ("I", "What vector database and embedding model do you use?"),
            ("R", "ChromaDB for the internal knowledge base prototype and Pinecone for production workloads that need scale and persistence guarantees. For embeddings we use OpenAI's text-embedding-3-large model. We evaluated several open-source alternatives but the quality difference for technical text was noticeable."),
            ("I", "What challenges have come up with LLM adoption?"),
            ("R", "Three main ones. First, cost — GPT-4o is expensive at scale, so we route simpler queries to cheaper models automatically. Second, hallucination — we have evaluation pipelines that run reference answers through the model and flag regressions in factual accuracy. Third, latency — streaming responses help perceived performance, but for synchronous use cases the round trip is still too slow for some workflows."),
            ("I", "How do you handle data governance for AI systems?"),
            ("R", "We have a model risk framework that classifies models by impact. High-impact models — anything that affects pricing, hiring, or major customer decisions — require a full review including bias assessment, explainability documentation, and sign-off from legal and compliance. Lower-impact models go through a lighter review. All models have an owner accountable for monitoring and retraining."),
            ("I", "What does model monitoring look like in practice?"),
            ("R", "We track prediction distribution drift using Evidently AI. Data quality checks run before every prediction job using Great Expectations. For LLM outputs we use an LLM-as-judge pattern — a separate model evaluates response quality on a sample of live traffic. Alerts fire when quality scores drop below threshold and trigger a human review cycle."),
            ("I", "Where do you see AI fitting into your engineering workflow in the next year?"),
            ("R", "Code generation is the big one. We're piloting GitHub Copilot across two teams and measuring throughput and defect rates. Early results are positive for boilerplate-heavy work but neutral for complex business logic. We're also investing in automated test generation — having an LLM draft initial unit tests from function signatures saves meaningful time. The honest answer is we're still learning what works and what doesn't."),
        ],
    },
]


def create_interview_doc(path: Path, turns: list) -> None:
    doc = Document()
    for speaker, text in turns:
        doc.add_paragraph(f"{speaker}:\t{text}")
    doc.save(path)


def main():
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    for interview in INTERVIEWS:
        path = data_dir / interview["filename"]
        create_interview_doc(path, interview["turns"])
        print(f"  Created: {path.name}")

    print(f"\nDemo data ready in '{data_dir}/'.")
    print("Next step: python \"vectorstore creator.py\"")


if __name__ == "__main__":
    main()
