import asyncio
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric

from eval.groq_llm import GroqEvalLLM
from sentence_transformers import SentenceTransformer
from api.services.ingestion_pipeline import IngestionPipeline
from api.services.querying import QueryService
from api.db.postgres import create_pool
from api.db.redis import create_redis
from api.db.qdrant import create_qdrant_client
from api.core.config import api_settings

settings = api_settings()

VIDEO_URL = "https://www.youtube.com/watch?v=T30Ul9AiIfs&pp=ygUZamF5IHNoZXR0eSBwb2RjYXN0IGRyZWFtcw%3D%3D"
VIDEO_ID = "T30Ul9AiIfs"
USER_ID = "eval_runner"

TEST_QUESTIONS = [
    "what is the podcast about?",
    "tell me more about this book 'This is why you dream'?",
    "what is the meaning of dreams?",
    "what do they discuss about dreams in the podcast?",
    "are dreams linked with reality?",
    "do dreams have categiories? if yes, what are they?",
]

async def evaluate_test_case(index: int, total: int, test_case: LLMTestCase, metrics: list,) -> tuple[str, dict]:
    """Evaluate all metrics for a single test case fully in parallel."""
    print(f"[{index}/{total}] Starting: '{test_case.input}'")

    async def run_metric(metric) -> tuple[str, dict]:
        metric_name = metric.__class__.__name__
        await metric.a_measure(test_case)
        result = {
            "score": metric.score,
            "passed": metric.is_successful(),
            "reason": metric.reason,
        }
        status = "✅" if result["passed"] else "❌"
        print(f"  [{index}/{total}] {status} {metric_name}: {metric.score:.2f}")
        return metric_name, result

    metric_results = await asyncio.gather(*[run_metric(m) for m in metrics])
    return test_case.input, dict(metric_results)

# ---

async def run_eval():
    await create_pool()
    await create_redis()
    await create_qdrant_client()

    model = SentenceTransformer(settings.embedding_model)
    ingestion_service = IngestionPipeline(model)
    try:
        await ingestion_service.run_ingestion_pipeline(video_id=VIDEO_ID, video_url=VIDEO_URL, user_id=USER_ID)
    except Exception as e:
        raise RuntimeError(str(e))
    query_service = QueryService(model)

    test_cases = []
    for question in TEST_QUESTIONS:
        response = await query_service.get_answer(video_id=VIDEO_ID, user_id=USER_ID, question=question)
        test_cases.append(LLMTestCase(
            input=question,
            actual_output=response["answer"],
            retrieval_context=response["chunk_texts"],
        ))
        await asyncio.sleep(0.5)

    llm = GroqEvalLLM()
    total = len(test_cases)

    def make_metrics():
        return [
            FaithfulnessMetric(model=llm, threshold=0.7),
            AnswerRelevancyMetric(model=llm, threshold=0.7),
            ContextualRelevancyMetric(model=llm, threshold=0.7),
        ]

    tasks = [
        evaluate_test_case(i + 1, total, tc, make_metrics())
        for i, tc in enumerate(test_cases)
    ]
    results = await asyncio.gather(*tasks)
    all_results = dict(results)

    print("\n===== EVALUATION SUMMARY =====")
    for question, scores in all_results.items():
        print(f"\nQ: {question}")
        for metric_name, result in scores.items():
            score_str = f"{result['score']:.2f}" if result["score"] is not None else "N/A"
            status = "✅" if result["passed"] else "❌"
            print(f"  {status} {metric_name}: {score_str}")


if __name__ == "__main__":
    asyncio.run(run_eval())