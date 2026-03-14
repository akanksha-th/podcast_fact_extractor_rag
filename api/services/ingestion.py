# from api.db.postgres import 
# from api.db.qdrant import 

class IngestionPipeline:
    def run_ingestion_pipeline(video_id: str, video_url: str, user_id: str):
        ...