from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    openai_api_key: str

    pg_host: str
    pg_port: int = 5432
    pg_user: str
    pg_password: str
    pg_database: str
    pg_table: str = "papers"

    main_llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1800
    embed_model: str = "text-embedding-3-large"
    embed_dimensions: int = 3072

    similarity_cutoff: float = 0.50
    rerank_top_n: int = 6
    cohere_api_key: str = ""
    rerank_model: str = "rerank-v3.5"
    pdf_parser: str = "docling"

    model_config = {"env_file": ".env", "case_sensitive": False}


settings = AppSettings()