"""vLLM entry point; registration only, activated by the explicit project architecture."""
def register():
    from vllm import ModelRegistry
    from vllm.model_executor.model_loader import register_model_loader
    from .bank_loader import BankModelLoader
    ModelRegistry.register_model("AparetoBankQwen3_5Moe", "apareto_loader.vllm027.bank_model:BankQwen3_5Moe")
    register_model_loader("apareto_bank")(BankModelLoader)
