from pydantic import BaseModel, model_validator
from typing import List

class HyperparamValidator(BaseModel):
    embeddingDimension: List[int]
    normalizationStrength: List[float]
    iterationWeights: List[List[float]]
    method: List[str]

    @model_validator(mode='before')
    def validate_all(cls, values):
        dims = values.get("embeddingDimension")
        norms = values.get("normalizationStrength")
        weights = values.get("iterationWeights")
        methods = values.get("method")
        allowed = {"cosine", "euclidean"}

        if any(x is None for x in [dims, norms, weights, methods]):
            raise ValueError("Do not allow None values.")

        for param in [dims, norms, weights]:
            if isinstance(param, list):
                if any(isinstance(i, str) for i in param):
                    raise ValueError("Do not allow string values inside lists of numbers.")

        if not isinstance(methods, list) or not all(m in allowed for m in methods):
            raise ValueError(f"Unsupported method(s): {methods}")

        return values
    
#class ParamsParser: