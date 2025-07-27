from pydantic import BaseModel, model_validator
from typing import List

class HyperparamValidator(BaseModel):
    embeddingDimension: List[int]
    normalizationStrength: List[float]
    iterationWeights: List[List[float]]
    method: List[str]

    @model_validator(mode='before')
    def check_consistent_lengths(cls, values):
        dims = values.get("embeddingDimension")
        norms = values.get("normalizationStrength")
        weights = values.get("iterationWeights")

        if any(x is None for x in [dims, norms, weights]):
            raise ValueError("Missing required fields for length check.")

        lengths = [len(dims), len(norms), len(weights)]
        if len(set(lengths)) > 1:
            raise ValueError(f"Inconsistent lengths: {lengths}")
        return values

    @model_validator(mode='before')
    def validate_methods(cls, values):
        allowed = {"cosine", "euclidean"}
        methods = values.get("method")
        if not isinstance(methods, list) or not all(m in allowed for m in methods):
            raise ValueError(f"Unsupported method(s): {methods}")
        return values
    
#class ParamsParser: